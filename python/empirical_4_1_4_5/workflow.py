from __future__ import annotations

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from .cli import parse_args
from .config import LEGACY_WTA_PARAMS, RANDOM_SEED
from .data import (
    encode_for_xgboost,
    encode_like_reference,
    get_feature_sets,
    get_subset_data,
    prepare_train_test,
)
from .evaluation import (
    calculate_metrics,
    get_budget_metrics,
    get_ideal_results,
    get_n_household_metrics,
    get_processed_results,
    get_random_baseline_monte_carlo,
    get_random_budget_baseline_monte_carlo,
    run_monte_carlo_random,
    update_eco_results_with_floor,
)
from .models import (
    append_final_wta_column,
    get_group_probabilities,
    train_xgb_model_bayesian,
    train_xgb_regressor_bayesian,
    train_xgb_regressor_fixed,
)
from .r_bridge import (
    run_r_logit,
    run_r_logit_weighted,
    run_r_ologit_wta,
    run_r_ologit_wta_weighted,
    run_r_simulated_predictions,
    run_r_simulated_predictions_weighted,
)
from .simulation import (
    append_metrics,
    clean_pct,
    simulate_policy_knowledge_upgrade,
    summarize_simulation_results,
)
from .tables import (
    build_combined_hyperparameter_table,
    format_metrics_table,
    write_latex_table,
)


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 60 else f"{seconds / 60:.1f} min"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if not args.weighted:
        ans = input("Also run weighted analysis (Appendix G)? [y/N] ").strip().lower()
        args.weighted = ans in ("y", "yes")

    _also_legacy = args.also_legacy
    if not args.use_legacy_wta_params and not args.also_legacy:
        ans = input("Also run legacy WTA comparison? [y/N] ").strip().lower()
        _also_legacy = ans in ("y", "yes")

    if not args.skip_main or args.weighted:
        _run(args, root)

    if _also_legacy:
        legacy_args = copy.copy(args)
        legacy_args.use_legacy_wta_params = True
        legacy_args.weighted = False
        legacy_args.skip_main = False
        legacy_args.output_subdir = args.output_subdir + "_legacy_wta"
        legacy_args.sim_output_subdir = (
            args.sim_output_subdir or "empirical4.5"
        ) + "_legacy_wta"
        print(f"\n{'='*60}")
        print("Re-running analysis with legacy WTA hyperparameters...")
        _run(legacy_args, root)


def _run(args, root: Path) -> None:
    t_total = time.perf_counter()
    output_dir = root / "results" / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root = root / "data" / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / args.output_subdir
    temp_dir.mkdir(parents=True, exist_ok=True)
    sim_output_subdir = args.sim_output_subdir or "empirical4.5"
    use_legacy_wta = args.use_legacy_wta_params

    # Use the post-weights file because it is the most complete intermediate dataset
    # (contains all R-derived variables). The 'weights' column it carries is excluded
    # from every feature set in data.py and is never passed to any model fitter —
    # Sections 4.1–4.4 are unweighted main-text analyses; weighting is Appendix G only.
    input_file = temp_root / "energy_wta_with_post_weights.csv"
    if not input_file.exists():
        raise FileNotFoundError(
            f"{input_file} not found. Run run_all.R first to generate weighted data."
        )

    est_clf = 6 * args.n_trials * 0.11
    est_reg = 6 * (args.reg_n_trials if not args.use_legacy_wta_params else 5) * 0.22
    est_r   = 60.0
    est_sim = 0.0 if args.skip_simulation else args.simulation_iterations * 23.0
    est_total = est_clf + est_reg + est_r + est_sim
    print("Estimated runtime (reference hardware):")
    print(f"  XGBoost training : ~{_fmt(est_clf + est_reg)}")
    print(f"  R models         : ~{_fmt(est_r)}")
    if not args.skip_simulation:
        print(f"  Section 4.5 sim  : ~{_fmt(est_sim)}")
    print(f"  Total            : ~{_fmt(est_total)}")
    print("  (Actual time varies by hardware. Use --skip-simulation to stop after Step 6.)\n")

    print("\n[Step 1/7] Loading and encoding data...")
    _t = time.perf_counter()
    df = pd.read_csv(input_file)
    train_raw, test_raw = prepare_train_test(df, temp_dir)
    train_encoded, test_encoded = encode_for_xgboost(train_raw, test_raw)
    feature_sets = get_feature_sets(train_encoded)
    print(f"  Done ({_fmt(time.perf_counter() - _t)})")

    train_car = get_subset_data(train_encoded, mode="car")
    train_elec = get_subset_data(train_encoded, mode="elec")
    train_green = get_subset_data(train_encoded, mode="green")

    train_tasks = [
        ("Car_Demos", train_car, "y_car", feature_sets["demos_car"], "Demos"),
        ("Car_All", train_car, "y_car", feature_sets["demos_all_car"], "All"),
        ("Elec_Demos", train_elec, "y_elec", feature_sets["demos_elec"], "Demos"),
        ("Elec_All", train_elec, "y_elec", feature_sets["demos_all_elec"], "All"),
        ("Green_Demos", train_green, "y_green", feature_sets["demos_green"], "Demos"),
        ("Green_All", train_green, "y_green", feature_sets["demos_all_green"], "All"),
    ]

    reg_tasks = [
        ("Car_Demos_Reg", train_car, "wta_car", feature_sets["demos_car"], "Demos"),
        ("Car_All_Reg", train_car, "wta_car", feature_sets["demos_all_car"], "All"),
        ("Elec_Demos_Reg", train_elec, "wta_elec", feature_sets["demos_elec"], "Demos"),
        ("Elec_All_Reg", train_elec, "wta_elec", feature_sets["demos_all_elec"], "All"),
        ("Green_Demos_Reg", train_green, "wta_green", feature_sets["demos_green"], "Demos"),
        ("Green_All_Reg", train_green, "wta_green", feature_sets["demos_all_green"], "All"),
    ]


    _skip_main = getattr(args, 'skip_main', False)
    if not _skip_main:
        print(f"\n[Step 2/7] Training PrefAlt XGBoost models...")
        _t = time.perf_counter()
        trained_models: dict[str, xgb.XGBClassifier] = {}
        for _i, (name, train_df, y_col, features, group_type) in enumerate(train_tasks, 1):
            trained_models[name] = train_xgb_model_bayesian(
                train_df[features],
                train_df[y_col],
                group_type=group_type,
                n_trials=args.n_trials,
                cv=args.cv,
            )
        print(f"  Done ({_fmt(time.perf_counter() - _t)})")

        print(f"\n[Step 3/7] Training WTA XGBoost models...")
        _t = time.perf_counter()
        trained_models_reg: dict[str, xgb.XGBRegressor] = {}
        for _i, (name, train_df, y_col, features, group_type) in enumerate(reg_tasks, 1):
            if use_legacy_wta:
                model = train_xgb_regressor_fixed(
                    train_df[features],
                    train_df[y_col],
                    LEGACY_WTA_PARAMS[name],
                )
            else:
                model = train_xgb_regressor_bayesian(
                    train_df[features],
                    train_df[y_col],
                    group_type=group_type,
                    n_trials=args.reg_n_trials,
                    cv=args.cv,
                )
            trained_models_reg[name] = model
        print(f"  Done ({_fmt(time.perf_counter() - _t)})")

        if use_legacy_wta:
            (output_dir / "legacy_params_used.txt").write_text(
                "WTA XGBoost regressors used fixed hyperparameters copied from the original manuscript table.\n"
                "Pref Alt classifiers were trained through the seeded Optuna workflow.\n",
                encoding="utf-8",
            )

        df_params = build_combined_hyperparameter_table(trained_models, trained_models_reg)
        df_params.to_csv(output_dir / "Table_C.1_xgboost_hyperparameters.csv", encoding="utf-8-sig")
        write_latex_table(
            df_params,
            output_dir / "Table_C.1_xgboost_hyperparameters.tex",
            "Table C.1. XGBoost Hyperparameters",
        )

        print("\n[Step 4/7] Computing prediction metrics...")
        _t = time.perf_counter()
        results_ideal = get_ideal_results(test_encoded)
        final_df_ideal = calculate_metrics(results_ideal, results_ideal).iloc[[0]].copy()
        final_df_ideal.index = ["Ideal (Upper Bound)"]

        demos_probs = get_group_probabilities(trained_models, test_encoded, "Demos")
        all_probs = get_group_probabilities(trained_models, test_encoded, "All")
        results_xgb_demos = get_processed_results(demos_probs, test_encoded, threshold=0)
        results_xgb_all = get_processed_results(all_probs, test_encoded, threshold=0)
        final_df_xgb = calculate_metrics(results_xgb_demos, results_xgb_all)
        print(f"  Done ({_fmt(time.perf_counter() - _t)})")

        print("\n[Step 5/7] Running R logit models...")
        _t = time.perf_counter()
        run_r_logit(root, args.rscript, args.output_subdir)
        logit_demos = pd.read_csv(temp_dir / "logit_probs_demos.csv")
        logit_all = pd.read_csv(temp_dir / "logit_probs_all.csv")
        logit_demos.index = test_encoded.index
        logit_all.index = test_encoded.index

        results_eco_demos = get_processed_results(logit_demos, test_encoded, threshold=0)
        results_eco_all = get_processed_results(logit_all, test_encoded, threshold=0)
        final_df_eco = calculate_metrics(results_eco_demos, results_eco_all)
        final_df_random = run_monte_carlo_random(test_encoded, n_iterations=args.random_iterations)
        print(f"  Done ({_fmt(time.perf_counter() - _t)})")

        print("\n[Step 6/7] Generating Sections 4.1–4.4 figures and tables...")
        _t6 = time.perf_counter()

        accuracy_data = {
            "Logistic regression I": clean_pct(final_df_eco.loc["Demos", "Perfect_Rate"]) * 100,
            "Logistic regression II": clean_pct(final_df_eco.loc["All", "Perfect_Rate"]) * 100,
            "XGBoost algorithm I": clean_pct(final_df_xgb.loc["Demos", "Perfect_Rate"]) * 100,
            "XGBoost algorithm II": clean_pct(final_df_xgb.loc["All", "Perfect_Rate"]) * 100,
        }
        accuracy_series = pd.Series(accuracy_data)
        accuracy_series.to_csv(output_dir / "Table_D.2_prediction_accuracy.csv", header=["Accuracy"])

        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        x_pos = np.arange(len(accuracy_series))
        bars = ax.bar(x_pos, accuracy_series.values, 0.4, color="black")
        ax.set_ylim(60, max(accuracy_series.max() + 2, 62))
        ax.set_ylabel("Prediction Accuracy (%)", fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(accuracy_series.index, rotation=15, ha="right", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        fig.tight_layout()
        fig.savefig(output_dir / "Figure_3_prediction_accuracy.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        ordered_keys = ["Random", "Logit_All", "Logit_Demos", "XGB_All", "XGB_Demos", "Ideal"]
        labels = [
            "Random assignment",
            "Logistic regression II",
            "Logistic regression I",
            "XGBoost algorithm II",
            "XGBoost algorithm I",
            "Perfect assignment",
        ]
        raw_plot_data = {
            "Random": {
                "Rate": clean_pct(final_df_random["Accept_Rate"].iloc[0]) * 100,
                "Cost": clean_pct(final_df_random["Accept_Cost"].iloc[0]),
            },
            "Logit_Demos": {
                "Rate": clean_pct(final_df_eco.loc["Demos", "Accept_Rate"]) * 100,
                "Cost": clean_pct(final_df_eco.loc["Demos", "Accept_Cost"]),
            },
            "Logit_All": {
                "Rate": clean_pct(final_df_eco.loc["All", "Accept_Rate"]) * 100,
                "Cost": clean_pct(final_df_eco.loc["All", "Accept_Cost"]),
            },
            "XGB_Demos": {
                "Rate": clean_pct(final_df_xgb.loc["Demos", "Accept_Rate"]) * 100,
                "Cost": clean_pct(final_df_xgb.loc["Demos", "Accept_Cost"]),
            },
            "XGB_All": {
                "Rate": clean_pct(final_df_xgb.loc["All", "Accept_Rate"]) * 100,
                "Cost": clean_pct(final_df_xgb.loc["All", "Accept_Cost"]),
            },
            "Ideal": {
                "Rate": clean_pct(final_df_ideal["Accept_Rate"].iloc[0]) * 100,
                "Cost": clean_pct(final_df_ideal["Accept_Cost"].iloc[0]),
            },
        }
        df_plot = pd.DataFrame([raw_plot_data[key] for key in ordered_keys], index=labels).apply(pd.to_numeric)
        df_plot.to_csv(output_dir / "Table_D.3_assignment_outcomes.csv", encoding="utf-8-sig")
        write_latex_table(
            df_plot,
            output_dir / "Table_D.3_assignment_outcomes.tex",
            "Table D.3. Assignment Outcomes",
        )

        all_metrics = pd.concat(
            [
                format_metrics_table(final_df_random),
                format_metrics_table(final_df_eco),
                format_metrics_table(final_df_xgb),
                format_metrics_table(final_df_ideal),
            ],
            axis=0,
        )
        fig, ax1 = plt.subplots(figsize=(12, 7), dpi=120)
        ax2 = ax1.twinx()
        x_pos = np.arange(len(labels))
        width = 0.4
        x_offset = 0.2
        bars = ax1.bar(
            x_pos,
            df_plot["Cost"],
            width,
            color="#d9d9d9",
            edgecolor="black",
            label="Average Compensation",
        )
        ax2.plot(
            x_pos + x_offset,
            df_plot["Rate"],
            color="#7f7f7f",
            marker="o",
            linestyle="--",
            linewidth=2,
            markersize=8,
            label="Acceptance Rate",
            zorder=5,
        )
        ax1.set_ylim(df_plot["Cost"].min() - 2, df_plot["Cost"].max() + 2)
        ax2.set_ylim(df_plot["Rate"].min() - 2, 100)
        ax1.set_ylabel("Average Compensation (¥)", fontsize=12)
        ax2.set_ylabel("Acceptance Rate (%)", fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=25, ha="right")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)

        for i, bar in enumerate(bars):
            ax1.annotate(
                f"{df_plot['Cost'].iloc[i]:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        for i in range(len(df_plot)):
            y_offset = 15 if i == 4 else 12
            label_x_offset = x_offset + 0.05 if i == 4 else x_offset
            ax2.annotate(
                f"{df_plot['Rate'].iloc[i]:.2f}%",
                xy=(x_pos[i] + label_x_offset, df_plot["Rate"].iloc[i]),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color="#444444",
                weight="bold",
            )

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "Figure_4_assignment_outcomes.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        print("  Running R ordered-logit WTA models...")
        _t = time.perf_counter()
        run_r_ologit_wta(root, args.rscript, args.output_subdir)
        print(f"  Done ({_fmt(time.perf_counter() - _t)})")
        results_xgb_all = append_final_wta_column(results_xgb_all, test_encoded, trained_models_reg, "All")
        results_xgb_demos = append_final_wta_column(results_xgb_demos, test_encoded, trained_models_reg, "Demos")
        results_eco_all = update_eco_results_with_floor(results_eco_all, temp_dir / "wta_preds_all.csv")
        results_eco_demos = update_eco_results_with_floor(results_eco_demos, temp_dir / "wta_preds_demos.csv")

        current_n_list = [55, 100, 145]
        m_ideal = get_n_household_metrics(results_ideal, test_encoded, n_list=current_n_list)
        m_xgb_all = get_n_household_metrics(results_xgb_all, test_encoded, n_list=current_n_list)
        m_xgb_demos = get_n_household_metrics(results_xgb_demos, test_encoded, n_list=current_n_list)
        m_eco_all = get_n_household_metrics(results_eco_all, test_encoded, n_list=current_n_list)
        m_eco_demos = get_n_household_metrics(results_eco_demos, test_encoded, n_list=current_n_list)
        m_random = get_random_baseline_monte_carlo(
            test_encoded, n_list=current_n_list, iterations=args.random_iterations
        )

        m_ideal["Group"] = "1. Ideal (Upper Bound)"
        m_xgb_all["Group"] = "2.1 XGBoost (All)"
        m_xgb_demos["Group"] = "2.2 XGBoost (Demos)"
        m_eco_all["Group"] = "3.1 Ordered Logit (All)"
        m_eco_demos["Group"] = "3.2 Ordered Logit (Demos)"
        m_random["Group"] = "4. Random (1000x Mean)"

        quota_metrics = pd.concat([m_ideal, m_xgb_all, m_xgb_demos, m_eco_all, m_eco_demos, m_random])
        quota_pivot = quota_metrics.pivot(index="Group", columns="Quota (N)")
        quota_cost = quota_pivot.xs("Accept_Cost", axis=1)
        quota_cost.to_csv(output_dir / "Table_D.4_quota_accept_cost.csv", encoding="utf-8-sig")
        write_latex_table(
            quota_cost,
            output_dir / "Table_D.4_quota_accept_cost.tex",
            "Table D.4. Average Compensation by Targeted Number of Households",
        )

        quota_n = quota_cost.columns.tolist()
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(10, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            dpi=120,
        )
        fig.subplots_adjust(hspace=0.1)
        styles_quota = {
            "4. Random (1000x Mean)": {
                "color": "gray",
                "linestyle": ":",
                "marker": "",
                "label": "Random assignment",
            },
            "3.2 Ordered Logit (Demos)": {
                "color": "gray",
                "linestyle": "-",
                "marker": "o",
                "label": "Logistic regression I",
            },
            "2.2 XGBoost (Demos)": {
                "color": "black",
                "linestyle": "-",
                "marker": "o",
                "label": "XGBoost algorithm I",
            },
            "3.1 Ordered Logit (All)": {
                "color": "gray",
                "linestyle": "-.",
                "marker": "^",
                "label": "Logistic regression II",
            },
            "2.1 XGBoost (All)": {
                "color": "black",
                "linestyle": "-.",
                "marker": "^",
                "label": "XGBoost algorithm II",
            },
            "1. Ideal (Upper Bound)": {
                "color": "black",
                "linestyle": "--",
                "marker": "",
                "label": "Perfect assignment",
            },
        }
        for group_name, style in styles_quota.items():
            if group_name in quota_cost.index:
                y_values = quota_cost.loc[group_name].values
                ax_top.plot(quota_n, y_values, **style, linewidth=1.5, markersize=7)
                ax_bottom.plot(quota_n, y_values, **style, linewidth=1.5, markersize=7)

        ax_top.set_ylim(20, m_random.loc[0, "Accept_Cost"] + 2)
        ax_bottom.set_ylim(0, m_ideal.loc[0, "Accept_Cost"] + 5)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_bottom.spines["right"].set_visible(False)
        ax_top.tick_params(labeltop=False)
        ax_bottom.xaxis.tick_bottom()
        d = 0.015
        kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bottom.transAxes)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        fig.text(
            0.04,
            0.5,
            "Compensation Spending (¥/month/household)",
            va="center",
            rotation="vertical",
            fontsize=12,
        )
        ax_bottom.set_xlabel("Targeted Numbers of Households", fontsize=12)
        ax_bottom.set_xticks(quota_n)
        ax_top.legend(loc="center left", bbox_to_anchor=(1.05, 0.3), frameon=False, fontsize=11)
        fig.savefig(output_dir / "Figure_5_quota_compensation.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        current_budgets = [1000, 2000, 3000, 4000]
        bm_ideal = get_budget_metrics(results_ideal, test_encoded, budget_list=current_budgets)
        bm_xgb_all = get_budget_metrics(results_xgb_all, test_encoded, budget_list=current_budgets)
        bm_xgb_demos = get_budget_metrics(results_xgb_demos, test_encoded, budget_list=current_budgets)
        bm_eco_all = get_budget_metrics(results_eco_all, test_encoded, budget_list=current_budgets)
        bm_eco_demos = get_budget_metrics(results_eco_demos, test_encoded, budget_list=current_budgets)
        bm_random = get_random_budget_baseline_monte_carlo(
            test_encoded, current_budgets, iterations=args.random_iterations
        )

        bm_ideal["Group"] = "1. Ideal (Theoretical Max)"
        bm_xgb_all["Group"] = "2.1 XGBoost (All Features)"
        bm_xgb_demos["Group"] = "2.2 XGBoost (Demos Only)"
        bm_eco_all["Group"] = "3.1 Ordered Logit (All Features)"
        bm_eco_demos["Group"] = "3.2 Ordered Logit (Demos Only)"
        bm_random["Group"] = "4. Random (1000x Mean)"

        budget_final = pd.concat([bm_ideal, bm_xgb_all, bm_xgb_demos, bm_eco_all, bm_eco_demos, bm_random])
        budget_pivot = budget_final.pivot(index="Group", columns="Budget_Limit")
        budget_pivot[("Accept_Rate", "Average")] = budget_pivot["Accept_Rate"].mean(axis=1)

        budget_table = budget_pivot["Total_Recruited (N)"].copy()
        budget_table["Average Acceptance Rate"] = budget_pivot[("Accept_Rate", "Average")]
        budget_table.to_csv(output_dir / "Table_D.5_budget_participation.csv", encoding="utf-8-sig")
        write_latex_table(
            budget_table,
            output_dir / "Table_D.5_budget_participation.tex",
            "Table D.5. Budget-Constrained Participation and Average Acceptance Rate",
        )

        plot_df = budget_pivot["Total_Recruited (N)"]
        avg_rates = budget_pivot[("Accept_Rate", "Average")]
        fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
        styles_budget = {
            "1. Ideal (Theoretical Max)": {
                "color": "black",
                "linestyle": "--",
                "marker": "",
                "label_base": "Perfect assignment",
            },
            "2.2 XGBoost (Demos Only)": {
                "color": "black",
                "linestyle": "-.",
                "marker": "^",
                "label_base": "XGBoost algorithm I",
            },
            "2.1 XGBoost (All Features)": {
                "color": "black",
                "linestyle": "-",
                "marker": "o",
                "label_base": "XGBoost algorithm II",
            },
            "3.2 Ordered Logit (Demos Only)": {
                "color": "gray",
                "linestyle": "-.",
                "marker": "^",
                "label_base": "Logistic regression I",
            },
            "3.1 Ordered Logit (All Features)": {
                "color": "gray",
                "linestyle": "-",
                "marker": "o",
                "label_base": "Logistic regression II",
            },
            "4. Random (1000x Mean)": {
                "color": "gray",
                "linestyle": ":",
                "marker": "",
                "label_base": "Random assignment",
            },
        }
        for group_name, style in styles_budget.items():
            if group_name in plot_df.index:
                y_values = plot_df.loc[group_name].values
                label = f"{style['label_base']} ({avg_rates.loc[group_name]:.2%})"
                ax.plot(
                    current_budgets,
                    y_values,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    label=label,
                    linewidth=1.8,
                    markersize=8,
                )
        ax.set_xlabel("Mitigation Budget (¥/month)", fontsize=12)
        ax.set_ylabel("Number of Participating Households", fontsize=12)
        ax.set_xticks(current_budgets)
        ax.set_xlim(min(current_budgets) - 200, max(current_budgets) + 200)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(plot_df.min().min() - 5, plot_df.max().max() + 5)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=11)
        fig.tight_layout()
        fig.savefig(output_dir / "Figure_6_budget_participation.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f"  Done ({_fmt(time.perf_counter() - _t6)})")

    # When main steps were skipped, recompute the lightweight variables the
    # weighted block depends on (ideal results and fixed quota/budget lists).
    if _skip_main:
        results_ideal = get_ideal_results(test_encoded)
        final_df_ideal = calculate_metrics(results_ideal, results_ideal).iloc[[0]].copy()
        current_n_list = [55, 100, 145]
        current_budgets = [1000, 2000, 3000, 4000]

    # ── Weighted Sections 4.1–4.4 analysis ────────────────────────────────────
    if args.weighted:
        _wN = 4 if args.skip_simulation else 6
        w_subdir = args.weighted_output_subdir
        w_sim_subdir = sim_output_subdir + "_weighted"
        w_output_dir = root / "results" / w_subdir
        w_output_dir.mkdir(parents=True, exist_ok=True)
        w_temp_dir = temp_root / w_subdir
        w_temp_dir.mkdir(parents=True, exist_ok=True)

        # Float-precision weights from unencoded data (train_encoded.astype(int) would truncate floats)
        w_car_arr   = train_raw.loc[train_car.index,   "weights"].values.astype(float)
        w_car_arr  /= w_car_arr.mean()
        w_elec_arr  = train_raw.loc[train_elec.index,  "weights"].values.astype(float)
        w_elec_arr /= w_elec_arr.mean()
        w_green_arr = train_raw.loc[train_green.index, "weights"].values.astype(float)
        w_green_arr/= w_green_arr.mean()
        _w_by_energy = {"Car": w_car_arr, "Elec": w_elec_arr, "Green": w_green_arr}

        print(f"\n[Weighted 1/{_wN}] Training weighted PrefAlt XGBoost models...", end="  ", flush=True)
        _t = time.perf_counter()
        w_trained_models: dict[str, xgb.XGBClassifier] = {}
        for _wname, _wdf, _wyc, _wfeat, _wgt in train_tasks:
            _wenergy = _wname.split("_")[0]
            w_trained_models[_wname] = train_xgb_model_bayesian(
                _wdf[_wfeat], _wdf[_wyc],
                group_type=_wgt, n_trials=args.weighted_n_trials, cv=args.cv,
                sample_weight=_w_by_energy[_wenergy], weighted=True,
            )
        print(f"Done ({_fmt(time.perf_counter() - _t)})")

        print(f"[Weighted 2/{_wN}] Training weighted WTA XGBoost models...", end="  ", flush=True)
        _t = time.perf_counter()
        w_trained_models_reg: dict[str, xgb.XGBRegressor] = {}
        for _wname, _wdf, _wyc, _wfeat, _wgt in reg_tasks:
            _wenergy = _wname.split("_")[0]
            w_trained_models_reg[_wname] = train_xgb_regressor_bayesian(
                _wdf[_wfeat], _wdf[_wyc],
                group_type=_wgt, n_trials=args.weighted_reg_n_trials, cv=args.cv,
                sample_weight=_w_by_energy[_wenergy], weighted=True,
            )
        print(f"Done ({_fmt(time.perf_counter() - _t)})")

        print(f"[Weighted 3/{_wN}] Running weighted R logit and WTA models...", end="  ", flush=True)
        _t = time.perf_counter()
        w_dprobs = get_group_probabilities(w_trained_models, test_encoded, "Demos")
        w_aprobs = get_group_probabilities(w_trained_models, test_encoded, "All")
        w_rxgb_d = get_processed_results(w_dprobs, test_encoded, threshold=0)
        w_rxgb_a = get_processed_results(w_aprobs, test_encoded, threshold=0)
        w_fxgb   = calculate_metrics(w_rxgb_d, w_rxgb_a)
        run_r_logit_weighted(root, args.rscript, args.output_subdir, w_subdir)
        w_ld = pd.read_csv(w_temp_dir / "logit_probs_demos.csv")
        w_la = pd.read_csv(w_temp_dir / "logit_probs_all.csv")
        w_ld.index = test_encoded.index
        w_la.index = test_encoded.index
        w_reco_d  = get_processed_results(w_ld, test_encoded, threshold=0)
        w_reco_a  = get_processed_results(w_la, test_encoded, threshold=0)
        w_feco    = calculate_metrics(w_reco_d, w_reco_a)
        w_frandom = run_monte_carlo_random(test_encoded, n_iterations=args.random_iterations)
        run_r_ologit_wta_weighted(root, args.rscript, args.output_subdir, w_subdir)
        w_rxgb_a = append_final_wta_column(w_rxgb_a, test_encoded, w_trained_models_reg, "All")
        w_rxgb_d = append_final_wta_column(w_rxgb_d, test_encoded, w_trained_models_reg, "Demos")
        w_reco_a = update_eco_results_with_floor(w_reco_a, w_temp_dir / "wta_preds_all.csv")
        w_reco_d = update_eco_results_with_floor(w_reco_d, w_temp_dir / "wta_preds_demos.csv")
        print(f"Done ({_fmt(time.perf_counter() - _t)})")

        print(f"[Weighted 4/{_wN}] Generating weighted Sections 4.1–4.4 figures (G.4–G.6)...", end="  ", flush=True)
        _t = time.perf_counter()

        # G.4 — assignment outcomes (also defines w_df_plot for G.7 baseline)
        _wkeys = ["Random", "Logit_All", "Logit_Demos", "XGB_All", "XGB_Demos", "Ideal"]
        _wlbls = ["Random assignment", "Logistic regression II", "Logistic regression I",
                  "XGBoost algorithm II", "XGBoost algorithm I", "Perfect assignment"]
        _wpd = {
            "Random":      {"Rate": clean_pct(w_frandom["Accept_Rate"].iloc[0]) * 100, "Cost": clean_pct(w_frandom["Accept_Cost"].iloc[0])},
            "Logit_Demos": {"Rate": clean_pct(w_feco.loc["Demos", "Accept_Rate"]) * 100, "Cost": clean_pct(w_feco.loc["Demos", "Accept_Cost"])},
            "Logit_All":   {"Rate": clean_pct(w_feco.loc["All",  "Accept_Rate"]) * 100, "Cost": clean_pct(w_feco.loc["All",  "Accept_Cost"])},
            "XGB_Demos":   {"Rate": clean_pct(w_fxgb.loc["Demos", "Accept_Rate"]) * 100, "Cost": clean_pct(w_fxgb.loc["Demos", "Accept_Cost"])},
            "XGB_All":     {"Rate": clean_pct(w_fxgb.loc["All",  "Accept_Rate"]) * 100, "Cost": clean_pct(w_fxgb.loc["All",  "Accept_Cost"])},
            "Ideal":       {"Rate": clean_pct(final_df_ideal["Accept_Rate"].iloc[0]) * 100, "Cost": clean_pct(final_df_ideal["Accept_Cost"].iloc[0])},
        }
        w_df_plot = pd.DataFrame([_wpd[k] for k in _wkeys], index=_wlbls).apply(pd.to_numeric)
        w_df_plot.to_csv(w_output_dir / "Figure_G.4_assignment_outcomes_data.csv", encoding="utf-8-sig")
        fig, ax1 = plt.subplots(figsize=(12, 7), dpi=120)
        ax2 = ax1.twinx()
        xp = np.arange(len(_wlbls)); wid = 0.4; xoff = 0.2
        bs = ax1.bar(xp, w_df_plot["Cost"], wid, color="#d9d9d9", edgecolor="black", label="Average Compensation")
        ax2.plot(xp + xoff, w_df_plot["Rate"], color="#7f7f7f", marker="o", linestyle="--", linewidth=2, markersize=8, label="Acceptance Rate", zorder=5)
        ax1.set_ylim(w_df_plot["Cost"].min() - 2, w_df_plot["Cost"].max() + 2)
        ax2.set_ylim(w_df_plot["Rate"].min() - 2, 100)
        ax1.set_ylabel("Average Compensation (¥)", fontsize=12)
        ax2.set_ylabel("Acceptance Rate (%)", fontsize=12)
        ax1.set_xticks(xp); ax1.set_xticklabels(_wlbls, rotation=25, ha="right")
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False); ax2.spines["left"].set_visible(False)
        for i, b in enumerate(bs):
            ax1.annotate(f"{w_df_plot['Cost'].iloc[i]:.2f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                         xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=10)
        for i in range(len(w_df_plot)):
            yo = 15 if i == 4 else 12; lxo = xoff + 0.05 if i == 4 else xoff
            ax2.annotate(f"{w_df_plot['Rate'].iloc[i]:.2f}%", xy=(xp[i] + lxo, w_df_plot["Rate"].iloc[i]),
                         xytext=(0, yo), textcoords="offset points", ha="center", fontsize=10, color="#444444", weight="bold")
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(w_output_dir / "Figure_G.4_assignment_outcomes.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        # G.5 — quota compensation
        wm_id = get_n_household_metrics(results_ideal,  test_encoded, n_list=current_n_list)
        wm_xa = get_n_household_metrics(w_rxgb_a, test_encoded, n_list=current_n_list)
        wm_xd = get_n_household_metrics(w_rxgb_d, test_encoded, n_list=current_n_list)
        wm_ea = get_n_household_metrics(w_reco_a, test_encoded, n_list=current_n_list)
        wm_ed = get_n_household_metrics(w_reco_d, test_encoded, n_list=current_n_list)
        wm_rn = get_random_baseline_monte_carlo(test_encoded, n_list=current_n_list, iterations=args.random_iterations)
        wm_id["Group"] = "1. Ideal (Upper Bound)"; wm_xa["Group"] = "2.1 XGBoost (All)"; wm_xd["Group"] = "2.2 XGBoost (Demos)"
        wm_ea["Group"] = "3.1 Ordered Logit (All)"; wm_ed["Group"] = "3.2 Ordered Logit (Demos)"; wm_rn["Group"] = "4. Random (1000x Mean)"
        wq_piv = pd.concat([wm_id, wm_xa, wm_xd, wm_ea, wm_ed, wm_rn]).pivot(index="Group", columns="Quota (N)")
        wq_cost = wq_piv.xs("Accept_Cost", axis=1)
        wq_cost.to_csv(w_output_dir / "Figure_G.5_quota_accept_cost_data.csv", encoding="utf-8-sig")
        wqn = wq_cost.columns.tolist()
        fig, (atop, abot) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, dpi=120)
        fig.subplots_adjust(hspace=0.1)
        _sq = {
            "4. Random (1000x Mean)":    {"color": "gray",  "linestyle": ":",  "marker": "",  "label": "Random assignment"},
            "3.2 Ordered Logit (Demos)": {"color": "gray",  "linestyle": "-",  "marker": "o", "label": "Logistic regression I"},
            "2.2 XGBoost (Demos)":       {"color": "black", "linestyle": "-",  "marker": "o", "label": "XGBoost algorithm I"},
            "3.1 Ordered Logit (All)":   {"color": "gray",  "linestyle": "-.", "marker": "^", "label": "Logistic regression II"},
            "2.1 XGBoost (All)":         {"color": "black", "linestyle": "-.", "marker": "^", "label": "XGBoost algorithm II"},
            "1. Ideal (Upper Bound)":    {"color": "black", "linestyle": "--", "marker": "",  "label": "Perfect assignment"},
        }
        for gn, st in _sq.items():
            if gn in wq_cost.index:
                for _ax in (atop, abot):
                    _ax.plot(wqn, wq_cost.loc[gn].values, **st, linewidth=1.5, markersize=7)
        atop.set_ylim(min(wm_xd.loc[0, "Accept_Cost"], wm_xa.loc[0, "Accept_Cost"]) - 1.5, wm_rn.loc[0, "Accept_Cost"] + 2)
        abot.set_ylim(0, wm_id.loc[0, "Accept_Cost"] + 5)
        atop.spines["bottom"].set_visible(False); atop.spines["top"].set_visible(False); atop.spines["right"].set_visible(False)
        abot.spines["top"].set_visible(False); abot.spines["right"].set_visible(False)
        atop.tick_params(labeltop=False); abot.xaxis.tick_bottom()
        d = 0.015
        atop.plot((-d, +d), (-d, +d), transform=atop.transAxes, color="black", clip_on=False)
        abot.plot((-d, +d), (1 - d, 1 + d), transform=abot.transAxes, color="black", clip_on=False)
        fig.text(0.04, 0.5, "Compensation Spending (¥/month/household)", va="center", rotation="vertical", fontsize=12)
        abot.set_xlabel("Targeted Numbers of Households", fontsize=12); abot.set_xticks(wqn)
        atop.legend(loc="center left", bbox_to_anchor=(1.05, 0.3), frameon=False, fontsize=11)
        fig.savefig(w_output_dir / "Figure_G.5_quota_compensation.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        # G.6 — budget participation
        wb_id = get_budget_metrics(results_ideal, test_encoded, budget_list=current_budgets)
        wb_xa = get_budget_metrics(w_rxgb_a, test_encoded, budget_list=current_budgets)
        wb_xd = get_budget_metrics(w_rxgb_d, test_encoded, budget_list=current_budgets)
        wb_ea = get_budget_metrics(w_reco_a, test_encoded, budget_list=current_budgets)
        wb_ed = get_budget_metrics(w_reco_d, test_encoded, budget_list=current_budgets)
        wb_rn = get_random_budget_baseline_monte_carlo(test_encoded, current_budgets, iterations=args.random_iterations)
        wb_id["Group"] = "1. Ideal (Theoretical Max)"; wb_xa["Group"] = "2.1 XGBoost (All Features)"; wb_xd["Group"] = "2.2 XGBoost (Demos Only)"
        wb_ea["Group"] = "3.1 Ordered Logit (All Features)"; wb_ed["Group"] = "3.2 Ordered Logit (Demos Only)"; wb_rn["Group"] = "4. Random (1000x Mean)"
        wb_piv = pd.concat([wb_id, wb_xa, wb_xd, wb_ea, wb_ed, wb_rn]).pivot(index="Group", columns="Budget_Limit")
        wb_piv[("Accept_Rate", "Average")] = wb_piv["Accept_Rate"].mean(axis=1)
        wb_pdf = wb_piv["Total_Recruited (N)"]; wb_avg = wb_piv[("Accept_Rate", "Average")]
        wb_pdf.to_csv(w_output_dir / "Figure_G.6_budget_participation_data.csv", encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
        _sb = {
            "1. Ideal (Theoretical Max)":     {"color": "black", "linestyle": "--", "marker": "",  "label_base": "Perfect assignment"},
            "2.2 XGBoost (Demos Only)":       {"color": "black", "linestyle": "-.", "marker": "^", "label_base": "XGBoost algorithm I"},
            "2.1 XGBoost (All Features)":     {"color": "black", "linestyle": "-",  "marker": "o", "label_base": "XGBoost algorithm II"},
            "3.2 Ordered Logit (Demos Only)": {"color": "gray",  "linestyle": "-.", "marker": "^", "label_base": "Logistic regression I"},
            "3.1 Ordered Logit (All Features)":{"color": "gray", "linestyle": "-",  "marker": "o", "label_base": "Logistic regression II"},
            "4. Random (1000x Mean)":         {"color": "gray",  "linestyle": ":",  "marker": "",  "label_base": "Random assignment"},
        }
        for gn, st in _sb.items():
            if gn in wb_pdf.index:
                ax.plot(current_budgets, wb_pdf.loc[gn].values, color=st["color"], linestyle=st["linestyle"],
                        marker=st["marker"], label=f"{st['label_base']} ({wb_avg.loc[gn]:.2%})", linewidth=1.8, markersize=8)
        ax.set_xlabel("Mitigation Budget (¥/month)", fontsize=12); ax.set_ylabel("Number of Participating Households", fontsize=12)
        ax.set_xticks(current_budgets); ax.set_xlim(min(current_budgets) - 200, max(current_budgets) + 200)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_ylim(wb_pdf.min().min() - 5, wb_pdf.max().max() + 5)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=11)
        fig.tight_layout()
        fig.savefig(w_output_dir / "Figure_G.6_budget_participation.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

        print(f"Done ({_fmt(time.perf_counter() - _t)})")
        print(f"  Weighted Sections 4.1–4.4 results saved to {w_output_dir}")

    test_base = test_raw.copy().reset_index(drop=True)

    # ── Weighted Section 4.5 simulation ──────────────────────────────────────
    if args.weighted:
        w_sim_dir = root / "results" / w_sim_subdir
        w_sim_dir.mkdir(parents=True, exist_ok=True)
        w_sim_temp_dir = temp_root / w_sim_subdir
        w_sim_temp_dir.mkdir(parents=True, exist_ok=True)
        w_simulate_results = {
            "ideal_sm": {}, "random_sm": {}, "xgb_all_sm": {},
            "xgb_demos_sm": {}, "eco_all_sm": {}, "eco_demos_sm": {},
        }
        _wt_total = args.simulation_iterations
        print(f"\n[Weighted 5/6] Weighted Section 4.5 simulation ({_wt_total} iterations)...")
        _tw7 = time.perf_counter()
        for i in range(_wt_total):
            _we = time.perf_counter() - _tw7
            _weta = "--" if i == 0 else _fmt(_we / i * (_wt_total - i))
            print(f"  Iteration {i + 1:4d}/{_wt_total}  (ETA: {_weta})     ", end="\r", flush=True)
            np.random.seed(RANDOM_SEED + i)
            wsim_raw = simulate_policy_knowledge_upgrade(test_base.copy())
            wsim_raw.to_csv(w_sim_temp_dir / "test_data_simulated.csv", index=False, encoding="utf-8-sig")
            run_r_simulated_predictions_weighted(root, args.rscript, args.output_subdir, w_sim_subdir)
            wl_demos = pd.read_csv(w_sim_temp_dir / "logit_probs_demos_simulated.csv")
            wl_all   = pd.read_csv(w_sim_temp_dir / "logit_probs_all_simulated.csv")
            wl_demos.index = wsim_raw.index
            wl_all.index   = wsim_raw.index
            weco_d = get_processed_results(wl_demos, wsim_raw, threshold=0)
            weco_a = get_processed_results(wl_all,   wsim_raw, threshold=0)
            wfdf_eco = calculate_metrics(weco_d, weco_a)
            wsim_enc = encode_like_reference(wsim_raw, test_encoded.columns)
            wfdf_ideal  = calculate_metrics(get_ideal_results(wsim_enc), get_ideal_results(wsim_enc)).iloc[[0]].copy()
            wfdf_random = run_monte_carlo_random(wsim_enc, n_iterations=args.random_iterations)
            wdp = get_group_probabilities(w_trained_models, wsim_enc, "Demos")
            wap = get_group_probabilities(w_trained_models, wsim_enc, "All")
            wfdf_xgb = calculate_metrics(
                get_processed_results(wdp, wsim_enc, threshold=0),
                get_processed_results(wap, wsim_enc, threshold=0),
            )
            append_metrics(w_simulate_results["ideal_sm"],     wfdf_ideal,  0)
            append_metrics(w_simulate_results["random_sm"],    wfdf_random, 0)
            append_metrics(w_simulate_results["xgb_demos_sm"], wfdf_xgb,   0)
            append_metrics(w_simulate_results["xgb_all_sm"],   wfdf_xgb,   1)
            append_metrics(w_simulate_results["eco_demos_sm"],  wfdf_eco,   0)
            append_metrics(w_simulate_results["eco_all_sm"],    wfdf_eco,   1)
        print()
        print(f"  Weighted simulation done ({_fmt(time.perf_counter() - _tw7)})")
        w_sim_summary = summarize_simulation_results(w_simulate_results)
        w_sim_summary.to_csv(w_sim_dir / "Figure_G.7_knowledge_growth_summary.csv", encoding="utf-8-sig")

        print(f"[Weighted 6/6] Generating weighted Section 4.5 figure (G.7)...", end="  ", flush=True)
        _t = time.perf_counter()
        w_sim_order = [
            ("random_sm", "Random assignment"), ("eco_all_sm", "Logistic regression II"),
            ("eco_demos_sm", "Logistic regression I"), ("xgb_all_sm", "XGBoost algorithm II"),
            ("xgb_demos_sm", "XGBoost algorithm I"), ("ideal_sm", "Perfect assignment"),
        ]
        w_sim_labels = [lb for _, lb in w_sim_order]
        w_bl_cost = w_df_plot.loc[w_sim_labels, "Cost"].values
        w_bl_rate = w_df_plot.loc[w_sim_labels, "Rate"].values
        w_cf_cost     = np.array([w_sim_summary.loc[k, "Cost_Mean"] for k, _ in w_sim_order])
        w_cf_cost_err = np.array([2 * w_sim_summary.loc[k, "Cost_SD"] for k, _ in w_sim_order])
        w_cf_rate     = np.array([w_sim_summary.loc[k, "Rate_Mean"] for k, _ in w_sim_order])
        w_cf_rate_low = np.array([w_sim_summary.loc[k, "Rate_Low"]  for k, _ in w_sim_order])
        w_cf_rate_hi  = np.array([w_sim_summary.loc[k, "Rate_High"] for k, _ in w_sim_order])
        fig, ax1 = plt.subplots(figsize=(14, 7), dpi=120)
        ax2 = ax1.twinx()
        xp = np.arange(len(w_sim_labels))
        wd = 0.32
        ax1.bar(xp - wd / 2, w_bl_cost, wd, color="#d9d9d9", edgecolor="black", label="Benchmark Average Compensation")
        ax1.bar(xp + wd / 2, w_cf_cost, wd, color="#666666", edgecolor="black",
                yerr=w_cf_cost_err, capsize=4, label="Counterfactual Average Compensation")
        ax2.plot(xp, w_bl_rate,  color="gray",  linestyle="--", marker="o", label="Benchmark Acceptance Rate")
        ax2.fill_between(xp, w_cf_rate_low, w_cf_rate_hi, color="gray", alpha=0.2, label="95% Confidence Interval")
        ax2.plot(xp, w_cf_rate, color="black", linestyle="-",  marker="s", label="Counterfactual Acceptance Rate")
        for idx, val in enumerate(w_bl_cost):
            ax1.annotate(f"{val:.2f}", xy=(idx - wd / 2, val), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
        for idx, val in enumerate(w_cf_cost):
            ax1.annotate(f"{val:.2f}", xy=(idx + wd / 2, val), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
        for idx, val in enumerate(w_cf_rate):
            ax2.annotate(f"{val:.2f}%", xy=(idx, val), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, weight="bold")
        ax1.set_ylim(15, max(w_bl_cost.max(), w_cf_cost.max()) + 2)
        ax1.set_ylabel("Average Compensation (¥)", fontsize=12)
        ax2.set_ylabel("Acceptance Rate (%)", fontsize=12)
        ax1.set_xticks(xp); ax1.set_xticklabels(w_sim_labels, rotation=15, ha="right")
        ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(w_sim_dir / "Figure_G.7_knowledge_growth.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Done ({_fmt(time.perf_counter() - _t)})")
        print(f"  Weighted Section 4.5 results saved to {w_sim_dir}")

    if args.skip_simulation or _skip_main:
        print(f"\nAll done! Results saved to {output_dir} (total time: {_fmt(time.perf_counter() - t_total)})")
        return

    sim_dir = root / "results" / sim_output_subdir
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim_temp_dir = temp_root / sim_output_subdir
    sim_temp_dir.mkdir(parents=True, exist_ok=True)
    simulate_results = {
        "ideal_sm": {},
        "random_sm": {},
        "xgb_all_sm": {},
        "xgb_demos_sm": {},
        "eco_all_sm": {},
        "eco_demos_sm": {},
    }

    _total_iters = args.simulation_iterations
    print(f"\n[Step 7/7] Section 4.5 knowledge-growth simulation ({_total_iters} iterations)...")
    _t7 = time.perf_counter()
    for i in range(_total_iters):
        _elapsed = time.perf_counter() - _t7
        _eta_str = "--" if i == 0 else _fmt(_elapsed / i * (_total_iters - i))
        print(f"  Iteration {i + 1:4d}/{_total_iters}  (ETA: {_eta_str})     ", end="\r", flush=True)

        np.random.seed(RANDOM_SEED + i)
        simulated_raw = simulate_policy_knowledge_upgrade(test_base.copy())
        simulated_raw.to_csv(sim_temp_dir / "test_data_simulated.csv", index=False, encoding="utf-8-sig")

        run_r_simulated_predictions(root, args.rscript, args.output_subdir, sim_output_subdir)
        logit_demos_sm = pd.read_csv(sim_temp_dir / "logit_probs_demos_simulated.csv")
        logit_all_sm = pd.read_csv(sim_temp_dir / "logit_probs_all_simulated.csv")
        logit_demos_sm.index = simulated_raw.index
        logit_all_sm.index = simulated_raw.index
        results_eco_demos_sm = get_processed_results(logit_demos_sm, simulated_raw, threshold=0)
        results_eco_all_sm = get_processed_results(logit_all_sm, simulated_raw, threshold=0)
        final_df_eco_sm = calculate_metrics(results_eco_demos_sm, results_eco_all_sm)

        simulated_encoded = encode_like_reference(simulated_raw, test_encoded.columns)
        results_ideal_sm = get_ideal_results(simulated_encoded)
        final_df_ideal_sm = calculate_metrics(results_ideal_sm, results_ideal_sm).iloc[[0]].copy()
        final_df_random_sm = run_monte_carlo_random(
            simulated_encoded, n_iterations=args.random_iterations
        )
        demos_probs_sm = get_group_probabilities(trained_models, simulated_encoded, "Demos")
        all_probs_sm = get_group_probabilities(trained_models, simulated_encoded, "All")
        results_xgb_demos_sm = get_processed_results(demos_probs_sm, simulated_encoded, threshold=0)
        results_xgb_all_sm = get_processed_results(all_probs_sm, simulated_encoded, threshold=0)
        final_df_xgb_sm = calculate_metrics(results_xgb_demos_sm, results_xgb_all_sm)

        append_metrics(simulate_results["ideal_sm"], final_df_ideal_sm, 0)
        append_metrics(simulate_results["random_sm"], final_df_random_sm, 0)
        append_metrics(simulate_results["xgb_demos_sm"], final_df_xgb_sm, 0)
        append_metrics(simulate_results["xgb_all_sm"], final_df_xgb_sm, 1)
        append_metrics(simulate_results["eco_demos_sm"], final_df_eco_sm, 0)
        append_metrics(simulate_results["eco_all_sm"], final_df_eco_sm, 1)

    print()
    print(f"  Simulation done ({_fmt(time.perf_counter() - _t7)})")
    sim_summary = summarize_simulation_results(simulate_results)
    sim_summary.to_csv(sim_dir / "Figure_7_knowledge_growth_summary.csv", encoding="utf-8-sig")

    sim_order = [
        ("random_sm", "Random assignment"),
        ("eco_all_sm", "Logistic regression II"),
        ("eco_demos_sm", "Logistic regression I"),
        ("xgb_all_sm", "XGBoost algorithm II"),
        ("xgb_demos_sm", "XGBoost algorithm I"),
        ("ideal_sm", "Perfect assignment"),
    ]
    sim_labels = [label for _, label in sim_order]
    baseline_cost = df_plot.loc[sim_labels, "Cost"].values
    baseline_rate = df_plot.loc[sim_labels, "Rate"].values
    cf_cost = np.array([sim_summary.loc[key, "Cost_Mean"] for key, _ in sim_order])
    cf_cost_err = np.array([2 * sim_summary.loc[key, "Cost_SD"] for key, _ in sim_order])
    cf_rate = np.array([sim_summary.loc[key, "Rate_Mean"] for key, _ in sim_order])
    cf_rate_low = np.array([sim_summary.loc[key, "Rate_Low"] for key, _ in sim_order])
    cf_rate_high = np.array([sim_summary.loc[key, "Rate_High"] for key, _ in sim_order])

    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=120)
    ax2 = ax1.twinx()
    x_pos = np.arange(len(sim_labels))
    width = 0.32
    ax1.bar(
        x_pos - width / 2,
        baseline_cost,
        width,
        color="#d9d9d9",
        edgecolor="black",
        label="Benchmark Average Compensation",
    )
    ax1.bar(
        x_pos + width / 2,
        cf_cost,
        width,
        color="#666666",
        edgecolor="black",
        yerr=cf_cost_err,
        capsize=4,
        label="Counterfactual Average Compensation",
    )
    ax2.plot(
        x_pos,
        baseline_rate,
        color="gray",
        linestyle="--",
        marker="o",
        label="Benchmark Acceptance Rate",
    )
    ax2.fill_between(
        x_pos,
        cf_rate_low,
        cf_rate_high,
        color="gray",
        alpha=0.2,
        label="95% Confidence Interval",
    )
    ax2.plot(
        x_pos,
        cf_rate,
        color="black",
        linestyle="-",
        marker="s",
        label="Counterfactual Acceptance Rate",
    )
    for idx, val in enumerate(baseline_cost):
        ax1.annotate(f"{val:.2f}", xy=(idx - width / 2, val), xytext=(0, 5),
                     textcoords="offset points", ha="center", fontsize=9)
    for idx, val in enumerate(cf_cost):
        ax1.annotate(f"{val:.2f}", xy=(idx + width / 2, val), xytext=(0, 5),
                     textcoords="offset points", ha="center", fontsize=9)
    for idx, val in enumerate(cf_rate):
        ax2.annotate(f"{val:.2f}%", xy=(idx, val), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=9, weight="bold")

    ax1.set_ylim(15, max(baseline_cost.max(), cf_cost.max()) + 2)
    ax1.set_ylabel("Average Compensation (¥)", fontsize=12)
    ax2.set_ylabel("Acceptance Rate (%)", fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sim_labels, rotation=15, ha="right")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(sim_dir / "Figure_7_knowledge_growth.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"\nAll done! Sections 4.1–4.4: {output_dir}  /  Section 4.5: {sim_dir}  (total time: {_fmt(time.perf_counter() - t_total)})")
