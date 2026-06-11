from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PYTHON_PACKAGE_DIR = Path(r"D:\Python\Packages")
if PYTHON_PACKAGE_DIR.exists():
    sys.path.insert(0, str(PYTHON_PACKAGE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split


RANDOM_SEED = 120
WTA_COLS = ["wta_car", "wta_elec", "wta_green"]
Y_COLS = ["y_car", "y_elec", "y_green"]
CATEGORICAL_COLS = [
    "location",
    "province",
    "weekday",
    "heard_about_global_warming",
    "know_about_low_carbon",
    "know_about_carbon_neutrality",
    "know_about_carbon_policy",
]
COST_MAP = {7: 0, 6: 10, 5: 25, 4: 50, 3: 75, 2: 100}
LEGACY_WTA_PARAMS = {
    "Car_Demos_Reg": {"n_estimators": 290, "max_depth": 6, "learning_rate": 0.048, "gamma": 1.642, "reg_lambda": 3.851},
    "Elec_Demos_Reg": {"n_estimators": 190, "max_depth": 4, "learning_rate": 0.047, "gamma": 1.920, "reg_lambda": 4.654},
    "Green_Demos_Reg": {"n_estimators": 50, "max_depth": 6, "learning_rate": 0.026, "gamma": 4.153, "reg_lambda": 1.626},
    "Car_All_Reg": {"n_estimators": 370, "max_depth": 4, "learning_rate": 0.091, "gamma": 3.251, "reg_lambda": 0.254},
    "Elec_All_Reg": {"n_estimators": 170, "max_depth": 3, "learning_rate": 0.062, "gamma": 1.317, "reg_lambda": 4.638},
    "Green_All_Reg": {"n_estimators": 270, "max_depth": 3, "learning_rate": 0.015, "gamma": 4.714, "reg_lambda": 0.216},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce empirical Section 4.2.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for each XGBoost model.",
    )
    parser.add_argument(
        "--reg-n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials for each XGBoost WTA regressor.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Cross-validation folds for Optuna objective.",
    )
    parser.add_argument(
        "--random-iterations",
        type=int,
        default=1000,
        help="Monte Carlo iterations for the random baseline.",
    )
    parser.add_argument(
        "--simulation-iterations",
        type=int,
        default=200,
        help="Knowledge-growth simulation iterations for Section 4.3.",
    )
    parser.add_argument(
        "--output-subdir",
        default="empirical4.2",
        help="Subdirectory under results/ for Section 4.2 outputs.",
    )
    parser.add_argument(
        "--sim-output-subdir",
        default=None,
        help="Subdirectory under results/ for Section 4.3 outputs. Defaults to empirical4.3.",
    )
    parser.add_argument(
        "--use-legacy-wta-params",
        action="store_true",
        help="Use the original WTA hyperparameters recorded in the manuscript table.",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip the Section 4.3 knowledge-growth simulation.",
    )
    parser.add_argument(
        "--rscript",
        type=Path,
        default=Path(r"D:\R-4.5.2\bin\Rscript.exe"),
        help="Path to Rscript.exe.",
    )
    return parser.parse_args()


def get_subset_data(data: pd.DataFrame, mode: str = "car") -> pd.DataFrame:
    if mode == "car":
        return data[data["publictrans"] < 5].copy()
    if mode == "elec":
        return data[data["conditionernumber"] == 1].copy()
    if mode == "green":
        return data[data["energy_consume2020"] > 1000].copy()
    raise ValueError("mode must be one of: car, elec, green")


def train_xgb_model_bayesian(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    group_type: str = "All",
    n_trials: int = 50,
    cv: int = 3,
) -> xgb.XGBClassifier:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=10),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.8, 2.6),
            "gamma": trial.suggest_float("gamma", 1.3, 3.6),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        return cross_val_score(model, train_x, train_y, cv=cv, scoring="accuracy").mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"Training {group_type} XGBoost model with Bayesian optimization...")
    study.optimize(objective, n_trials=n_trials)

    final_params = {
        **study.best_params,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**final_params)
    model.fit(train_x, train_y)
    return model


def train_xgb_regressor_bayesian(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    group_type: str = "All",
    n_trials: int = 50,
    cv: int = 3,
) -> xgb.XGBRegressor:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 490, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
            "gamma": trial.suggest_float("gamma", 0, 2.0),
            "objective": "reg:squarederror",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }
        model = xgb.XGBRegressor(**params)
        return cross_val_score(
            model,
            train_x,
            train_y,
            cv=cv,
            scoring="neg_mean_squared_error",
        ).mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"Training {group_type} XGBoost WTA regressor with Bayesian optimization...")
    study.optimize(objective, n_trials=n_trials)

    final_params = {
        **study.best_params,
        "objective": "reg:squarederror",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    model = xgb.XGBRegressor(**final_params)
    model.fit(train_x, train_y)
    return model


def train_xgb_regressor_fixed(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    params: dict[str, float | int],
) -> xgb.XGBRegressor:
    final_params = {**params, "random_state": RANDOM_SEED}
    model = xgb.XGBRegressor(**final_params)
    model.fit(train_x, train_y)
    return model


def get_group_probabilities(
    model_dict: dict[str, xgb.XGBClassifier],
    test_df: pd.DataFrame,
    group_suffix: str = "Demos",
) -> pd.DataFrame:
    prob_df = pd.DataFrame(index=test_df.index)
    for sub in ["Car", "Elec", "Green"]:
        model_name = f"{sub}_{group_suffix}"
        model = model_dict[model_name]
        features = model.get_booster().feature_names
        prob_df[f"prob_{sub.lower()}"] = model.predict_proba(test_df[features])[:, 1]
    return prob_df


def get_processed_results(
    probs_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 0,
) -> pd.DataFrame:
    res = probs_df.copy()
    prob_cols = ["prob_car", "prob_elec", "prob_green"]
    res["max_probability"] = res[prob_cols].max(axis=1)
    res["best_option"] = res[prob_cols].idxmax(axis=1).str.replace("prob_", "", regex=False)

    mask_car = (res["best_option"] == "car") & (test_df["publictrans"] < 5)
    mask_elec = (res["best_option"] == "elec") & (test_df["conditionernumber"] == 1)
    mask_green = (res["best_option"] == "green") & (test_df["energy_consume2020"] > 1000)
    res["is_valid"] = (
        (mask_car | mask_elec | mask_green) & (res["max_probability"] > threshold)
    ).astype(int)

    res["real_wta_val"] = np.nan
    for opt in ["car", "elec", "green"]:
        mask = res["best_option"] == opt
        res.loc[mask, "real_wta_val"] = test_df.loc[mask, f"wta_{opt}"]

    res["is_accept"] = ((res["is_valid"] == 1) & (res["real_wta_val"] > 1)).astype(int)
    res["is_perfect"] = 0
    for opt in ["car", "elec", "green"]:
        mask = (res["best_option"] == opt) & (res["is_valid"] == 1)
        res.loc[mask, "is_perfect"] = test_df.loc[mask, f"y_{opt}"]

    return res


def append_final_wta_column(
    results_df: pd.DataFrame,
    test_df: pd.DataFrame,
    xgb_reg_dict: dict[str, xgb.XGBRegressor],
    group_suffix: str = "All",
) -> pd.DataFrame:
    df = results_df.copy()
    temp_cols = []

    for opt in ["car", "elec", "green"]:
        model_key = f"{opt.capitalize()}_{group_suffix}_Reg"
        temp_col = f"temp_pred_{opt}"
        if model_key in xgb_reg_dict:
            model = xgb_reg_dict[model_key]
            features = model.get_booster().feature_names
            df[temp_col] = np.floor(model.predict(test_df[features])).astype(int)
            temp_cols.append(temp_col)

    df["pred_wta_val"] = np.nan
    for opt in ["car", "elec", "green"]:
        mask = df["best_option"].str.lower() == opt
        temp_col = f"temp_pred_{opt}"
        if temp_col in df.columns:
            df.loc[mask, "pred_wta_val"] = df.loc[mask, temp_col]

    return df.drop(columns=temp_cols)


def update_eco_results_with_floor(results_df: pd.DataFrame, preds_path: Path) -> pd.DataFrame:
    results = results_df.copy()
    preds_raw = pd.read_csv(preds_path)
    preds_raw.index = results.index
    opt_map = {
        "Car": "pred_wta_car",
        "car": "pred_wta_car",
        "Elec": "pred_wta_elec",
        "elec": "pred_wta_elec",
        "Green": "pred_wta_green",
        "green": "pred_wta_green",
    }

    results["pred_wta_val"] = np.nan
    for idx in results.index:
        best_opt = results.loc[idx, "best_option"]
        if results.loc[idx, "is_valid"] == 1 and pd.notna(best_opt):
            col_name = opt_map.get(best_opt)
            if col_name is not None:
                results.loc[idx, "pred_wta_val"] = np.floor(preds_raw.loc[idx, col_name])
    return results


def calculate_metrics(demos_df: pd.DataFrame, all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, data in [("Demos", demos_df), ("All", all_df)]:
        valid = data[data["is_valid"] == 1].copy()
        n_total = len(data)
        n_valid = len(valid)
        accepted = valid[valid["is_accept"] == 1]
        rows.append(
            {
                "Group": name,
                "Total_Tests": n_total,
                "Valid_Predictions": n_valid,
                "Validity_Rate": n_valid / n_total if n_total else 0,
                "Accept_Rate": valid["is_accept"].mean() if n_valid else 0,
                "Perfect_Rate": valid["is_perfect"].mean() if n_valid else 0,
                "Accept_Cost": accepted["real_wta_val"].map(COST_MAP).mean()
                if len(accepted) > 0
                else 0,
            }
        )
    return pd.DataFrame(rows).set_index("Group")


def get_ideal_results(test_df: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame(index=test_df.index)
    res["best_option"] = "None"
    for opt in ["car", "elec", "green"]:
        res.loc[test_df[f"y_{opt}"] == 1, "best_option"] = opt

    res["real_wta_val"] = np.nan
    for opt in ["car", "elec", "green"]:
        mask = res["best_option"] == opt
        res.loc[mask, "real_wta_val"] = test_df.loc[mask, f"wta_{opt}"]

    res["is_valid"] = (res["best_option"] != "None").astype(int)
    res["is_accept"] = res["is_valid"]
    res["is_perfect"] = res["is_valid"]
    res["max_probability"] = 1.0
    return res


def run_monte_carlo_random(test_df: pd.DataFrame, n_iterations: int = 1000) -> pd.DataFrame:
    can_car = test_df["publictrans"] < 5
    can_elec = test_df["conditionernumber"] == 1
    can_green = test_df["energy_consume2020"] > 1000
    metrics = []

    for i in range(n_iterations):
        rng = np.random.default_rng(RANDOM_SEED + i)
        res = pd.DataFrame(index=test_df.index)
        res["best_option"] = "None"

        for idx in test_df.index:
            available = []
            if can_car.loc[idx]:
                available.append("car")
            if can_elec.loc[idx]:
                available.append("elec")
            if can_green.loc[idx]:
                available.append("green")
            if available:
                res.loc[idx, "best_option"] = rng.choice(available)

        res["is_valid"] = (res["best_option"] != "None").astype(int)
        res["real_wta_val"] = np.nan
        res["is_perfect"] = 0
        for opt in ["car", "elec", "green"]:
            mask = res["best_option"] == opt
            res.loc[mask, "real_wta_val"] = test_df.loc[mask, f"wta_{opt}"]
            res.loc[mask, "is_perfect"] = test_df.loc[mask, f"y_{opt}"].astype(int)

        accept_mask = (res["is_valid"] == 1) & (res["real_wta_val"] > 1)
        valid_n = int(res["is_valid"].sum())
        accepted = res.loc[accept_mask, "real_wta_val"]
        metrics.append(
            {
                "Validity_Rate": valid_n / len(test_df),
                "Accept_Rate": accept_mask.sum() / valid_n if valid_n else 0,
                "Perfect_Rate": res["is_perfect"].sum() / valid_n if valid_n else 0,
                "Accept_Cost": accepted.map(COST_MAP).mean() if len(accepted) else 0,
            }
        )

    avg = pd.DataFrame(metrics).mean()
    return pd.DataFrame(
        [
            {
                "Group": "Random (Lower Bound)",
                "Total_Tests": len(test_df),
                "Valid_Predictions": int(avg["Validity_Rate"] * len(test_df)),
                "Validity_Rate": avg["Validity_Rate"],
                "Accept_Rate": avg["Accept_Rate"],
                "Perfect_Rate": avg["Perfect_Rate"],
                "Accept_Cost": avg["Accept_Cost"],
            }
        ]
    ).set_index("Group")


def get_n_household_metrics(
    results_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_list: list[int] | None = None,
    sort: bool = True,
) -> pd.DataFrame:
    if n_list is None:
        n_list = [55, 100, 145]

    cost_map = {7: 0, 6: 10, 5: 25, 4: 50, 3: 75, 2: 100, 1: 0}
    eval_pool = results_df[results_df["is_valid"] == 1].copy()
    eval_pool["real_wta_code"] = np.nan

    for opt in ["car", "elec", "green"]:
        mask = eval_pool["best_option"].str.lower() == opt
        if mask.any():
            eval_pool.loc[mask, "real_wta_code"] = test_df.loc[eval_pool[mask].index, f"wta_{opt}"]

    if sort:
        sort_cols = []
        if "pred_wta_val" in eval_pool.columns:
            sort_cols.append("pred_wta_val")
        elif "real_wta_val" in eval_pool.columns:
            sort_cols.append("real_wta_val")
        if "max_probability" in eval_pool.columns:
            sort_cols.append("max_probability")
        if sort_cols:
            eval_pool = eval_pool.sort_values(by=sort_cols, ascending=False)

    rows = []
    for n in n_list:
        if len(eval_pool) >= n:
            winners = eval_pool.head(n)
            valid_winners = winners.dropna(subset=["real_wta_code"])
            accepted = valid_winners[valid_winners["real_wta_code"] > 1]
            rows.append(
                {
                    "Quota (N)": n,
                    "Accept_Rate": len(accepted) / n,
                    "Accept_Cost": accepted["real_wta_code"].map(cost_map).mean()
                    if len(accepted) > 0
                    else 0.0,
                }
            )
        else:
            rows.append({"Quota (N)": n, "Accept_Rate": np.nan, "Accept_Cost": np.nan})
    return pd.DataFrame(rows)


def get_random_baseline_monte_carlo(
    test_df: pd.DataFrame,
    n_list: list[int],
    iterations: int = 1000,
) -> pd.DataFrame:
    all_results = []
    for i in range(iterations):
        rng = np.random.default_rng(RANDOM_SEED + i)
        random_rows = []
        for idx, row in test_df.iterrows():
            eligible = []
            if row["publictrans"] < 5:
                eligible.append("Car")
            if row["conditionernumber"] == 1:
                eligible.append("Elec")
            if row["energy_consume2020"] > 1000:
                eligible.append("Green")
            random_rows.append(
                {"best_option": rng.choice(eligible), "is_valid": 1}
                if eligible
                else {"best_option": "None", "is_valid": 0}
            )
        sim_df = pd.DataFrame(random_rows, index=test_df.index)
        all_results.append(get_n_household_metrics(sim_df, test_df, n_list=n_list, sort=False))
    return pd.concat(all_results).groupby("Quota (N)").mean().reset_index()


def get_budget_metrics(
    results_df: pd.DataFrame,
    test_df: pd.DataFrame,
    budget_list: list[int] | None = None,
    sort: bool = True,
) -> pd.DataFrame:
    if budget_list is None:
        budget_list = [1000, 2000, 3000, 4000]

    cost_map = {7: 0, 6: 10, 5: 25, 4: 50, 3: 75, 2: 100, 1: 0}
    eval_pool = results_df[results_df["is_valid"] == 1].copy()
    eval_pool["real_wta_code"] = np.nan

    for opt in ["car", "elec", "green"]:
        mask = eval_pool["best_option"].str.lower() == opt
        if mask.any():
            eval_pool.loc[mask, "real_wta_code"] = test_df.loc[eval_pool[mask].index, f"wta_{opt}"]

    eval_pool["individual_cost"] = eval_pool["real_wta_code"].map(cost_map)

    if sort:
        sort_cols = []
        if "pred_wta_val" in eval_pool.columns:
            sort_cols.append("pred_wta_val")
        elif "real_wta_val" in eval_pool.columns:
            sort_cols.append("real_wta_val")
        if "max_probability" in eval_pool.columns:
            sort_cols.append("max_probability")
        if sort_cols:
            eval_pool = eval_pool.sort_values(by=sort_cols, ascending=False)

    rows = []
    for budget in budget_list:
        eval_pool["cum_cost"] = eval_pool["individual_cost"].cumsum()
        winners = eval_pool[eval_pool["cum_cost"] <= budget]
        n_recruited = len(winners)
        accept_rate = (
            len(winners[winners["real_wta_code"] > 1]) / n_recruited if n_recruited > 0 else 0.0
        )
        rows.append(
            {
                "Budget_Limit": budget,
                "Total_Recruited (N)": n_recruited,
                "Accept_Rate": accept_rate,
            }
        )
    return pd.DataFrame(rows)


def get_random_budget_baseline_monte_carlo(
    test_df: pd.DataFrame,
    budget_list: list[int],
    iterations: int = 1000,
) -> pd.DataFrame:
    all_results = []
    for i in range(iterations):
        rng = np.random.default_rng(RANDOM_SEED + i)
        random_rows = []
        for idx, row in test_df.iterrows():
            eligible = []
            if row["publictrans"] < 5:
                eligible.append("Car")
            if row["conditionernumber"] == 1:
                eligible.append("Elec")
            if row["energy_consume2020"] > 1000:
                eligible.append("Green")
            random_rows.append(
                {"best_option": rng.choice(eligible), "is_valid": 1}
                if eligible
                else {"best_option": "None", "is_valid": 0}
            )
        sim_df = pd.DataFrame(random_rows, index=test_df.index)
        all_results.append(get_budget_metrics(sim_df, test_df, budget_list=budget_list, sort=False))
    return pd.concat(all_results).groupby("Budget_Limit").mean().reset_index()


def clean_pct(value: object) -> float:
    if isinstance(value, str):
        return float(value.replace("%", "").replace(",", "").strip())
    return float(value)


def simulate_policy_knowledge_upgrade(df: pd.DataFrame) -> pd.DataFrame:
    sim_df = df.copy()
    for col in WTA_COLS:
        mask_not_max = (sim_df[col] < 7) & (sim_df[col] >= 1)
        sim_df.loc[mask_not_max, col] = sim_df.loc[mask_not_max, col] + np.random.uniform(
            0, 1, size=mask_not_max.sum()
        )

    params = {
        "wta_car": {"heard_not_know": (0.183, 0.186), "heard_know": (0.697, 0.366)},
        "wta_elec": {"heard_not_know": (0.641, 0.208), "heard_know": (0.888, 0.363)},
        "wta_green": {"heard_not_know": (0.357, 0.159), "heard_know": (0.972, 0.286)},
    }
    policy_col = "know_about_carbon_policy"
    mask_never = sim_df[policy_col] == "never"
    mask_not_know = sim_df[policy_col] == "heard but do not know"

    for target_wta in WTA_COLS:
        mean_know, se_know = params[target_wta]["heard_know"]
        mean_not, se_not = params[target_wta]["heard_not_know"]

        if mask_never.any():
            delta_never = np.random.normal(mean_know, se_know, size=mask_never.sum())
            sim_df.loc[mask_never, target_wta] += np.maximum(delta_never, 0)

        if mask_not_know.any():
            delta_know = np.random.normal(mean_know, se_know, size=mask_not_know.sum())
            delta_not = np.random.normal(mean_not, se_not, size=mask_not_know.sum())
            sim_df.loc[mask_not_know, target_wta] += np.maximum(delta_know - delta_not, 0)

        sim_df[target_wta] = np.floor(sim_df[target_wta]).clip(lower=1, upper=7)

    affected = mask_never | mask_not_know
    sim_df.loc[affected, policy_col] = "heard and know"
    sim_df = build_y_variables(sim_df)
    return sim_df


def append_metrics(target_dict: dict[str, list[float]], df: pd.DataFrame, row_idx: int) -> None:
    target_dict.setdefault("accept_cost", []).append(float(df.iloc[row_idx]["Accept_Cost"]))
    target_dict.setdefault("accept_rate", []).append(float(df.iloc[row_idx]["Accept_Rate"]))


def summarize_simulation_results(results: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    rows = []
    for key, values in results.items():
        costs = np.asarray(values["accept_cost"], dtype=float)
        rates = np.asarray(values["accept_rate"], dtype=float) * 100
        cost_sd = costs.std(ddof=1) if len(costs) > 1 else 0.0
        rate_sd = rates.std(ddof=1) if len(rates) > 1 else 0.0
        rows.append(
            {
                "Key": key,
                "Cost_Mean": costs.mean(),
                "Cost_SD": cost_sd,
                "Cost_Low": costs.mean() - 2 * cost_sd,
                "Cost_High": costs.mean() + 2 * cost_sd,
                "Rate_Mean": rates.mean(),
                "Rate_SD": rate_sd,
                "Rate_Low": rates.mean() - 2 * rate_sd,
                "Rate_High": rates.mean() + 2 * rate_sd,
            }
        )
    return pd.DataFrame(rows).set_index("Key")


def format_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Validity_Rate", "Accept_Rate", "Perfect_Rate"]:
        out[col] = out[col].map(lambda x: f"{100 * float(x):.2f}%")
    out["Accept_Cost"] = out["Accept_Cost"].map(lambda x: f"{float(x):.2f}")
    return out


def write_latex_table(df: pd.DataFrame, path: Path, caption: str) -> None:
    latex = df.to_latex(index=True, escape=True, caption=caption)
    path.write_text(latex, encoding="utf-8")


def hyperparameter_row(model_name: str, model: xgb.XGBModel, block: str) -> dict[str, object]:
    parts = model_name.split("_")
    outcome = parts[0]
    suffix = parts[1]

    outcome_labels = {
        "Car": "Transportation-mode shifts",
        "Elec": "Home-energy reduction",
        "Green": "Green-electricity",
    }
    suffix_labels = {"Demos": "I", "All": "II"}
    params = model.get_params()

    return {
        "Block": block,
        "Dependent Variable": f"{outcome_labels[outcome]} {suffix_labels[suffix]}",
        "B": params.get("n_estimators"),
        "max(log2(T)+1)": params.get("max_depth"),
        "eta": f"{params.get('learning_rate'):.3f}",
        "gamma": f"{params.get('gamma'):.3f}",
        "lambda": f"{params.get('reg_lambda'):.3f}",
    }


def build_combined_hyperparameter_table(
    pref_alt_models: dict[str, xgb.XGBClassifier],
    wta_models: dict[str, xgb.XGBRegressor],
) -> pd.DataFrame:
    ordered_pref = [
        "Car_Demos",
        "Elec_Demos",
        "Green_Demos",
        "Car_All",
        "Elec_All",
        "Green_All",
    ]
    ordered_wta = [
        "Car_Demos_Reg",
        "Elec_Demos_Reg",
        "Green_Demos_Reg",
        "Car_All_Reg",
        "Elec_All_Reg",
        "Green_All_Reg",
    ]

    rows = []
    for name in ordered_pref:
        rows.append(hyperparameter_row(name, pref_alt_models[name], "Pref Alt"))
    for name in ordered_wta:
        rows.append(hyperparameter_row(name, wta_models[name], "WTA"))

    table = pd.DataFrame(rows)
    return table.set_index(["Block", "Dependent Variable"])


def build_y_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["max_wta"] = df[WTA_COLS].max(axis=1)
    for wta_col, y_col in zip(WTA_COLS, Y_COLS):
        df[y_col] = ((df[wta_col] == df["max_wta"]) & (df["max_wta"] > 1)).astype(int)
    return df


def prepare_train_test(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_y_variables(df)
    df["stratify_key"] = df["y_car"].astype(str) + df["y_elec"].astype(str) + df["y_green"].astype(str)

    train_data, test_data = train_test_split(
        df,
        test_size=0.3,
        random_state=RANDOM_SEED,
        stratify=df["stratify_key"],
    )

    train_data = train_data.drop(columns=["stratify_key"])
    test_data = test_data.drop(columns=["stratify_key"])
    train_data.to_csv(output_dir / "train_data.csv", index=False, encoding="utf-8-sig")
    test_data.to_csv(output_dir / "test_data.csv", index=False, encoding="utf-8-sig")
    return train_data, test_data


def encode_for_xgboost(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = pd.get_dummies(train_data, columns=CATEGORICAL_COLS, drop_first=False)
    test_encoded = pd.get_dummies(test_data, columns=CATEGORICAL_COLS, drop_first=False)
    train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)
    return train_encoded.astype(int, errors="ignore"), test_encoded.astype(int, errors="ignore")


def encode_like_reference(data: pd.DataFrame, reference_columns: pd.Index) -> pd.DataFrame:
    encoded = pd.get_dummies(data, columns=CATEGORICAL_COLS, drop_first=False)
    encoded = encoded.reindex(columns=reference_columns, fill_value=0)
    return encoded.astype(int, errors="ignore")


def get_feature_sets(train_encoded: pd.DataFrame) -> dict[str, list[str]]:
    train_car = get_subset_data(train_encoded, mode="car")
    demos_all = [
        col
        for col in train_car.columns
        if col
        not in [
            "wta_car",
            "wta_elec",
            "wta_green",
            "y_car",
            "y_elec",
            "y_green",
            "id",
            "weights",
            "max_wta",
            "publictrans",
            "conditionernumber",
            "energy_consume2020",
        ]
    ]
    knowledge_keys = [
        "heard_about_global_warming",
        "know_about_low_carbon",
        "know_about_carbon_neutrality",
        "know_about_carbon_policy",
    ]
    demos = [col for col in demos_all if not any(key in col for key in knowledge_keys)]

    exclude = {
        "car": ["conditioner1month", "mainuseelec"],
        "elec": ["caruse", "mainuseelec"],
        "green": ["caruse", "conditioner1month"],
    }
    return {
        "demos_all_car": [c for c in demos_all if c not in exclude["car"]],
        "demos_all_elec": [c for c in demos_all if c not in exclude["elec"]],
        "demos_all_green": [c for c in demos_all if c not in exclude["green"]],
        "demos_car": [c for c in demos if c not in exclude["car"]],
        "demos_elec": [c for c in demos if c not in exclude["elec"]],
        "demos_green": [c for c in demos if c not in exclude["green"]],
    }


def run_r_logit(root: Path, rscript: Path, output_subdir: str) -> None:
    script = root / "R" / "07_logit_4_2.R"
    subprocess.run(
        [str(rscript), str(script), str(root), output_subdir],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="ignore",
    )


def run_r_ologit_wta(root: Path, rscript: Path, output_subdir: str) -> None:
    script = root / "R" / "08_ologit_wta_4_2.R"
    subprocess.run(
        [str(rscript), str(script), str(root), output_subdir],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="ignore",
    )


def run_r_simulated_predictions(
    root: Path,
    rscript: Path,
    train_subdir: str,
    sim_subdir: str,
) -> None:
    script = root / "R" / "09_simulated_predictions_4_3.R"
    subprocess.run(
        [str(rscript), str(script), str(root), train_subdir, sim_subdir],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="ignore",
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / "results" / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_output_subdir = args.sim_output_subdir or "empirical4.3"
    use_legacy_wta = args.use_legacy_wta_params

    input_file = root / "results" / "energy_wta_with_post_weights.csv"
    if not input_file.exists():
        raise FileNotFoundError(
            f"{input_file} not found. Run run_all.R first to generate weighted data."
        )

    df = pd.read_csv(input_file)
    train_raw, test_raw = prepare_train_test(df, output_dir)
    train_encoded, test_encoded = encode_for_xgboost(train_raw, test_raw)
    feature_sets = get_feature_sets(train_encoded)

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

    trained_models: dict[str, xgb.XGBClassifier] = {}
    for name, train_df, y_col, features, group_type in train_tasks:
        print(f"Training {name}...")
        trained_models[name] = train_xgb_model_bayesian(
            train_df[features],
            train_df[y_col],
            group_type=group_type,
            n_trials=args.n_trials,
            cv=args.cv,
        )

    reg_tasks = [
        ("Car_Demos_Reg", train_car, "wta_car", feature_sets["demos_car"], "Demos"),
        ("Car_All_Reg", train_car, "wta_car", feature_sets["demos_all_car"], "All"),
        ("Elec_Demos_Reg", train_elec, "wta_elec", feature_sets["demos_elec"], "Demos"),
        ("Elec_All_Reg", train_elec, "wta_elec", feature_sets["demos_all_elec"], "All"),
        ("Green_Demos_Reg", train_green, "wta_green", feature_sets["demos_green"], "Demos"),
        ("Green_All_Reg", train_green, "wta_green", feature_sets["demos_all_green"], "All"),
    ]

    trained_models_reg: dict[str, xgb.XGBRegressor] = {}
    for name, train_df, y_col, features, group_type in reg_tasks:
        print(f"Training {name}...")
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
        preds = model.predict(test_encoded[features])
        mae = np.mean(np.abs(test_encoded[y_col] - preds))
        print(f"{name} test MAE: {mae:.4f}")

    if use_legacy_wta:
        (output_dir / "legacy_params_used.txt").write_text(
            "WTA XGBoost regressors used fixed hyperparameters copied from the original manuscript table.\n"
            "Pref Alt classifiers were trained through the seeded Optuna workflow.\n",
            encoding="utf-8",
        )

    df_params = build_combined_hyperparameter_table(trained_models, trained_models_reg)
    df_params.to_csv(output_dir / "Table_D.2_xgboost_hyperparameters.csv", encoding="utf-8-sig")
    write_latex_table(
        df_params,
        output_dir / "Table_D.2_xgboost_hyperparameters.tex",
        "Table D.2. XGBoost Hyperparameters",
    )

    results_ideal = get_ideal_results(test_encoded)
    final_df_ideal = calculate_metrics(results_ideal, results_ideal).iloc[[0]].copy()
    final_df_ideal.index = ["Ideal (Upper Bound)"]

    demos_probs = get_group_probabilities(trained_models, test_encoded, "Demos")
    all_probs = get_group_probabilities(trained_models, test_encoded, "All")
    results_xgb_demos = get_processed_results(demos_probs, test_encoded, threshold=0)
    results_xgb_all = get_processed_results(all_probs, test_encoded, threshold=0)
    final_df_xgb = calculate_metrics(results_xgb_demos, results_xgb_all)

    print("Running R logit models...")
    run_r_logit(root, args.rscript, args.output_subdir)
    logit_demos = pd.read_csv(output_dir / "logit_probs_demos.csv")
    logit_all = pd.read_csv(output_dir / "logit_probs_all.csv")
    logit_demos.index = test_encoded.index
    logit_all.index = test_encoded.index

    results_eco_demos = get_processed_results(logit_demos, test_encoded, threshold=0)
    results_eco_all = get_processed_results(logit_all, test_encoded, threshold=0)
    final_df_eco = calculate_metrics(results_eco_demos, results_eco_all)
    final_df_random = run_monte_carlo_random(test_encoded, n_iterations=args.random_iterations)

    accuracy_data = {
        "Logistic regression I": clean_pct(final_df_eco.loc["Demos", "Perfect_Rate"]) * 100,
        "Logistic regression II": clean_pct(final_df_eco.loc["All", "Perfect_Rate"]) * 100,
        "XGBoost algorithm I": clean_pct(final_df_xgb.loc["Demos", "Perfect_Rate"]) * 100,
        "XGBoost algorithm II": clean_pct(final_df_xgb.loc["All", "Perfect_Rate"]) * 100,
    }
    accuracy_series = pd.Series(accuracy_data)
    accuracy_series.to_csv(output_dir / "Figure_3_prediction_accuracy_data.csv", header=["Accuracy"])

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
    all_metrics.to_csv(output_dir / "empirical_4.2_metrics.csv", encoding="utf-8-sig")

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

    print("Running R ordered-logit WTA models...")
    run_r_ologit_wta(root, args.rscript, args.output_subdir)
    results_xgb_all = append_final_wta_column(results_xgb_all, test_encoded, trained_models_reg, "All")
    results_xgb_demos = append_final_wta_column(results_xgb_demos, test_encoded, trained_models_reg, "Demos")
    results_eco_all = update_eco_results_with_floor(results_eco_all, output_dir / "wta_preds_all.csv")
    results_eco_demos = update_eco_results_with_floor(results_eco_demos, output_dir / "wta_preds_demos.csv")

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
    quota_pivot.to_csv(output_dir / "Table_D.4_quota_metrics_full.csv", encoding="utf-8-sig")

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

    ax_top.set_ylim(
        min(m_xgb_demos.loc[0, "Accept_Cost"], m_xgb_all.loc[0, "Accept_Cost"]) - 1.5,
        m_random.loc[0, "Accept_Cost"] + 2,
    )
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
    budget_pivot.to_csv(output_dir / "Table_D.5_budget_metrics_full.csv", encoding="utf-8-sig")

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

    if args.skip_simulation:
        print("Section 4.3 simulation skipped.")
        print(f"Empirical 4.2 outputs written to: {output_dir}")
        return

    sim_dir = root / "results" / sim_output_subdir
    sim_dir.mkdir(parents=True, exist_ok=True)
    simulate_results = {
        "ideal_sm": {},
        "random_sm": {},
        "xgb_all_sm": {},
        "xgb_demos_sm": {},
        "eco_all_sm": {},
        "eco_demos_sm": {},
    }
    test_base = test_raw.copy().reset_index(drop=True)

    for i in range(args.simulation_iterations):
        np.random.seed(RANDOM_SEED + i)
        simulated_raw = simulate_policy_knowledge_upgrade(test_base.copy())
        simulated_raw.to_csv(sim_dir / "test_data_simulated.csv", index=False, encoding="utf-8-sig")

        run_r_simulated_predictions(root, args.rscript, args.output_subdir, sim_output_subdir)
        logit_demos_sm = pd.read_csv(sim_dir / "logit_probs_demos_simulated.csv")
        logit_all_sm = pd.read_csv(sim_dir / "logit_probs_all_simulated.csv")
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

    print(f"Empirical 4.2 outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
