from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import COST_MAP, RANDOM_SEED


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

