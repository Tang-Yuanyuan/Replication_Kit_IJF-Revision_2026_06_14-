from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WTA_COLS
from .data import build_y_variables


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

