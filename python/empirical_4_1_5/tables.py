from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb


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

