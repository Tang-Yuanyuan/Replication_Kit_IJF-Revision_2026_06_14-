from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CATEGORICAL_COLS, RANDOM_SEED, WTA_COLS, Y_COLS


def get_subset_data(data: pd.DataFrame, mode: str = "car") -> pd.DataFrame:
    if mode == "car":
        return data[data["publictrans"] < 5].copy()
    if mode == "elec":
        return data[data["conditionernumber"] == 1].copy()
    if mode == "green":
        return data[data["energy_consume2020"] > 1000].copy()
    raise ValueError("mode must be one of: car, elec, green")

def build_y_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["max_wta"] = df[WTA_COLS].max(axis=1)
    for wta_col, y_col in zip(WTA_COLS, Y_COLS):
        df[y_col] = ((df[wta_col] == df["max_wta"]) & (df["max_wta"] > 1)).astype(int)
    return df

def prepare_train_test(df: pd.DataFrame, temp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    train_data.to_csv(temp_dir / "train_data.csv", index=False, encoding="utf-8-sig")
    test_data.to_csv(temp_dir / "test_data.csv", index=False, encoding="utf-8-sig")
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
            "weights",        # post-stratification weight — Appendix G only, not used here
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

