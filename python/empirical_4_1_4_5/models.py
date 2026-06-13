from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from .config import RANDOM_SEED


def train_xgb_model_bayesian(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    group_type: str = "All",
    n_trials: int = 50,
    cv: int = 3,
    sample_weight: "np.ndarray | None" = None,
    weighted: bool = False,
) -> xgb.XGBClassifier:
    def objective(trial: optuna.Trial) -> float:
        if weighted:
            if group_type == "All":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 190, 250, step=5),
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
                    "reg_lambda": trial.suggest_float("reg_lambda", 2, 2.6),
                    "gamma": trial.suggest_float("gamma", 3, 4),
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 140, 150, step=1),
                    "max_depth": trial.suggest_int("max_depth", 4, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 1),
                    "gamma": trial.suggest_float("gamma", 6, 6),
                }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 20, 200, step=10),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.8, 2.6),
                "gamma": trial.suggest_float("gamma", 1.3, 3.6),
            }
        params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_SEED,
            "n_jobs": -1 if weighted else 1,
        })
        model = xgb.XGBClassifier(**params)
        cv_kwargs = {"params": {"sample_weight": sample_weight}} if sample_weight is not None else {}
        return cross_val_score(model, train_x, train_y, cv=cv, scoring="accuracy", **cv_kwargs).mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    final_params = {
        **study.best_params,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**final_params)
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(train_x, train_y, **fit_kwargs)
    return model

def train_xgb_regressor_bayesian(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    group_type: str = "All",
    n_trials: int = 50,
    cv: int = 3,
    sample_weight: "np.ndarray | None" = None,
    weighted: bool = False,
) -> xgb.XGBRegressor:
    def objective(trial: optuna.Trial) -> float:
        if weighted:
            if group_type == "All":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 90, step=40),
                    "max_depth": trial.suggest_int("max_depth", 5, 5),
                    "learning_rate": trial.suggest_float("learning_rate", 0, 0.05),
                    "reg_lambda": trial.suggest_float("reg_lambda", 5, 5),
                    "gamma": trial.suggest_float("gamma", 0.5, 0.5),
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 100, step=10),
                    "max_depth": trial.suggest_int("max_depth", 4, 5),
                    "learning_rate": trial.suggest_float("learning_rate", 0.3, 0.5),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 0.5),
                    "gamma": trial.suggest_float("gamma", 0.5, 0.5),
                }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 490, step=20),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
                "gamma": trial.suggest_float("gamma", 0, 2.0),
            }
        params.update({
            "objective": "reg:squarederror",
            "random_state": RANDOM_SEED,
            "n_jobs": -1 if weighted else 1,
        })
        model = xgb.XGBRegressor(**params)
        cv_kwargs = {"params": {"sample_weight": sample_weight}} if sample_weight is not None else {}
        return cross_val_score(
            model,
            train_x,
            train_y,
            cv=cv,
            scoring="neg_mean_squared_error",
            **cv_kwargs,
        ).mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    final_params = {
        **study.best_params,
        "objective": "reg:squarederror",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    model = xgb.XGBRegressor(**final_params)
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(train_x, train_y, **fit_kwargs)
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

