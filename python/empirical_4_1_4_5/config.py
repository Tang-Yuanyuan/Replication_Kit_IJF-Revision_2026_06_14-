from __future__ import annotations


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
