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
