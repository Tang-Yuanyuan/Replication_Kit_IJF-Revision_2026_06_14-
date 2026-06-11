# Reproduction Package

This folder contains the code and data needed to reproduce the Section 4.1
empirical results.

## Directory Structure

- `data/raw/energy_wta.csv`: analysis data used by the scripts.
- `R/`: modular R scripts for data preparation, models, figures, robustness
  checks, and weighted analysis.
- `results/tables/`: generated CSV and LaTeX tables.
- `results/figures/`: generated figures.
- `results/logs/`: model diagnostics and R session information.
- `run_all.R`: one-click reproduction script.

The original ad hoc scripts are kept in the project root for traceability.
The reproducible workflow uses the scripts under `R/`.

## How to Reproduce

First reproduce Section 4.1 and the post-stratification weights:

Open R from this folder and run:

```r
source("run_all.R", encoding = "UTF-8")
```

Or run from a terminal:

```powershell
& "D:\R-4.5.2\bin\Rscript.exe" "D:\RUC\revision_package\run_all.R"
```

Then reproduce Section 4.2:

```powershell
& "D:\Python\python.exe" "D:\RUC\revision_package\python\empirical_4_2.py" --root "D:\RUC\revision_package"
```

The default Section 4.2 command uses 50 Optuna trials for preference-alternative
classification, 100 Optuna trials for WTA regression, 1000 random-baseline Monte
Carlo draws, and 200 knowledge-growth simulation repetitions.

To rerun Section 4.2 with the original WTA hyperparameters fixed, write the
outputs to a separate directory:

```powershell
& "D:\Python\python.exe" "D:\RUC\revision_package\python\empirical_4_2.py" --root "D:\RUC\revision_package" --use-legacy-wta-params --output-subdir "empirical4.2_legacy_wta" --skip-simulation
```

This mode keeps the preference-alternative XGBoost classifiers on the seeded
Optuna workflow and fixes only the WTA XGBoost regressors to the values
reported in `Table_D.2_xgboost_hyperparameters`.

For a faster smoke test of the Section 4.2 workflow:

```powershell
& "D:\Python\python.exe" "D:\RUC\revision_package\python\empirical_4_2.py" --root "D:\RUC\revision_package" --n-trials 1 --reg-n-trials 1 --cv 2 --random-iterations 5
```

## R Environment

The scripts add this local package library when it exists:

```r
.libPaths(c("D:/R-4.5.2/Packages", .libPaths()))
```

The Python script adds this local package directory when it exists:

```python
sys.path.insert(0, "D:/Python/Packages")
```

Required R packages:

- `MASS`
- `dplyr`
- `brant`
- `survey`
- `ggplot2`

Required Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `optuna`
- `matplotlib`

After the run, `results/logs/session_info.txt` records the exact R session.

## Numbered Outputs

Key outputs are named to match the manuscript numbering:

- Figures: `F_B.1.png`, `F_B.2.png`
- Main tables: `Table_3_main_transport`, `Table_4_main_home_energy`,
  `Table_5_main_green_electricity`
- Robustness tables: `Table_F.7_transport_robustness`,
  `Table_F.8_home_energy_conditioner_time`,
  `Table_F.9_green_electricity_importance`
- Weighting tables: `Table_G.10_unweighted_balance`,
  `Table_G.11_weighted_balance_test`,
  `Table_G.12_weighted_transport`,
  `Table_G.13_weighted_home_energy`,
  `Table_G.14_weighted_green_electricity`
- Section 4.2 outputs: `Figure_3_prediction_accuracy`,
  `Figure_4_assignment_outcomes`, `Table_D.2_xgboost_hyperparameters`
  (combined Pref Alt and WTA XGBoost hyperparameters),
  `Table_D.3_assignment_outcomes`, `Figure_5_quota_compensation`,
  `Table_D.4_quota_accept_cost`, `Figure_6_budget_participation`,
  `Table_D.5_budget_participation`
- Section 4.3 outputs: `Figure_7_knowledge_growth`,
  `Figure_7_knowledge_growth_summary`

Tables are exported as both `.csv` and `.tex` when applicable.

## Notes

- All paths in the reproducible workflow are relative to the project root.
- The data column names are trimmed on load. This fixes the original `MADT `
  column name, which had a trailing space in the CSV.
- If `weather` is not present in the data, the scripts use `ifsunny` as the
  weather control, matching the available data file.
