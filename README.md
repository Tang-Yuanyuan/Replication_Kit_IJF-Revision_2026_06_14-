# Reproduction Package

This folder contains the code and data needed to reproduce the empirical
results.

## Directory Structure

- `data/raw/energy_wta.csv`: analysis data used by the scripts.
- `data/temp/`: generated intermediate data shared between R and Python scripts.
- `R/`: modular R scripts for data preparation, models, figures, robustness
  checks, and weighted analysis.
- `results/empirical4.1/tables/`: generated Section 4.1 CSV and LaTeX tables.
- `results/empirical4.1/figures/`: generated Section 4.1 figures.
- `results/empirical4.1/logs/`: Section 4.1 model diagnostics and R session information.
- `run_all.R`: one-click reproduction script.
- `run_empirical4_2_and_4_3.py`: root-level entry point for Sections 4.2 and 4.3.
- `run_empirical4_2_3.py`: deprecated compatibility entry point that forwards to the new name.
- `python/empirical_4_2_4_3/`: modular Python implementation for Sections 4.2 and 4.3.
- `python/empirical_4_2.py`: deprecated compatibility wrapper for older commands.
- `ENGINEERING_PLAN.md`: current reorganization plan and status.
- `archive/historical_sources/`: original ad hoc scripts and reference files
  kept for traceability only.

The reproducible workflow uses `run_all.R`, `run_empirical4_2_and_4_3.py`, scripts
under `R/`, and the `python/empirical_4_2_4_3/` package. Files under
`archive/historical_sources/` are not called by the current workflow.

## Setup

Tested environment: **R 4.5.2**, **Python 3.x**.

**Step 1 — Install R packages** (all direct and transitive dependencies,
pinned to tested versions):

```powershell
Rscript install.R
```

**Step 2 — Install Python packages** (all direct and transitive dependencies,
pinned to tested versions):

```powershell
pip install -r requirements.txt
```

`install.R` covers 37 R packages (5 direct + 32 transitive).
`requirements.txt` covers 22 Python packages (6 direct + 16 transitive).

If a package fails to install at the pinned version, try omitting the version
pin — the latest CRAN/PyPI release is usually compatible.

## How to Reproduce

First reproduce Section 4.1 and the post-stratification weights:

Open R from this folder and run:

```r
source("run_all.R", encoding = "UTF-8")
```

Or run from a terminal (cd to the project root first):

```powershell
Rscript run_all.R
```

Then reproduce Sections 4.2 and 4.3:

```powershell
python run_empirical4_2_and_4_3.py
```

The default Sections 4.2 and 4.3 command uses 50 Optuna trials for preference-alternative
classification, 100 Optuna trials for WTA regression, 1000 random-baseline Monte
Carlo draws, and 200 knowledge-growth simulation repetitions.

To rerun Section 4.2 with the original WTA hyperparameters fixed, write the
outputs to a separate directory:

```powershell
python run_empirical4_2_and_4_3.py --use-legacy-wta-params --output-subdir "empirical4.2_legacy_wta" --skip-simulation
```

This mode keeps the preference-alternative XGBoost classifiers on the seeded
Optuna workflow and fixes only the WTA XGBoost regressors to the values
reported in `Table_D.2_xgboost_hyperparameters`.

For a faster smoke test of the Section 4.2 workflow:

```powershell
python run_empirical4_2_and_4_3.py --n-trials 1 --reg-n-trials 1 --cv 2 --random-iterations 5
```

## R Environment

All paths in the scripts are resolved relative to the project root at runtime.
No hard-coded machine-specific paths are required.

If your R packages are installed in a non-standard location, set the environment
variable `R_PACKAGE_DIR` before running:

```powershell
$env:R_PACKAGE_DIR = "C:\path\to\your\R\packages"
Rscript run_all.R
```

If your Python packages are installed in a non-standard location, set the
environment variable `PYTHON_PACKAGE_DIR` before running:

```powershell
$env:PYTHON_PACKAGE_DIR = "C:\path\to\your\python\packages"
python run_empirical4_2_and_4_3.py
```

If Rscript is not on your PATH, pass it explicitly:

```powershell
python run_empirical4_2_and_4_3.py --rscript "C:\path\to\R\bin\Rscript.exe"
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

After the run, `results/empirical4.1/logs/session_info.txt` records the exact R session.
Intermediate files such as post-stratification data, train/test splits, logit
probabilities, ordered-logit WTA predictions, and simulated test data are
written under `data/temp/`.

## Numbered Outputs

Key outputs are named to match the manuscript numbering:

- Figures: `F_B.1.png`, `F_B.2.png`
- Main tables: `Table_3_main_transport`, `Table_4_main_home_energy`,
  `Table_5_main_green_electricity`
- Diagnostic tables: `Table_E.6_brant_tests` (proportional odds assumption tests for all ordered logit models)
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
