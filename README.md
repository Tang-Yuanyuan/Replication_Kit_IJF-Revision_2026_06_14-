# Replication Kit: IJF-D-25-00886R1

Yuanyuan Tang (School of Applied Economics, Renmin University of China)

## Overview

The code in this replication kit reproduces all figures and tables in the
empirical sections of the paper "Designing Cost-effective Climate Policies
through Preference Prediction: Evidence from Chinese Households'
Low-carbon Alternatives." The analysis combines R (Section 4.1:
ordered logit models, robustness checks, and post-stratification weighting)
and Python (Sections 4.1–4.5: machine-learning prediction, assignment
optimisation, and knowledge-growth simulation).

The kit generates **12 figures** and **19 tables** across Sections 4.1–4.5
and Appendices B–G.

### Generated outputs

**Section 4.1 — Ordered logit models** (written to `results/empirical3/`)

| Output                                          | File name                                             |
| ----------------------------------------------- | ----------------------------------------------------- |
| Descriptive statistics                          | `Table1_categorical.csv`, `Table1_continuous.csv` |
| WTA distribution                                | `Table2_wta_distribution.csv`                       |
| Main results: transport                         | `Table_3_main_transport.csv/.tex`                   |
| Main results: home energy                       | `Table_4_main_home_energy.csv/.tex`                 |
| Main results: green electricity                 | `Table_5_main_green_electricity.csv/.tex`           |
| Appendix B: descriptive figures                 | `F_B.1.png`, `F_B.2.png`                          |
| Appendix E: Brant tests                         | `Table_E.6_brant_tests.csv/.tex`                    |
| Appendix F: robustness, transport               | `Table_F.7_transport_robustness.csv/.tex`           |
| Appendix F: robustness, home energy             | `Table_F.8_home_energy_conditioner_time.csv/.tex`   |
| Appendix F: robustness, green electricity       | `Table_F.9_green_electricity_importance.csv/.tex`   |
| Appendix G: covariate balance (unweighted)      | `Table_G.10_unweighted_balance.csv/.tex`            |
| Appendix G: covariate balance (weighted)        | `Table_G.11_weighted_balance_test.csv/.tex`         |
| Appendix G: weighted results, transport         | `Table_G.12_weighted_transport.csv/.tex`            |
| Appendix G: weighted results, home energy       | `Table_G.13_weighted_home_energy.csv/.tex`          |
| Appendix G: weighted results, green electricity | `Table_G.14_weighted_green_electricity.csv/.tex`    |

**Sections 4.2–4.4 — Machine-learning prediction and assignment** (written to `results/empirical4.1_4.4/`)

| Output                              | File name                                      |
| ----------------------------------- | ---------------------------------------------- |
| Figure 3: prediction accuracy       | `Figure_3_prediction_accuracy.png`           |
| Figure 4: assignment outcomes       | `Figure_4_assignment_outcomes.png`           |
| Figure 5: quota compensation        | `Figure_5_quota_compensation.png`            |
| Figure 6: budget participation      | `Figure_6_budget_participation.png`          |
| Appendix C: XGBoost hyperparameters | `Table_C.1_xgboost_hyperparameters.csv/.tex` |
| Appendix D: prediction accuracy     | `Table_D.2_prediction_accuracy.csv`          |
| Appendix D: assignment outcomes     | `Table_D.3_assignment_outcomes.csv/.tex`     |
| Appendix D: quota accept cost       | `Table_D.4_quota_accept_cost.csv/.tex`       |
| Appendix D: budget participation    | `Table_D.5_budget_participation.csv/.tex`    |

**Section 4.5 — Knowledge-growth simulation** (written to `results/empirical4.5/`)

| Output                                  | File name                                 |
| --------------------------------------- | ----------------------------------------- |
| Figure 7: knowledge-growth trajectories | `Figure_7_knowledge_growth.png`         |
| Summary statistics                      | `Figure_7_knowledge_growth_summary.csv` |

**Appendix G — Post-stratification weighted ML analysis** (written to `results/empirical4.1_4.4_weighted/` and `results/empirical4.5_weighted/`)

| Output                                    | File name                               |
| ----------------------------------------- | --------------------------------------- |
| Figure G.3: weighted prediction accuracy  | `Figure_G.3_prediction_accuracy.png`  |
| Figure G.4: weighted assignment outcomes  | `Figure_G.4_assignment_outcomes.png`  |
| Figure G.5: weighted quota compensation   | `Figure_G.5_quota_compensation.png`   |
| Figure G.6: weighted budget participation | `Figure_G.6_budget_participation.png` |
| Figure G.7: weighted knowledge growth     | `Figure_G.7_knowledge_growth.png`     |

### Repository structure

```
revision_package/
├── data/
│   ├── raw/energy_wta.csv          # Survey data (1,487 Chinese households)
│   └── temp/                        # Generated intermediate files
├── R/                               # Modular R scripts for Section 4.1
├── python/empirical_4_1_4_5/        # Python package for Sections 4.2–4.5
├── results/                         # All outputs (created automatically)
├── run_all.R                        # Entry point: R workflow (Section 4.1)
├── run_empirical4_1_and_4_5.py      # Entry point: Python workflow (Sections 4.2–4.5)
├── install.R                        # Installs R packages at pinned versions
├── requirements.txt                 # Python packages at pinned versions
├── redraw_figure_g3.py              # Utility: regenerate Figure G.3 from saved data
├── redraw_figure_g7.py              # Utility: regenerate Figure G.7 from saved data
└── archive/historical_sources/      # Original ad hoc scripts (not part of workflow)
```

## Instructions and computational requirements

### Software requirements

- **R 4.5.2**
- **Python 3.10.0**

### Step 1 — Install R packages

```powershell
Rscript install.R
```

Installs 5 direct packages and 32 transitive dependencies at tested versions.
If a package fails at the pinned version, omit the version constraint — the
latest CRAN release is typically compatible.

Direct dependencies: `MASS` (7.3-65), `dplyr` (1.2.0), `brant` (0.3-0),
`survey` (4.5), `ggplot2` (4.0.3).

### Step 2 — Install Python packages

```powershell
pip install -r requirements.txt
```

Installs 6 direct packages and 16 transitive dependencies at tested versions.

Direct dependencies: `pandas` (2.0.0), `numpy` (1.23.5), `scikit-learn`
(1.6.1), `xgboost` (3.1.2), `optuna` (4.7.0), `matplotlib` (3.8.2).

### Step 3 — Run the R workflow (Section 4.1)

From the project root:

```powershell
Rscript run_all.R
```

Or from within an R session:

```r
source("run_all.R", encoding = "UTF-8")
```

This reproduces Section 4.1 (ordered logit models, robustness checks,
post-stratification weighted tables) and writes all Section 4.1 and
Appendix B–G table outputs to `results/empirical3/`. It also generates
`data/temp/energy_wta_with_post_weights.csv`, which Step 4 requires.

### Step 4 — Run the Python workflow (Sections 4.1–4.5)

```powershell
python run_empirical4_1_and_4_5.py
```

**Step 3 must complete before Step 4.**

Reproduces Sections 4.2–4.4 (ML prediction, assignment optimisation) and
Section 4.5 (knowledge-growth simulation). Main outputs go to
`results/empirical4.1_4.4/`; simulation outputs go to `results/empirical4.5/`.

Default settings: 50 Optuna trials per XGBoost classifier, 100 Optuna trials
per WTA regressor, 3-fold cross-validation, 1,000 Monte Carlo draws for the
random baseline, 200 knowledge-growth simulation iterations.

### Optional: Appendix G weighted ML analysis

```powershell
python run_empirical4_1_and_4_5.py --weighted
```

Adds Figures G.3–G.6 to `results/empirical4.1_4.4_weighted/` and Figure G.7
to `results/empirical4.5_weighted/`. Uses 20 Optuna trials per weighted
classifier and 20 per weighted WTA regressor by default.

If needed, Figures G.3 and G.7 can be regenerated from the saved CSV data
without re-running the full analysis:

```powershell
python redraw_figure_g3.py
python redraw_figure_g7.py
```

### Optional: legacy WTA hyperparameters

To fix only the WTA regressors to the values reported in Table C.1 of the
manuscript (keeping the PrefAlt classifiers on the standard Optuna workflow):

```powershell
python run_empirical4_1_and_4_5.py --use-legacy-wta-params --output-subdir "empirical4.1_4.4_legacy_wta" --skip-simulation
```

### Smoke test (fast verification)

```powershell
python run_empirical4_1_and_4_5.py --n-trials 1 --reg-n-trials 1 --cv 2 --random-iterations 5
```

## Environment notes

All paths in the scripts are resolved relative to the project root at runtime.
No machine-specific paths are required.

If R packages are installed in a non-standard location, set `R_PACKAGE_DIR`
before running:

```powershell
$env:R_PACKAGE_DIR = "C:\path\to\your\R\packages"
Rscript run_all.R
```

If `Rscript` is not on `PATH`, pass it explicitly:

```powershell
python run_empirical4_1_and_4_5.py --rscript "C:\path\to\R\bin\Rscript.exe"
```

After the R run, `results/empirical3/logs/session_info.txt` records the exact
R session information. Intermediate files (post-stratification data,
train/test splits, logit probabilities, WTA predictions, simulated test data)
are written to `data/temp/`.

## Data availability

The analysis data are in `data/raw/energy_wta.csv` (1,487 Chinese household
survey responses) and are included in this package. No additional data
download is required. Column names are trimmed on load; this corrects a
trailing space in the original `MADT ` column.
