# Reproduction Engineering Plan

This note tracks the light reorganization of the reproduction package.

## Current Organization

- `run_all.R` remains a root-level entry point for empirical Section 4.1.
- `run_empirical4_2_3.py` is now a root-level Python entry point for empirical Sections 4.2 and 4.3.
- `R/` contains the modular R implementation for Section 4.1.
- `python/empirical_4_2.py` remains the main Python implementation for Sections 4.2 and 4.3.
- Section 4.1 tables, figures, and logs are written to `results/empirical4.1/`.
- Section 4.2 and 4.3 outputs continue to use the existing Python output arguments under `results/`.
- Intermediate data shared across R and Python is written to `data/temp/`.

## Completed

1. Routed Section 4.1 table, figure, and log outputs to `results/empirical4.1/`.
2. Updated the Section 4.1 R entry messages so the workflow is clearly labeled as empirical 4.1.
3. Added a root-level Python entry script, `run_empirical4_2_3.py`, which forwards to `python/empirical_4_2.py`.
4. Updated documentation paths for the new Section 4.1 output directory.
5. Routed generated intermediate data to `data/temp/`, leaving `results/` for formal tables, figures, and logs.

## Near-Term Plan

1. Keep root-level entry points stable for reproduction users.
2. Keep implementation code under `R/` and `python/`.
3. Avoid a large Python refactor until the current outputs are stable.
4. When ready, split `python/empirical_4_2.py` along functional boundaries:
   - data loading and preprocessing
   - model training and prediction
   - tables and figure export
   - quota and budget simulations
   - Section 4.3 knowledge-growth simulation

## Suggested Commands

Run Section 4.1:

```powershell
& "D:\R-4.5.2\bin\Rscript.exe" "D:\RUC\revision_package\run_all.R"
```

Run Sections 4.2 and 4.3:

```powershell
& "D:\Python\python.exe" "D:\RUC\revision_package\run_empirical4_2_3.py" --root "D:\RUC\revision_package"
```

Fast smoke test for Sections 4.2 and 4.3:

```powershell
& "D:\Python\python.exe" "D:\RUC\revision_package\run_empirical4_2_3.py" --root "D:\RUC\revision_package" --n-trials 1 --reg-n-trials 1 --cv 2 --random-iterations 5 --simulation-iterations 1
```
