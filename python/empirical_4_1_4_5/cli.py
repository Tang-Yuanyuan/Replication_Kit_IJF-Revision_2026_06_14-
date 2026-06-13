from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _find_rscript() -> Path | None:
    found = shutil.which("Rscript")
    return Path(found) if found else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce empirical Sections 4.1–4.5.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for each XGBoost model.",
    )
    parser.add_argument(
        "--reg-n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials for each XGBoost WTA regressor.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Cross-validation folds for Optuna objective.",
    )
    parser.add_argument(
        "--random-iterations",
        type=int,
        default=1000,
        help="Monte Carlo iterations for the random baseline.",
    )
    parser.add_argument(
        "--simulation-iterations",
        type=int,
        default=200,
        help="Knowledge-growth simulation iterations for Section 4.5.",
    )
    parser.add_argument(
        "--output-subdir",
        default="empirical4.1_4.4",
        help="Subdirectory under results/ for Sections 4.1–4.4 outputs.",
    )
    parser.add_argument(
        "--sim-output-subdir",
        default=None,
        help="Subdirectory under results/ for Section 4.5 outputs. Defaults to empirical4.5.",
    )
    parser.add_argument(
        "--use-legacy-wta-params",
        action="store_true",
        help="Use the original WTA hyperparameters recorded in the manuscript table.",
    )
    parser.add_argument(
        "--also-legacy",
        action="store_true",
        help="Always run the legacy WTA comparison without prompting.",
    )
    parser.add_argument(
        "--skip-main",
        action="store_true",
        help=(
            "Skip the main (unweighted) analysis Steps 2–7. "
            "Useful when outputs already exist and you only want --weighted "
            "and/or --also-legacy re-runs."
        ),
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Skip the Section 4.5 knowledge-growth simulation.",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Also run post-stratification weighted analysis (Sections 4.1–4.5, Appendix G).",
    )
    parser.add_argument(
        "--weighted-n-trials",
        type=int,
        default=20,
        help="Optuna trials for each weighted XGBoost classifier (default: 20).",
    )
    parser.add_argument(
        "--weighted-reg-n-trials",
        type=int,
        default=20,
        help="Optuna trials for each weighted XGBoost WTA regressor (default: 20).",
    )
    parser.add_argument(
        "--weighted-output-subdir",
        default="empirical4.1_4.4_weighted",
        help="Subdirectory under results/ for weighted Sections 4.1–4.4 outputs.",
    )
    parser.add_argument(
        "--rscript",
        type=Path,
        default=_find_rscript(),
        help="Path to Rscript executable. Auto-detected from PATH if not specified.",
    )
    args = parser.parse_args()
    if args.rscript is None:
        print("Rscript not found in PATH.")
        while True:
            raw = input("Please enter the full path to your Rscript executable: ").strip().strip('"')
            candidate = Path(raw)
            if candidate.is_file():
                args.rscript = candidate
                break
            print(f"  Path not found: {candidate!r}. Please try again.")
    return args
