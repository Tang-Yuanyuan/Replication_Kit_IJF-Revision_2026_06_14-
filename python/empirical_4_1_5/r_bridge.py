from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        script = cmd[1] if len(cmd) > 1 else "R script"
        raise RuntimeError(
            f"R script failed: {script}\n{detail}"
        ) from exc


def run_r_logit(root: Path, rscript: Path, output_subdir: str) -> None:
    script = root / "R" / "07_logit_4_1.R"
    _run([str(rscript), str(script), str(root), output_subdir])

def run_r_ologit_wta(root: Path, rscript: Path, output_subdir: str) -> None:
    script = root / "R" / "08_ologit_wta_4_1.R"
    _run([str(rscript), str(script), str(root), output_subdir])

def run_r_simulated_predictions(
    root: Path,
    rscript: Path,
    train_subdir: str,
    sim_subdir: str,
) -> None:
    script = root / "R" / "09_simulated_predictions_5.R"
    _run([str(rscript), str(script), str(root), train_subdir, sim_subdir])


def run_r_logit_weighted(root: Path, rscript: Path, read_subdir: str, write_subdir: str) -> None:
    script = root / "R" / "07_logit_4_1_weighted.R"
    _run([str(rscript), str(script), str(root), read_subdir, write_subdir])


def run_r_ologit_wta_weighted(root: Path, rscript: Path, read_subdir: str, write_subdir: str) -> None:
    script = root / "R" / "08_ologit_wta_4_1_weighted.R"
    _run([str(rscript), str(script), str(root), read_subdir, write_subdir])


def run_r_simulated_predictions_weighted(
    root: Path,
    rscript: Path,
    train_subdir: str,
    sim_subdir: str,
) -> None:
    script = root / "R" / "09_simulated_predictions_5_weighted.R"
    _run([str(rscript), str(script), str(root), train_subdir, sim_subdir])

