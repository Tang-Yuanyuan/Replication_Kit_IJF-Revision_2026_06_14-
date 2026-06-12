from __future__ import annotations

import os
import sys
from pathlib import Path


_pkg_dir_env = os.environ.get("PYTHON_PACKAGE_DIR")
if _pkg_dir_env:
    _pkg_dir = Path(_pkg_dir_env)
    if _pkg_dir.exists() and str(_pkg_dir) not in sys.path:
        sys.path.insert(0, str(_pkg_dir))


__all__ = ["main"]


def main() -> None:
    from .workflow import main as workflow_main

    workflow_main()
