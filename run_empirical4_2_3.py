from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
IMPLEMENTATION_SCRIPT = PROJECT_ROOT / "python" / "empirical_4_2.py"


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("empirical_4_2", IMPLEMENTATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Python implementation: {IMPLEMENTATION_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()
