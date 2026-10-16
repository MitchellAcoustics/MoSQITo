"""Load validation scripts as modules for the tests in this folder."""

import runpy
import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATIONS = REPO_ROOT / "validations"


def purge_input_modules():
    """Several scripts import from their own ``input`` package; they all share
    that top-level name, so cached modules must not leak between scripts."""
    for name in [n for n in sys.modules if n == "input" or n.startswith("input.")]:
        del sys.modules[name]


@lru_cache(maxsize=None)
def load_script(rel_path):
    """Load a validation script without running its ``__main__`` block and
    return its namespace (functions, reference data)."""
    script = VALIDATIONS / rel_path
    saved_path = list(sys.path)
    sys.path[:0] = [str(script.parent), str(REPO_ROOT)]
    purge_input_modules()
    try:
        return runpy.run_path(str(script), run_name=f"validations.{script.stem}")
    finally:
        purge_input_modules()
        sys.path[:] = saved_path
