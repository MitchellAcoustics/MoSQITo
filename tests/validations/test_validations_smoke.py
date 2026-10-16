"""Smoke test: every script under ``validations/`` must run to completion.

The scripts are executed in-process with the Agg backend. ``plt.show`` and
``Figure.savefig`` are patched to no-ops so the tracked reference PNGs under
``validations/**/output/`` are never overwritten.

This only checks that the scripts *run*. It does not check the numbers they
compute: the scripts contain no assertions.
"""

import os
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATIONS = REPO_ROOT / "validations"

# Scripts that take minutes to run (measured on a single CPU core).
SLOW = {
    "sq_metrics/loudness_ecma/hearing_model_validation.py",
    "sq_metrics/roughness_ecma/validation_roughness_ecma.py",
    "sq_metrics/roughness_ecma/comparison_roughness_ecma.py",
    "sq_metrics/loudness_zwtv/validation_loudness_zwtv.py",
}


def _discover():
    params = []
    for path in sorted(VALIDATIONS.rglob("*.py")):
        # ``input/`` folders hold reference data and helpers, not scripts
        if "input" in path.relative_to(VALIDATIONS).parts:
            continue
        rel = path.relative_to(VALIDATIONS).as_posix()
        marks = [pytest.mark.slow] if rel in SLOW else []
        params.append(pytest.param(path, id=rel, marks=marks))
    return params


@pytest.mark.parametrize("script", _discover())
def test_validation_script_runs(script, monkeypatch):
    # Several scripts do ``from input.references import ...``; all of them use
    # the same top-level name, so drop cached modules between scripts.
    for name in [n for n in sys.modules if n == "input" or n.startswith("input.")]:
        monkeypatch.delitem(sys.modules, name)

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.syspath_prepend(str(REPO_ROOT))

    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        plt.close("all")
        for name in [n for n in sys.modules if n == "input" or n.startswith("input.")]:
            sys.modules.pop(name, None)
