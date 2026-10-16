"""Smoke test: the validation scripts whose verdicts are not asserted must run.

Scripts in ASSERTED_SCRIPTS are exercised case by case in
test_validations_compliance.py; every other script under ``validations/`` is
run here as ``__main__``, in-process (plots are suppressed by conftest.py).

This only checks that these scripts *run*. They have no pass criterion, so
the numbers they compute are not checked.
"""

import runpy

import pytest

from ._scripts import ASSERTED_SCRIPTS, REPO_ROOT, VALIDATIONS, purge_input_modules

# Scripts that take minutes to run (measured on a single CPU core).
SLOW = {
    "sq_metrics/loudness_ecma/hearing_model_validation.py",
    "sq_metrics/roughness_ecma/comparison_roughness_ecma.py",
}


def _discover():
    params = []
    for path in sorted(VALIDATIONS.rglob("*.py")):
        rel = path.relative_to(VALIDATIONS)
        # ``input/`` folders hold reference data and helpers, not scripts
        if "input" in rel.parts or rel.as_posix() in ASSERTED_SCRIPTS:
            continue
        marks = [pytest.mark.slow] if rel.as_posix() in SLOW else []
        params.append(pytest.param(path, id=rel.as_posix(), marks=marks))
    return params


@pytest.mark.parametrize("script", _discover())
def test_validation_script_runs(script, monkeypatch):
    # Several scripts do ``from input.references import ...``; they all share
    # that top-level name, so cached modules must not leak between scripts.
    purge_input_modules()
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        purge_input_modules()
