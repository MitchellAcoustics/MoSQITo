#!/usr/bin/env python3
"""Export reference outputs for the ISO 532-1 stationary loudness core.

Unlike `gen_golden_noct.py`, `_main_loudness` and `_calc_slopes` are imported
directly from the installed `mosqito` package rather than transcribed: their
control flow (an unusual low-frequency threshold search in `_main_loudness`,
a multi-branch slope-attachment loop in `_calc_slopes`) is intricate enough
that a hand transcription would itself need verifying, whereas importing the
real functions cannot drift from what MoSQITo actually computes. Run with the
full dev environment installed (`uv pip install -e .` from the repo root, so
`mosqito` and its matplotlib/pyuff/pandas/openpyxl dependencies resolve).

Covers hundreds of random spectra, not just the reference corpus, because
`_main_loudness`'s low-frequency correction only checks for a threshold
transition across 7 of its 8 RAP levels (see main_loudness.rs's module doc) —
a case the reference corpus signals may not exercise, and only random spectra
reaching that edge would evidence it.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden_zwst.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/mosqito-core/tests/golden_zwst.json"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import _calc_slopes
from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness

RNG = np.random.default_rng(20240918)


def random_spectrum(low_lo=20.0, low_hi=90.0, rest_lo=0.0, rest_hi=100.0) -> np.ndarray:
    """A plausible third-octave spectrum: 28 bands, dB SPL."""
    spec = RNG.uniform(rest_lo, rest_hi, size=28)
    spec[0:11] = RNG.uniform(low_lo, low_hi, size=11)
    return spec


def main() -> None:
    cases = {"main_loudness": [], "calc_slopes": []}

    # --- main_loudness: broad random coverage + targeted edge cases --------
    spectra = [random_spectrum() for _ in range(300)]
    # Edge cases near the RAP transition boundaries (including the ceiling
    # this port's module doc calls out) and near each RAP threshold exactly.
    for level in (44.0, 45.0, 46.0, 54.0, 55.0, 70.0, 71.0, 89.0, 90.0, 99.0, 100.0, 115.0, 119.0, 119.999):
        spec = random_spectrum()
        spec[0:11] = level
        spectra.append(spec)
    # A silent spectrum and a loud-but-uniform one.
    spectra.append(np.zeros(28))
    spectra.append(np.full(28, 60.0))

    for spec in spectra:
        for field_type in ("free", "diffuse"):
            nm = _main_loudness(spec.copy(), field_type)
            cases["main_loudness"].append({
                "spec": spec.tolist(),
                "field_type": field_type,
                "nm": nm.tolist(),
            })

    # --- calc_slopes: fed from the nm outputs above, plus direct nm arrays -
    for case in cases["main_loudness"]:
        nm = np.array(case["nm"])
        n, n_specific = _calc_slopes(nm)
        cases["calc_slopes"].append({
            "nm": nm.tolist(),
            "N": float(n),
            "N_specific": n_specific.tolist(),
        })

    # A handful of synthetic nm arrays exercising monotone-rising, flat, and
    # sharply-falling patterns directly (bypassing main_loudness), to probe
    # calc_slopes' branches independently of what main_loudness happens to
    # produce.
    synthetic_nm = [
        np.linspace(0.0, 20.0, 21),  # monotonically rising
        np.linspace(20.0, 0.0, 21),  # monotonically falling
        np.concatenate([np.full(10, 15.0), np.full(11, 0.5)]),  # one sharp drop
        np.concatenate([np.full(5, 0.2), np.full(5, 25.0), np.full(11, 0.1)]),  # rise then sharp fall
        np.zeros(21),
    ]
    for nm in synthetic_nm:
        n, n_specific = _calc_slopes(nm)
        cases["calc_slopes"].append({
            "nm": nm.tolist(),
            "N": float(n),
            "N_specific": n_specific.tolist(),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes), "
          f"{len(cases['main_loudness'])} main_loudness cases, "
          f"{len(cases['calc_slopes'])} calc_slopes cases")


if __name__ == "__main__":
    main()
