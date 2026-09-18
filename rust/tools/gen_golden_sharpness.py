#!/usr/bin/env python3
"""Export reference outputs for `sharpness_din_from_loudness` from the
installed `mosqito` package.

Covers the scalar (`S.size == 1`) branch and the segmented (`N < 0.1`
masking) branch separately, across all four weightings, using `N_specific`
arrays produced by the already-ported `loudness_zwst` core (random
third-octave spectra through `_main_loudness`/`_calc_slopes`) so the inputs
are physically plausible specific-loudness patterns rather than arbitrary
noise.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden_sharpness.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "crates/mosqito-core/tests/golden_sharpness.json"
)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import _calc_slopes
from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness
from mosqito.sq_metrics.sharpness.sharpness_din.sharpness_din_from_loudness import (
    sharpness_din_from_loudness,
)

RNG = np.random.default_rng(20240919)
WEIGHTINGS = ("din", "aures", "bismarck", "fastl")


def random_spectrum() -> np.ndarray:
    spec = RNG.uniform(0.0, 100.0, size=28)
    spec[0:11] = RNG.uniform(20.0, 90.0, size=11)
    return spec


def random_n_n_specific() -> tuple[float, np.ndarray]:
    nm = _main_loudness(random_spectrum(), "free")
    n, n_specific = _calc_slopes(nm)
    return float(n), n_specific


def main() -> None:
    cases = {"scalar": [], "segmented": []}

    # --- scalar branch: one (N, N_specific) pair per case ------------------
    for _ in range(80):
        n, n_specific = random_n_n_specific()
        for weighting in WEIGHTINGS:
            s = sharpness_din_from_loudness(n, n_specific, weighting=weighting)
            cases["scalar"].append(
                {
                    "N": n,
                    "N_specific": n_specific.tolist(),
                    "weighting": weighting,
                    "S": float(s),
                }
            )

    # --- segmented branch: several segments, including some below the
    # N < 0.1 masking threshold ---------------------------------------------
    for _ in range(20):
        pairs = [random_n_n_specific() for _ in range(6)]
        n_vec = np.array([p[0] for p in pairs])
        n_specific_mat = np.stack([p[1] for p in pairs], axis=1)  # (240, nseg)
        # Force a couple of segments below the masking threshold.
        n_vec[0] = 0.0
        n_specific_mat[:, 0] = 0.0
        n_vec[2] = 0.05
        for weighting in WEIGHTINGS:
            s = sharpness_din_from_loudness(
                n_vec.copy(), n_specific_mat.copy(), weighting=weighting
            )
            cases["segmented"].append(
                {
                    "N": n_vec.tolist(),
                    "N_specific": n_specific_mat.tolist(),
                    "weighting": weighting,
                    "S": np.asarray(s).tolist(),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases))
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes), "
        f"{len(cases['scalar'])} scalar cases, {len(cases['segmented'])} segmented cases"
    )


if __name__ == "__main__":
    main()
