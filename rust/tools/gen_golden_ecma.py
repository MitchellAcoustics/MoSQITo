#!/usr/bin/env python3
"""Export reference outputs for the ECMA-418-2 loudness pipeline's own
stages, and the full pipeline on short signals, from the installed
`mosqito` package.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden_ecma.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "crates/mosqito-core/tests/golden_ecma_loudness.json"
)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import (
    _auditory_filters_centre_freq,
)
from mosqito.sq_metrics.loudness.loudness_ecma._gammatone import _gammatone
from mosqito.sq_metrics.loudness.loudness_ecma._nonlinearity import _nonlinearity
from mosqito.sq_metrics.loudness.loudness_ecma.loudness_ecma import loudness_ecma

RNG = np.random.default_rng(20240921)


def complex_to_json(c: np.ndarray) -> list[list[float]]:
    return [[float(v.real), float(v.imag)] for v in c]


def main() -> None:
    cases: dict = {
        "centre_freq": [],
        "gammatone": [],
        "nonlinearity": [],
        "loudness_ecma": [],
    }

    centre_freq = _auditory_filters_centre_freq()
    cases["centre_freq"] = centre_freq.tolist()

    for band in (0, 1, 13, 26, 40, 52):
        bm, am = _gammatone(centre_freq[band], k=5, fs=48000.0)
        cases["gammatone"].append(
            {
                "band": band,
                "freq": float(centre_freq[band]),
                "bm": complex_to_json(bm),
                "am": complex_to_json(am),
            }
        )

    p_values = np.concatenate(
        [
            np.array([0.0]),
            10.0 ** RNG.uniform(-6, 1, size=50),
        ]
    )
    a_prime = _nonlinearity(p_values)
    cases["nonlinearity"] = [
        {"p": float(p), "a_prime": float(a)} for p, a in zip(p_values, a_prime)
    ]

    fs = 48000
    for dur, freq0 in ((0.1, 1000.0), (0.15, 500.0)):
        n = int(fs * dur)
        t = np.arange(n) / fs
        stimulus = 0.5 * (1 + np.sin(2 * np.pi * freq0 * t))
        rms = np.sqrt(np.mean(stimulus**2))
        ampl = 0.00002 * 10 ** (60 / 20) / rms
        stimulus = (stimulus * ampl).astype(np.float64)
        n_val, n_time, n_spec, bark_axis, time_array = loudness_ecma(
            stimulus.copy(), fs, sb=2048, sh=1024
        )
        cases["loudness_ecma"].append(
            {
                "signal": stimulus.tolist(),
                "fs": fs,
                "sb": 2048,
                "sh": 1024,
                "N": float(n_val),
                "N_time": np.asarray(n_time).tolist(),
                "N_specific": np.asarray(n_spec).tolist(),
                "time_axis": np.asarray(time_array[0]).tolist(),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases))
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes), "
        f"{len(cases['gammatone'])} gammatone, "
        f"{len(cases['nonlinearity'])} nonlinearity, "
        f"{len(cases['loudness_ecma'])} loudness_ecma cases"
    )


if __name__ == "__main__":
    main()
