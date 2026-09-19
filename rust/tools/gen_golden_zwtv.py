#!/usr/bin/env python3
"""Export reference outputs for the loudness_zwtv pipeline's own stages from
the installed `mosqito` package.

Covers `_nl_loudness` (the sequential nonlinear-decay recurrence — including
its col=0 wraparound-dependent behaviour, exercised directly with small
random `core_loudness` matrices rather than only through the full pipeline,
so a divergence there is attributable) and `_third_octave_levels` (the ISO
532-1 Table A.1/A.2 filter bank), each in isolation, plus a couple of
short full-pipeline `loudness_zwtv` cases end to end.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden_zwtv.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "crates/mosqito-core/tests/golden_zwtv.json"
)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mosqito.sq_metrics.loudness.loudness_zwtv._nonlinear_decay import _nl_loudness
from mosqito.sq_metrics.loudness.loudness_zwtv._third_octave_levels import (
    _third_octave_levels,
)
from mosqito.sq_metrics.loudness.loudness_zwtv.loudness_zwtv import loudness_zwtv

RNG = np.random.default_rng(20240920)


def main() -> None:
    cases = {"nl_loudness": [], "third_octave_levels": [], "loudness_zwtv": []}

    # --- _nl_loudness: small random (21, ntime) core_loudness matrices,
    # including short ones where the col=0 wraparound to the *last* column
    # is most visible in the output. Values are plausible nm outputs: mostly
    # small non-negative numbers, occasionally larger. ------------------
    for ntime in (2, 3, 5, 10, 50):
        for trial in range(6):
            core_loudness = RNG.uniform(0.0, 15.0, size=(21, ntime)) * (
                RNG.uniform(size=(21, ntime)) < 0.7
            )
            out = _nl_loudness(core_loudness.copy())
            cases["nl_loudness"].append(
                {
                    "core_loudness": core_loudness.tolist(),
                    "nl_loudness": out.tolist(),
                }
            )

    # --- _third_octave_levels: short synthetic 48 kHz signals -------------
    fs = 48000
    for seed_sig in (
        RNG.uniform(-1.0, 1.0, size=4800),  # 0.1 s broadband noise
        0.5 * np.sin(2 * np.pi * 1000 * np.arange(2400) / fs),  # 1 kHz tone, 0.05 s
        np.zeros(2400),
    ):
        levels, time_axis, freq = _third_octave_levels(seed_sig, fs)
        cases["third_octave_levels"].append(
            {
                "sig": seed_sig.tolist(),
                "levels": levels.tolist(),
                "time_axis": time_axis.tolist(),
            }
        )

    # --- full loudness_zwtv pipeline: short signals only, for tractable
    # golden-file size (the DIN/ISO-scale conformance corpus is checked
    # directly against the reference wavs at the Rust test level instead). -
    rng2 = np.random.default_rng(1)
    for dur, freq0 in ((0.05, 1000.0), (0.1, 500.0)):
        n = int(fs * dur)
        t = np.arange(n) / fs
        stimulus = 0.5 * (1 + np.sin(2 * np.pi * freq0 * t))
        rms = np.sqrt(np.mean(stimulus**2))
        ampl = 0.00002 * 10 ** (60 / 20) / rms
        stimulus = (stimulus * ampl).astype(np.float64)
        n_out, n_spec, bark, time_axis = loudness_zwtv(stimulus, fs, field_type="free")
        cases["loudness_zwtv"].append(
            {
                "signal": stimulus.tolist(),
                "fs": fs,
                "N": n_out.tolist(),
                "N_specific": n_spec.tolist(),
                "time_axis": time_axis.tolist(),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases))
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes), "
        f"{len(cases['nl_loudness'])} nl_loudness, "
        f"{len(cases['third_octave_levels'])} third_octave_levels, "
        f"{len(cases['loudness_zwtv'])} loudness_zwtv cases"
    )


if __name__ == "__main__":
    main()
