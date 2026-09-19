"""Conformance test for `mosqito_rs.roughness_ecma`.

The full ECMA-418-2 Annex C + Zwicker-Fastl conformance grid (105 points,
~80 s) runs at the Rust level
(`crates/mosqito-core/tests/conformance_roughness_ecma.rs`) — see that
file's module doc for why installed `mosqito`'s own `roughness_ecma` is
*not* used as a reference anywhere in this port's test suite (its
`_lowpass_filter` bug and uncalibrated `c_R` make its own output
non-conformant with the standard; see `DEVIATIONS.md`). This file adds one
fast conformance point through the Python API specifically, rather than a
full grid, to keep routine test runs quick.
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs


@pytest.mark.conformance
def test_roughness_ecma_matches_ecma_418_2_annex_c_1khz_70hz():
    # ECMA-418-2 Annex C: fc=1000 Hz, fmod=70 Hz -> ref_ecma = 1.000 asper_HMS
    # (the standard's own peak-roughness reference point for a 1 kHz tone).
    fs = 48000.0
    duration = 1.5
    fc = 1000.0
    fmod = 70.0
    level_db = 60.0

    t = np.arange(int(duration * fs)) / fs
    xmod = np.sin(2 * np.pi * fmod * t)
    y = (1 + xmod) * np.sin(2 * np.pi * fc * t)
    p_ref = 20e-6
    a_rms = p_ref * 10 ** (level_db / 20)
    y = y * (a_rms / np.std(y))

    r, r_time, r_spec, bark_axis, time_axis = mosqito_rs.roughness_ecma(y.astype(np.float64), fs)

    assert bark_axis.shape == (53,)
    assert r_spec.shape == (53,)
    assert r_time.shape == time_axis.shape

    ref_ecma = 1.000
    ref_zf = 0.9908
    assert abs(r - ref_ecma) / ref_ecma < 0.30
    assert abs(r - ref_zf) < 0.1
