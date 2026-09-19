"""Differential tests for `mosqito_rs.roughness_dw*`, through the Python API.

The Zwicker & Fastl standards-anchored gate lives at the Rust level
(`crates/mosqito-core/tests/conformance_roughness_dw.rs`, ±0.1 asper, >=90%
of the 84-point (fc, fmod) grid — MoSQITo's own `roughness_dw` does not
reach 100% compliance either, confirmed directly against the installed
package). This file differentially checks the Python wrapper glue against an
installed `mosqito`.

`roughness_dw`'s pipeline chains several FFT/IFFT passes
(`comp_spectrum`'s own FFT, plus per-channel excitation/envelope FFTs in
`_roughness_dw_main_calc`), so rustfft and NumPy's FFT — different
algorithms — accumulate slightly different floating-point rounding over
that many passes; ~1e-4 relative agreement is what's actually observed
between the two (`crates/mosqito-core/tests/golden_roughness_dw.rs` checks
the same computation against a captured Python snapshot at 1e-6, so this
is specifically about two *live* FFT implementations disagreeing at the
rounding level, not a port defect — the same characteristic
`golden_utils.rs`'s comp_spectrum dB comparison documents).
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs
from mosqito_rs import am_sine_generator


@pytest.mark.differential
def test_roughness_dw_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 48000
    duration = 0.6
    time = np.arange(0, duration, 1 / fs)
    xmod = np.sin(2 * np.pi * 70 * time)
    stimulus, _ = am_sine_generator(xmod, fs, fc=1000, spl_level=60)

    r_rs, spec_rs, bark_rs, time_rs = mosqito_rs.roughness_dw(stimulus, fs, overlap=0.5)
    r_py, spec_py, bark_py, time_py = mosqito.sq_metrics.roughness_dw(stimulus, fs, overlap=0.5)

    # R_spec's per-channel values span down to genuinely near-zero (a
    # channel with negligible excitation) — an absolute floor keeps the
    # comparison meaningful there the same way `golden_utils.rs`'s
    # comp_spectrum dB comparison needs one, rather than rtol alone flagging
    # two independently-computed near-zero floats as wildly different.
    np.testing.assert_allclose(r_rs, r_py, rtol=2e-4, atol=1e-9)
    np.testing.assert_allclose(spec_rs, spec_py, rtol=2e-4, atol=1e-3)
    np.testing.assert_allclose(bark_rs, bark_py, rtol=1e-9)
    np.testing.assert_allclose(time_rs, time_py, rtol=1e-9)


@pytest.mark.differential
def test_roughness_dw_freq_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 48000
    duration = 0.2
    time = np.arange(0, duration, 1 / fs)
    xmod = np.sin(2 * np.pi * 70 * time)
    stimulus, _ = am_sine_generator(xmod, fs, fc=1000, spl_level=60)

    spec, freqs = mosqito.sound_level_meter.comp_spectrum(stimulus, fs, db=False)
    spectrum = np.abs(spec)

    r_rs, spec_rs, bark_rs = mosqito_rs.roughness_dw_freq(spectrum, freqs)
    r_py, spec_py, bark_py = mosqito.sq_metrics.roughness_dw_freq(spectrum, freqs)

    np.testing.assert_allclose(r_rs, r_py, rtol=2e-4, atol=1e-9)
    np.testing.assert_allclose(spec_rs, spec_py, rtol=2e-4, atol=1e-3)
    np.testing.assert_allclose(bark_rs, bark_py, rtol=1e-9)
