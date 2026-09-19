"""Differential tests for `mosqito_rs`'s signal generators, through the
Python API.

`am_noise_generator` is not bit-for-bit comparable to MoSQITo's Python (see
its docstring and `DEVIATIONS.md`: both draw from OS-entropy-seeded or
explicitly-seeded RNGs that don't match `numpy.random.default_rng`'s stream)
— only its statistical properties (achieved RMS level, modulation index) are
checked instead of an exact comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs


@pytest.mark.differential
def test_sine_wave_generator_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    sig_rs, time_rs = mosqito_rs.sine_wave_generator(48000, 0.05, 200.0, 65.0)
    sig_py, time_py = mosqito.utils.sine_wave_generator(48000, 0.05, 200.0, 65.0)
    # atol matters here: near a sine's zero crossings the values themselves
    # are within a few ULPs of zero, where rtol alone flags meaningless
    # floating-point noise as a large relative error.
    np.testing.assert_allclose(sig_rs, sig_py, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(time_rs, time_py, rtol=1e-9)


@pytest.mark.differential
def test_am_sine_generator_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 48000
    t = np.arange(2400) / fs
    xmod = 0.6 * np.sin(2 * np.pi * 10.0 * t)

    y_rs, m_rs = mosqito_rs.am_sine_generator(xmod, fs, fc=1000.0, spl_level=70.0)
    y_py, m_py = mosqito.utils.am_sine_generator(xmod, fs, fc=1000.0, spl_level=70.0)
    np.testing.assert_allclose(y_rs, y_py, rtol=1e-9, atol=1e-12)
    assert m_rs == pytest.approx(m_py)


@pytest.mark.differential
def test_fm_sine_generator_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 48000
    t = np.arange(2400) / fs
    xmod = 0.6 * np.sin(2 * np.pi * 10.0 * t)

    y_rs, f_rs, delta_rs, m_rs = mosqito_rs.fm_sine_generator(
        xmod, fs, fc=500.0, k=200.0, spl_level=70.0
    )
    y_py, f_py, delta_py, m_py = mosqito.utils.fm_sine_generator(
        xmod, fs, fc=500.0, k=200.0, spl_level=70.0
    )
    np.testing.assert_allclose(y_rs, y_py, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(f_rs, f_py, rtol=1e-9)
    assert delta_rs == pytest.approx(delta_py)
    assert m_rs == pytest.approx(m_py)


def test_am_noise_generator_reaches_the_requested_level_and_is_reproducible():
    fs = 48000
    t = np.arange(20000) / fs
    xmod = 0.5 * np.sin(2 * np.pi * 4.0 * t)

    y1, m1 = mosqito_rs.am_noise_generator(xmod, spl_level=60.0, seed=42)
    y2, m2 = mosqito_rs.am_noise_generator(xmod, spl_level=60.0, seed=42)
    np.testing.assert_array_equal(y1, y2)
    assert m1 == m2 == pytest.approx(0.5)

    want_std = 20e-6 * 10 ** (60.0 / 20)
    got_std = np.std(y1)
    assert got_std == pytest.approx(want_std, rel=1e-6)

    # A different seed gives a different (but still correctly normalised)
    # signal — confirms the seed is actually threaded through.
    y3, _ = mosqito_rs.am_noise_generator(xmod, spl_level=60.0, seed=7)
    assert not np.array_equal(y1, y3)
