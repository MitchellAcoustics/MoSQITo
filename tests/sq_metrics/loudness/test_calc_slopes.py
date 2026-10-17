# -*- coding: utf-8 -*-
import numpy as np
import pytest

from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import _calc_slopes
from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness


def _core(n_frames=6):
    spec = np.random.default_rng(3).uniform(20, 90, (28, n_frames))
    return _main_loudness(spec, "free")


@pytest.mark.loudness_zwst
def test_shapes():
    n, s = _calc_slopes(_core(6))
    assert n.shape == (6,) and s.shape == (240, 6)
    n1, s1 = _calc_slopes(_core(6)[:, 0])
    assert np.ndim(n1) == 0 and s1.shape == (240,)


@pytest.mark.loudness_zwst
def test_single_frame_equals_column_of_many():
    core = _core(6)
    n, s = _calc_slopes(core)
    for i in range(6):
        n1, s1 = _calc_slopes(core[:, i])
        assert n1 == n[i]
        np.testing.assert_allclose(s1, s[:, i], rtol=0, atol=1e-12)


@pytest.mark.loudness_zwst
def test_input_untouched_and_silence():
    core = _core(4)
    before = core.copy()
    _calc_slopes(core)
    np.testing.assert_array_equal(core, before)
    n, s = _calc_slopes(np.zeros((21, 3)))
    assert not n.any() and not s.any()


@pytest.mark.loudness_zwst
def test_total_is_rounded_as_in_iso_532_1():
    n, _ = _calc_slopes(_core(20))
    quantum = np.where(n <= 16, 1e-3, 1e-2)
    np.testing.assert_allclose(n / quantum, np.round(n / quantum), atol=1e-6)


# ------------------------------------------------ compiled vs vectorised
from mosqito.sq_metrics.loudness.loudness_zwst import _calc_slopes as _mod  # noqa: E402
from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import (  # noqa: E402
    _calc_slopes_frames,
    _calc_slopes_vectorised,
)
from mosqito.utils import _jit  # noqa: E402

needs_numba = pytest.mark.skipif(
    not _jit.HAS_NUMBA, reason="numba is not installed or disabled (MOSQITO_DISABLE_NUMBA=1)"
)


def _spectra(kind):
    rng = np.random.default_rng(11)
    if kind == "uniform":
        return rng.uniform(10, 110, (28, 300))
    if kind == "tilted":
        tilt = np.linspace(100, 20, 28)[:, None]
        return np.clip(tilt + rng.normal(0, 6, (28, 300)), 10, 110)
    if kind == "narrow":
        spec = np.full((28, 300), 10.0)
        band = rng.integers(0, 28, 300)
        spec[band, np.arange(300)] = rng.uniform(40, 110, 300)
        return spec
    return np.clip(rng.normal(60, 30, (28, 300)), 10, 110)


@needs_numba
@pytest.mark.loudness_zwst
@pytest.mark.parametrize("kind", ["uniform", "tilted", "narrow", "clipped"])
def test_matches_vectorised_on_realistic_input(kind):
    # _main_loudness always yields nm[20] == 0, so inputs with nm[20] > nm[0]
    # (where the original vectorised routine wraps around to the last band)
    # are out of scope.
    core = _main_loudness(_spectra(kind), "free")
    assert not core[20].any()
    n, s = _calc_slopes(core)
    n_ref, s_ref = _calc_slopes_vectorised(core)
    np.testing.assert_array_equal(n, n_ref)
    np.testing.assert_allclose(s, s_ref, rtol=0, atol=1e-12)


@needs_numba
@pytest.mark.loudness_zwst
@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_single_frame_and_non_finite_use_vectorised_routine():
    before = _calc_slopes_frames.is_compiled
    core = _core(5)
    for single in (core[:, 0], core[:, :1]):
        n, s = _calc_slopes(single)
        n_ref, s_ref = _calc_slopes_vectorised(single)
        np.testing.assert_array_equal(n, n_ref)
        np.testing.assert_array_equal(s, s_ref)
    assert _calc_slopes_frames.is_compiled == before

    bad = core.copy()
    bad[3, 2] = np.nan
    bad[5, 4] = np.inf
    n, s = _calc_slopes(bad)
    n_ref, s_ref = _calc_slopes_vectorised(bad)
    np.testing.assert_array_equal(n, n_ref)
    np.testing.assert_array_equal(s, s_ref)
    assert np.isnan(n[2])
    assert _calc_slopes_frames.is_compiled == before
