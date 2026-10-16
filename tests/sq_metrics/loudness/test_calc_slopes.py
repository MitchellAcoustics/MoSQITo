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
        np.testing.assert_array_equal(s1, s[:, i])


@pytest.mark.loudness_zwst
def test_input_untouched_and_silence():
    core = _core(4)
    before = core.copy()
    _calc_slopes(core)
    np.testing.assert_array_equal(core, before)
    n, s = _calc_slopes(np.zeros((21, 3)))
    assert not n.any() and not s.any()


@pytest.mark.loudness_zwst
def test_degenerate_pattern_is_not_negative():
    # A linear integer pattern, which no spectrum can produce, gave negative
    # specific loudness in the previous vectorised implementation.
    n, s = _calc_slopes(np.arange(21.0))
    assert n >= 0 and s.min() >= 0


@pytest.mark.loudness_zwst
def test_total_is_rounded_as_in_iso_532_1():
    n, _ = _calc_slopes(_core(20))
    quantum = np.where(n <= 16, 1e-3, 1e-2)
    np.testing.assert_allclose(n / quantum, np.round(n / quantum), atol=1e-6)
