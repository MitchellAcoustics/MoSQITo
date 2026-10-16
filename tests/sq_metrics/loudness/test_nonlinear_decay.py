# -*- coding: utf-8 -*-
import numpy as np
import pytest

from mosqito.sq_metrics.loudness.loudness_zwtv._nonlinear_decay import _nl_loudness


@pytest.mark.loudness_zwtv
@pytest.mark.parametrize("n_frames", [1, 2, 3, 50])
def test_shape_and_input_untouched(n_frames):
    core = np.abs(np.random.default_rng(0).standard_normal((21, n_frames)))
    before = core.copy()
    out = _nl_loudness(core)
    assert out.shape == core.shape
    np.testing.assert_array_equal(core, before)


@pytest.mark.loudness_zwtv
def test_silence_stays_silent():
    assert not np.any(_nl_loudness(np.zeros((21, 10))))


@pytest.mark.loudness_zwtv
def test_decay_is_slower_than_the_input_fall():
    core = np.zeros((21, 40))
    core[:, 5:10] = 6.0
    out = _nl_loudness(core)
    # after the input drops to zero the output decays gradually, never below zero
    assert np.all(out >= 0)
    assert np.all(out[:, 10] > 0)
    assert np.all(np.diff(out[:, 10:], axis=1) <= 1e-12)
