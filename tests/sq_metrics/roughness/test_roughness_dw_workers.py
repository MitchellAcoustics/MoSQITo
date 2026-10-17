# -*- coding: utf-8 -*-
import numpy as np
import pytest

from mosqito.sq_metrics import roughness_dw

FS = 48000


def _signal(n=FS):
    t = np.arange(n) / FS
    rng = np.random.default_rng(0)
    return 0.05 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t) + 0.005 * rng.standard_normal(n)


@pytest.mark.roughness_dw
@pytest.mark.parametrize("n", [FS // 5, FS, 3 * FS // 2])
def test_workers_do_not_change_results(n):
    x = _signal(n)
    one = roughness_dw(x, FS, workers=1)
    four = roughness_dw(x, FS, workers=4)
    for a, b in zip(one, four):
        np.testing.assert_array_equal(a, b)


@pytest.mark.roughness_dw
@pytest.mark.parametrize("bad", [0, -3, 2.5])
def test_invalid_workers(bad):
    with pytest.raises(ValueError):
        roughness_dw(_signal(FS // 5), FS, workers=bad)


@pytest.mark.roughness_dw
def test_output_shapes():
    R, R_spec, bark, time = roughness_dw(_signal(), FS)
    assert R.shape == time.shape and R_spec.shape == (47, len(R)) and bark.shape == (47,)


@pytest.mark.roughness_dw
def test_component_above_the_last_channel_raises():
    # as before the channels were batched: an audible component above about
    # 24 bark (16 kHz) has no excitation channel and raises IndexError
    t = np.arange(FS // 2) / FS
    x = 2.0 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 16000 * t)
    with pytest.raises(IndexError):
        roughness_dw(x, FS)
