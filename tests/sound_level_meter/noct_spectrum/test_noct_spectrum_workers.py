# -*- coding: utf-8 -*-
import numpy as np
import pytest

from mosqito.sound_level_meter import noct_spectrum
from mosqito.sound_level_meter.noct_spectrum._bandpass_design import _design_cached, _design_sos


def _signal(n=24000):
    return 0.02 * np.random.default_rng(0).standard_normal(n)


@pytest.mark.noct_spectrum
def test_workers_do_not_change_results():
    x = _signal()
    spec1, f1 = noct_spectrum(x, 48000, 24, 12500, workers=1)
    spec4, f4 = noct_spectrum(x, 48000, 24, 12500, workers=4)
    np.testing.assert_array_equal(spec1, spec4)
    np.testing.assert_array_equal(f1, f4)


@pytest.mark.noct_spectrum
def test_2d_input_with_workers():
    x = np.stack([_signal(12000), 0.5 * _signal(12000)], axis=1)
    a = noct_spectrum(x, 48000, 24, 12500, workers=1)[0]
    b = noct_spectrum(x, 48000, 24, 12500, workers=3)[0]
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)


@pytest.mark.noct_spectrum
@pytest.mark.parametrize("bad", [0, -2, 1.5])
def test_invalid_workers(bad):
    with pytest.raises(ValueError):
        noct_spectrum(_signal(4000), 48000, 24, 12500, workers=bad)


@pytest.mark.noct_spectrum
def test_filter_design_is_cached_and_protected():
    _design_cached.cache_clear()
    first = _design_sos(3, 0.1, 0.2)
    first[:] = 0.0  # a caller altering its copy must not alter the cached design
    second = _design_sos(3, 0.1, 0.2)
    assert _design_cached.cache_info().hits == 1
    assert np.any(second != 0.0)
    # calling the metric twice reuses the designs of the first call
    x = _signal(8000)
    noct_spectrum(x, 48000, 24, 12500)
    misses = _design_cached.cache_info().misses
    noct_spectrum(x, 48000, 24, 12500)
    assert _design_cached.cache_info().misses == misses
