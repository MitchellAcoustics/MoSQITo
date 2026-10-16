# -*- coding: utf-8 -*-
"""Roughness (Daniel and Weber) must not depend on floating point rounding noise.

For sparse spectra some channels carry no real modulation, only rounding noise of
about 1e-17. Their correlation with neighbouring channels is meaningless and must
not reach the roughness.
"""
import importlib

import numpy as np
import pytest

from mosqito.sound_level_meter.comp_spectrum import comp_spectrum
from mosqito.utils.time_segmentation import time_segmentation

main_calc_module = importlib.import_module(
    "mosqito.sq_metrics.roughness.roughness_dw._roughness_dw_main_calc"
)
from mosqito.sq_metrics.roughness.roughness_dw._gzi_weighting import _gzi_weighting  # noqa: E402
from mosqito.sq_metrics.roughness.roughness_dw._H_weighting import _H_weighting  # noqa: E402

FS = 48000
NPERSEG = int(0.2 * FS)


def _two_tone_inputs():
    t = np.arange(FS) / FS
    x = 0.03 * np.sin(2 * np.pi * 500 * t) + 0.03 * np.sin(2 * np.pi * 600 * t)
    blocks, _ = time_segmentation(x, FS, nperseg=NPERSEG, noverlap=NPERSEG // 2)
    spec, _ = comp_spectrum(blocks, FS, nfft="default", window="blackman", db=False)
    freq_axis = np.arange(1, NPERSEG // 2 + 1) * (FS / NPERSEG)
    return spec[:, 0].copy(), freq_axis, _gzi_weighting(np.arange(1, 48) / 2), _H_weighting(NPERSEG, FS)


def _roughness(inputs):
    spec, freq_axis, gzi, h_weight = inputs
    return main_calc_module._roughness_dw_main_calc(spec.copy(), freq_axis, FS, gzi, h_weight)[0]


def _noisy(real_fft, rng):
    """Wrap an FFT routine so its output carries 1e-16 relative rounding noise."""

    def wrapper(x, *args, **kwargs):
        y = real_fft(x, *args, **kwargs)
        scale = 1e-16 * np.max(np.abs(y))
        return y + scale * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))

    return wrapper


@pytest.mark.roughness_dw
@pytest.mark.parametrize("seed", range(10))
def test_roughness_ignores_rounding_noise_channels(monkeypatch, seed):
    inputs = _two_tone_inputs()
    reference = _roughness(inputs)
    rng = np.random.default_rng(seed)
    monkeypatch.setattr(main_calc_module, "fft", _noisy(main_calc_module.fft, rng))
    monkeypatch.setattr(main_calc_module, "ifft", _noisy(main_calc_module.ifft, rng))
    np.testing.assert_allclose(_roughness(inputs), reference, rtol=1e-9)


@pytest.mark.roughness_dw
def test_channel_correlations_skip_inactive_channels():
    rng = np.random.default_rng(0)
    n = 4000
    base = rng.standard_normal(n)
    hBP = np.zeros((47, n))
    mod_depth = np.zeros(47)
    # channels 5 and 7: real, correlated modulation
    hBP[5], hBP[7] = base, base + 0.1 * rng.standard_normal(n)
    mod_depth[[5, 7]] = 0.4
    # channels 9 and 11: rounding noise only next to a real channel, and an exactly empty one
    hBP[9] = 1e-17 * rng.standard_normal(n)
    mod_depth[9] = 1e-16
    hBP[11] = 0.0
    hBP[3] = base
    mod_depth[3] = 0.4
    ki = main_calc_module._channel_correlations(hBP, mod_depth)
    # ki[i] correlates channel i with channel i + 2
    assert ki[3] > 0.99                     # channels 3 and 5: identical modulation
    assert ki[5] > 0.9                      # channels 5 and 7: real, correlated
    assert ki[7] == 0.0                     # channel 9 is rounding noise
    assert ki[9] == 0.0                     # noise channel 9 and empty channel 11
    assert np.count_nonzero(ki) == 2        # nothing else is correlated
