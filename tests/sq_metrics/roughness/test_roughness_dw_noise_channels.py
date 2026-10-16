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
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.xfail(strict=True, reason="channels holding only rounding noise are correlated (fixed in the next commit)")
def test_roughness_ignores_rounding_noise_channels(monkeypatch, seed):
    inputs = _two_tone_inputs()
    reference = _roughness(inputs)
    rng = np.random.default_rng(seed)
    monkeypatch.setattr(main_calc_module, "fft", _noisy(main_calc_module.fft, rng))
    monkeypatch.setattr(main_calc_module, "ifft", _noisy(main_calc_module.ifft, rng))
    np.testing.assert_allclose(_roughness(inputs), reference, rtol=1e-9)
