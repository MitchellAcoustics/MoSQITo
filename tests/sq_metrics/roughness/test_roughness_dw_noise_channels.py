# -*- coding: utf-8 -*-
"""Roughness (Daniel and Weber) must not depend on floating point rounding noise.

For sparse spectra some channels carry no real modulation, only rounding noise
with a modulation depth of order 1e-17 to 1e-16. Their correlation with
neighbouring channels is meaningless and must not reach the roughness.
"""
import importlib

import numpy as np
import pytest

from mosqito.sq_metrics import roughness_dw

# the package re-exports a function under the module's name: import the module itself
main_calc_module = importlib.import_module(
    "mosqito.sq_metrics.roughness.roughness_dw._roughness_dw_main_calc"
)

FS = 48000


def _two_tones(amplitude=0.03):
    """Steady 500 + 600 Hz for 1 s: most channels carry no modulation."""
    t = np.arange(FS) / FS
    return amplitude * (np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 600 * t))


def _noisy(real_fft, rng):
    """Wrap an FFT routine so its output carries 1e-16 relative rounding noise."""

    def wrapper(x, *args, **kwargs):
        y = real_fft(x, *args, **kwargs)
        scale = 1e-16 * np.max(np.abs(y))
        return y + scale * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))

    return wrapper


@pytest.mark.roughness_dw
@pytest.mark.xfail(strict=True, reason="fixed in the next commit")
@pytest.mark.parametrize("seed", range(3))
def test_roughness_ignores_rounding_noise_channels(monkeypatch, seed):
    reference = roughness_dw(_two_tones(), FS)[0]
    rng = np.random.default_rng(seed)
    monkeypatch.setattr(main_calc_module, "fft", _noisy(main_calc_module.fft, rng))
    monkeypatch.setattr(main_calc_module, "ifft", _noisy(main_calc_module.ifft, rng))
    np.testing.assert_allclose(roughness_dw(_two_tones(), FS)[0], reference, rtol=1e-9)


@pytest.mark.roughness_dw
@pytest.mark.xfail(strict=True, reason="fixed in the next commit")
@pytest.mark.parametrize("amplitude", [0.003, 0.03, 0.3])
def test_roughness_dw_two_tones_is_the_same_in_every_block(amplitude):
    # every 200 ms block holds the same periodic content, so the roughness must
    # be identical in all blocks, whatever the rounding noise in empty channels
    R = roughness_dw(_two_tones(amplitude), FS)[0]
    assert R.mean() > 0
    assert np.ptp(R) / R.mean() < 1e-9


@pytest.mark.roughness_dw
@pytest.mark.xfail(strict=True, reason="fixed in the next commit")
def test_roughness_dw_two_tones_value():
    # 0.14502 before the fix; a floor that dropped real channels would move it too
    assert roughness_dw(_two_tones(), FS)[0][0] == pytest.approx(0.1449049, rel=1e-6)
