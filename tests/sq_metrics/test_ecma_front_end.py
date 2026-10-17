# -*- coding: utf-8 -*-
"""Memory budget, thread-count independence and edge cases of the ECMA-418-2 metrics."""
import tracemalloc

import numpy as np
import pytest

from mosqito.sq_metrics import loudness_ecma, roughness_ecma
from mosqito.sq_metrics.loudness.loudness_ecma._band_blocks import (
    band_blocks,
    band_specific_loudness,
    ear_filtered,
    ecma_block_index,
    ecma_time_axis,
)
from mosqito.sq_metrics.loudness.loudness_ecma._band_pass_signals import _band_pass_signals
from mosqito.sq_metrics.loudness.loudness_ecma._ecma_time_segmentation import (
    _ecma_time_segmentation,
)
from mosqito.sq_metrics.loudness.loudness_ecma._loudness_from_bandpass import (
    _loudness_from_bandpass,
)
from mosqito.sq_metrics.loudness.loudness_ecma._preprocessing import _preprocessing

FS = 48000


def _am_tone(n):
    t = np.arange(n) / FS
    return 0.05 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)


def _peak_mb(fn, *args, **kwargs):
    tracemalloc.start()
    try:
        fn(*args, **kwargs)
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


@pytest.mark.loudness_ecma
def test_loudness_ecma_memory_budget():
    # the previous implementation peaked at about 720 MB for 5 s of signal
    assert _peak_mb(loudness_ecma, _am_tone(5 * FS), FS) < 100


@pytest.mark.roughness_ecma
def test_roughness_ecma_memory_budget():
    # the previous implementation peaked at about 590 MB for 1 s of signal
    assert _peak_mb(roughness_ecma, _am_tone(FS), FS) < 100


def _same(a, b):
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.loudness_ecma
def test_loudness_ecma_workers_do_not_change_results():
    x = _am_tone(FS // 2)
    _same(loudness_ecma(x, FS, workers=1), loudness_ecma(x, FS, workers=4))


@pytest.mark.roughness_ecma
def test_roughness_ecma_workers_do_not_change_results():
    x = _am_tone(FS // 2)
    _same(roughness_ecma(x, FS, workers=1), roughness_ecma(x, FS, workers=4))


@pytest.mark.loudness_ecma
@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_invalid_workers(bad):
    with pytest.raises(ValueError):
        loudness_ecma(_am_tone(4000), FS, workers=bad)
    with pytest.raises(ValueError):
        roughness_ecma(_am_tone(4000), FS, workers=bad)


@pytest.mark.loudness_ecma
@pytest.mark.parametrize("n", [500, 2047, 2048, 5000])
def test_loudness_ecma_short_signals(n):
    N, N_time, N_spec, bark, time = loudness_ecma(_am_tone(n), FS)
    assert np.isfinite(N) and len(N_spec) == 53 and len(time) == 53
    assert all(len(row) == len(N_time) for row in N_spec)


@pytest.mark.loudness_ecma
@pytest.mark.parametrize("n", [500, 2047, 5000])
@pytest.mark.parametrize("sb, sh", [(2048, 1024), (1000, 300)])
@pytest.mark.parametrize("band", [0, 26, 52])
def test_band_helpers_match_the_former_helpers(n, sb, sh, band):
    x = _am_tone(n)
    padded, n_new = _preprocessing(x, sb, sh)

    old_bandpass = _band_pass_signals(padded, sb, sh)
    old_blocks, old_time = _ecma_time_segmentation(old_bandpass, sb, sh, n_new)
    old_loudness, _ = _loudness_from_bandpass(np.asarray(old_blocks))

    index = ecma_block_index(n_new, sb, sh)
    blocks = band_blocks(ear_filtered(padded), band, index)
    # same samples in the same blocks, and the same time axis
    np.testing.assert_array_equal(blocks, old_blocks[band])
    np.testing.assert_array_equal(ecma_time_axis(len(padded), index), old_time[band])
    np.testing.assert_allclose(
        band_specific_loudness(blocks, band), old_loudness[band], rtol=1e-12
    )


@pytest.mark.loudness_ecma
def test_loudness_ecma_accepts_uniform_per_band_sizes():
    x = _am_tone(FS // 4)
    expected = loudness_ecma(x, FS)
    got = loudness_ecma(x, FS, sb=np.full(53, 2048), sh=np.full(53, 1024))
    _same(expected, got)
    with pytest.raises(ValueError):
        loudness_ecma(x, FS, sb=[2048] * 52)
