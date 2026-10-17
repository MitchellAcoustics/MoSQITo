# -*- coding: utf-8 -*-
"""Memory budget, thread-count independence and edge cases of the ECMA-418-2 metrics."""
import tracemalloc

import numpy as np
import pytest

from mosqito.sq_metrics import loudness_ecma, roughness_ecma
from mosqito.sq_metrics.loudness.loudness_ecma._band_pass_signals import _band_pass_signals
from mosqito.sq_metrics.loudness.loudness_ecma._ecma_time_segmentation import _ecma_time_segmentation
from mosqito.sq_metrics.loudness.loudness_ecma._loudness_from_bandpass import _loudness_from_bandpass
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


@pytest.mark.loudness_ecma
def test_loudness_ecma_workers_do_not_change_results():
    x = _am_tone(FS // 2)
    np.testing.assert_equal(loudness_ecma(x, FS, workers=1), loudness_ecma(x, FS, workers=4))


@pytest.mark.roughness_ecma
def test_roughness_ecma_workers_do_not_change_results():
    x = _am_tone(FS // 2)
    np.testing.assert_equal(roughness_ecma(x, FS, workers=1), roughness_ecma(x, FS, workers=4))


@pytest.mark.loudness_ecma
@pytest.mark.parametrize("n", [500, 2047, 2048, 5000])
def test_loudness_ecma_short_signals(n):
    N, N_time, N_spec, bark, time = loudness_ecma(_am_tone(n), FS)
    assert np.isfinite(N) and len(N_spec) == 53 and len(time) == 53
    assert all(len(row) == len(N_time) for row in N_spec)


@pytest.mark.loudness_ecma
def test_loudness_ecma_accepts_uniform_per_band_sizes():
    x = _am_tone(FS // 4)
    expected = loudness_ecma(x, FS)
    got = loudness_ecma(x, FS, sb=np.full(53, 2048), sh=np.full(53, 1024))
    np.testing.assert_equal(expected, got)
    with pytest.raises(ValueError):
        loudness_ecma(x, FS, sb=[2048] * 52)


@pytest.mark.loudness_ecma
@pytest.mark.parametrize("sb, sh", [(2048, 1024), (1000, 300)])
def test_loudness_ecma_matches_the_whole_array_helpers(sb, sh):
    # the band-by-band driver must agree with the helpers that process all 53 bands at once
    x = _am_tone(FS // 2)
    signal, n_new = _preprocessing(x.copy(), sb, sh)
    blocks, time = _ecma_time_segmentation(_band_pass_signals(signal, sb, sh), sb, sh, n_new)
    n_specific, _ = _loudness_from_bandpass(blocks)
    _, _, N_spec, _, time_array = loudness_ecma(x, FS, sb=sb, sh=sh)
    np.testing.assert_equal(N_spec, n_specific)
    np.testing.assert_equal(time_array, time)
