# -*- coding: utf-8 -*-
"""Memory budget, thread-count independence and edge cases of the ECMA-418-2 metrics."""
import tracemalloc

import numpy as np
import pytest

from mosqito.sq_metrics import loudness_ecma, roughness_ecma

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
def test_workers_taken_from_environment(monkeypatch):
    x = _am_tone(FS // 4)
    reference = loudness_ecma(x, FS)
    monkeypatch.setenv("MOSQITO_WORKERS", "3")
    _same(reference, loudness_ecma(x, FS))


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
