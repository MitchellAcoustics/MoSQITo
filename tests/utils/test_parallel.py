# -*- coding: utf-8 -*-
import concurrent.futures

import numpy as np
import pytest

from mosqito.utils import _parallel
from mosqito.sound_level_meter import noct_spectrum
from mosqito.sq_metrics import loudness_ecma, roughness_dw, roughness_ecma
from mosqito.utils._parallel import parallel_map, resolve_workers


def test_default_is_one_worker():
    assert resolve_workers(None) == 1


def test_any_integral_is_accepted():
    assert resolve_workers(2) == 2
    assert resolve_workers(np.int64(3)) == 3


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True])
def test_invalid_argument(bad):
    with pytest.raises(ValueError):
        resolve_workers(bad)


def test_parallel_map_keeps_order():
    assert parallel_map(lambda x: x * x, range(20), workers=4) == [x * x for x in range(20)]


def test_parallel_map_single_worker_uses_no_pool(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a pool must not be created for one worker")

    monkeypatch.setattr(_parallel, "ThreadPoolExecutor", boom)
    assert parallel_map(abs, [-1, -2], workers=1) == [1, 2]


def test_pool_is_capped_at_number_of_tasks(monkeypatch):
    sizes = []
    real = concurrent.futures.ThreadPoolExecutor

    def spy(max_workers=None, *a, **k):
        sizes.append(max_workers)
        return real(max_workers, *a, **k)

    monkeypatch.setattr(_parallel, "ThreadPoolExecutor", spy)
    parallel_map(abs, [-1, -2, -3], workers=16)
    assert sizes == [3]


def test_empty_input():
    assert parallel_map(abs, [], workers=4) == []


def test_exceptions_propagate():
    def fail(x):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        parallel_map(fail, [1, 2], workers=2)


def _noise(n):
    return 0.02 * np.random.default_rng(0).standard_normal(n)


def _am_tone(n):
    t = np.arange(n) / 48000
    return 0.05 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)


@pytest.mark.parametrize(
    "metric, args",
    [
        (noct_spectrum, (_noise(8000), 48000, 24, 12500)),
        (loudness_ecma, (_noise(8000), 48000)),
        (roughness_ecma, (_am_tone(24000), 48000)),
        (roughness_dw, (_noise(24000), 48000)),
    ],
    ids=["noct_spectrum", "loudness_ecma", "roughness_ecma", "roughness_dw"],
)
def test_metrics_pass_workers_to_the_pool(monkeypatch, metric, args):
    sizes = []
    real = concurrent.futures.ThreadPoolExecutor

    def spy(max_workers=None, *a, **k):
        sizes.append(max_workers)
        return real(max_workers, *a, **k)

    monkeypatch.setattr(_parallel, "ThreadPoolExecutor", spy)
    metric(*args)
    assert sizes == []
    metric(*args, workers=3)
    assert sizes == [3]
