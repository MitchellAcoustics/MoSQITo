# -*- coding: utf-8 -*-
import concurrent.futures

import pytest

from mosqito.utils import _parallel
from mosqito.utils._parallel import parallel_map, resolve_workers


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MOSQITO_WORKERS", raising=False)


def test_default_is_one_worker():
    assert resolve_workers(None) == 1


def test_argument_beats_environment(monkeypatch):
    monkeypatch.setenv("MOSQITO_WORKERS", "3")
    assert resolve_workers(2) == 2
    assert resolve_workers(None) == 3


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True])
def test_invalid_argument(bad):
    with pytest.raises(ValueError):
        resolve_workers(bad)


@pytest.mark.parametrize("bad", ["0", "-2", "many", "1.5"])
def test_invalid_environment(monkeypatch, bad):
    monkeypatch.setenv("MOSQITO_WORKERS", bad)
    with pytest.raises(ValueError, match="MOSQITO_WORKERS"):
        resolve_workers(None)


def test_empty_environment_means_default(monkeypatch):
    monkeypatch.setenv("MOSQITO_WORKERS", "")
    assert resolve_workers(None) == 1


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
