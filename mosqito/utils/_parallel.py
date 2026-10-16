# -*- coding: utf-8 -*-
"""Opt-in thread parallelism for independent, GIL-releasing work items.

The metrics that use this (the 53 ECMA bands, the n-octave filter bands, the
200 ms roughness segments) run SciPy/NumPy filters and FFTs that release the
GIL, so plain threads scale. Nothing runs in parallel unless asked: the default
is one worker.
"""
import os
from concurrent.futures import ThreadPoolExecutor

ENV_VAR = "MOSQITO_WORKERS"


def _as_worker_count(value, origin):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be an integer >= 1, got %r" % (origin, value))
    return value


def resolve_workers(workers=None):
    """Number of workers requested: the argument, else ``MOSQITO_WORKERS``, else 1."""
    if workers is not None:
        return _as_worker_count(workers, "workers")
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return 1
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("%s must be an integer >= 1, got %r" % (ENV_VAR, raw))
    return _as_worker_count(value, ENV_VAR)


def parallel_map(fn, items, workers=None):
    """``[fn(x) for x in items]``, optionally on a thread pool, order preserved.

    The pool never has more threads than there are items. With one worker (the
    default) no pool is created.
    """
    items = list(items)
    n_threads = min(resolve_workers(workers), len(items))
    if n_threads <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(n_threads) as pool:
        return list(pool.map(fn, items))
