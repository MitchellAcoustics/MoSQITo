# -*- coding: utf-8 -*-
"""Opt-in thread parallelism for independent, GIL-releasing work items.

The metrics that use this (the 53 ECMA bands, the n-octave filter bands, the
200 ms roughness segments) run SciPy/NumPy filters and FFTs that release the
GIL, so plain threads scale. Nothing runs in parallel unless asked: the default
is one worker.
"""
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral


def resolve_workers(workers=None):
    """Number of workers requested: the ``workers`` argument, or 1 if None."""
    if workers is None:
        return 1
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("workers must be an integer >= 1, got %r" % (workers,))
    return int(workers)


def parallel_map(fn, items, workers=None):
    """``[fn(x) for x in items]``, optionally on a thread pool, order preserved.

    The pool never has more threads than there are items. With one worker
    (``workers=None`` or 1) no pool is created.
    """
    items = list(items)
    n_threads = min(resolve_workers(workers), len(items))
    if n_threads <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(n_threads) as pool:
        return list(pool.map(fn, items))
