# -*- coding: utf-8 -*-
"""Wall time and peak memory of the main metrics on fixed, seeded signals.

Usage, from the repository root::

    python benchmarks/bench_metrics.py [--duration 1 5] [--repeat 3] [--workers 1 4]

Prints a markdown table. Run it on two commits (``PYTHONPATH=<checkout>``) to
compare them. Memory is measured in a separate pass (tracemalloc slows
numpy-call-heavy code a lot, so times are never taken while it is running).
``--workers`` is passed to the metrics that accept it and ignored by the others.
"""
import argparse
import inspect
import os
import sys
import time
import tracemalloc
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.simplefilter("ignore")

import mosqito  # noqa: E402
from mosqito.sound_level_meter import noct_spectrum  # noqa: E402
from mosqito.sq_metrics import (  # noqa: E402
    loudness_ecma, loudness_zwst, loudness_zwst_perseg, loudness_zwtv, roughness_dw,
    roughness_ecma, sharpness_din_st, sharpness_din_tv,
)

FS = 48000


def signals(duration):
    n = int(FS * duration)
    t = np.arange(n) / FS
    tone = 0.05 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)
    noise = 0.02 * np.random.default_rng(0).standard_normal(n)
    return tone, noise


def cases(duration):
    tone, noise = signals(duration)
    return [
        ("loudness_zwst", loudness_zwst, noise, ()),
        ("loudness_zwst_perseg", loudness_zwst_perseg, noise, ()),
        ("loudness_zwtv", loudness_zwtv, noise, ()),
        ("sharpness_din_st", sharpness_din_st, noise, ()),
        ("sharpness_din_tv", sharpness_din_tv, noise, ()),
        ("loudness_ecma", loudness_ecma, tone, ()),
        ("roughness_ecma", roughness_ecma, tone, ()),
        ("roughness_dw (tone)", roughness_dw, tone, ()),
        ("roughness_dw (noise)", roughness_dw, noise, ()),
        ("noct_spectrum", noct_spectrum, noise, (24, 12500)),
    ]


def best_time(fn, args, kwargs, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times)


def peak_mb(fn, args, kwargs):
    tracemalloc.start()
    try:
        fn(*args, **kwargs)
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--duration", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--workers", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    print("mosqito from %s; %d CPUs\n" % (os.path.dirname(mosqito.__file__), os.cpu_count()))
    print("| metric | duration (s) | workers | best time (s) | peak memory (MB) |")
    print("|---|---:|---:|---:|---:|")
    for duration in args.duration:
        for name, fn, x, extra in cases(duration):
            accepts_workers = "workers" in inspect.signature(fn).parameters
            for w in args.workers if accepts_workers else [None]:
                kwargs = {"workers": w} if w is not None else {}
                call = (x.copy(), FS) + extra
                fn(*call, **kwargs)  # warm-up
                t = best_time(fn, call, kwargs, args.repeat)
                m = peak_mb(fn, call, kwargs)
                print("| %s | %g | %s | %.3f | %.0f |" % (name, duration, w if w is not None else "-", t, m))


if __name__ == "__main__":
    main()
