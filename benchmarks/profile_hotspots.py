"""Where does MoSQITo spend its time? Stage-level wall-clock and peak-memory
profile of every public metric on synthetic signals.

    PYTHONPATH=. python benchmarks/profile_hotspots.py [duration_s ...] [--memory]

Prints, per metric and signal duration: total wall time, and the time spent in
each private helper of the heavy pipelines (loudness_zwtv, loudness_ecma,
roughness_ecma). These numbers are the basis of RUST_ACCELERATION_PLAN.md.
"""
import sys
import time
import tracemalloc
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import mosqito  # noqa: E402,F401
from mosqito.sq_metrics import (  # noqa: E402
    loudness_ecma, loudness_zwst, loudness_zwst_perseg, loudness_zwtv, pr_ecma_st,
    roughness_dw, roughness_ecma, sharpness_din_st, sharpness_din_tv, sii_ansi, tnr_ecma_st,
)
from mosqito.sound_level_meter import noct_spectrum  # noqa: E402

FS = 48000
RE = sys.modules["mosqito.sq_metrics.roughness.roughness_ecma.roughness_ecma"]
ZT = sys.modules["mosqito.sq_metrics.loudness.loudness_zwtv.loudness_zwtv"]
LE = sys.modules["mosqito.sq_metrics.loudness.loudness_ecma.loudness_ecma"]
STAGE_TIMES = {}


def _instrument(module, names):
    for name in names:
        if not hasattr(module, name):
            continue
        fn = getattr(module, name)

        def wrapped(*a, _fn=fn, _name=name, **k):
            t0 = time.perf_counter()
            out = _fn(*a, **k)
            STAGE_TIMES[_name] = STAGE_TIMES.get(_name, 0.0) + time.perf_counter() - t0
            return out

        setattr(module, name, wrapped)


_instrument(RE, ["_preprocessing", "_band_pass_signals", "_ecma_time_segmentation",
                 "_loudness_from_bandpass", "hilbert", "decimate", "fft", "_noise_reduction",
                 "_peak_picking", "_estimate_fund_mod_rate", "_interpolation_50",
                 "_non_linear_transform", "_lowpass_filter"])
_instrument(ZT, ["_third_octave_levels", "_main_loudness", "_nl_loudness", "_calc_slopes",
                 "_temporal_weighting"])
_instrument(LE, ["_preprocessing", "_band_pass_signals", "_ecma_time_segmentation",
                 "_nonlinearity"])


def run(duration, track_memory=False):
    n = int(FS * duration)
    t = np.arange(n) / FS
    tone = 0.05 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)
    noise = 0.02 * np.random.default_rng(0).standard_normal(n)
    cases = [
        ("loudness_zwst", loudness_zwst, (noise, FS)),
        ("loudness_zwst_perseg", loudness_zwst_perseg, (noise, FS)),
        ("loudness_zwtv", loudness_zwtv, (noise, FS)),
        ("loudness_ecma", loudness_ecma, (tone, FS)),
        ("sharpness_din_st", sharpness_din_st, (noise, FS)),
        ("sharpness_din_tv", sharpness_din_tv, (noise, FS)),
        ("roughness_dw", roughness_dw, (tone, FS)),
        ("roughness_ecma", roughness_ecma, (tone, FS)),
        ("tnr_ecma_st", tnr_ecma_st, (tone, FS)),
        ("pr_ecma_st", pr_ecma_st, (tone, FS)),
        ("sii_ansi", sii_ansi, (noise, FS, "critical", "normal")),
        ("noct_spectrum", noct_spectrum, (noise, FS, 24, 12500)),
    ]
    print(f"\n########## signal duration {duration} s ##########")
    for name, fn, args in cases:
        STAGE_TIMES.clear()
        args = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args)
        if track_memory:
            tracemalloc.start()
        t0 = time.perf_counter()
        fn(*args)
        wall = time.perf_counter() - t0
        mem = ""
        if track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem = f"  peak alloc {peak / 1e6:.0f} MB"
        print(f"{name:22s} {wall:8.3f} s{mem}")
        for k, v in sorted(STAGE_TIMES.items(), key=lambda kv: -kv[1]):
            if v > 0.002:
                print(f"    {k:28s} {v:8.3f} s  ({100 * v / wall:4.1f}%)")


if __name__ == "__main__":
    track_memory = "--memory" in sys.argv
    durations = [float(d) for d in sys.argv[1:] if not d.startswith("--")] or [1.0, 5.0]
    # tracemalloc distorts the numpy-call-heavy loops badly (30x on _nl_loudness), so
    # only trust the wall times of a run without --memory.
    for d in durations:
        run(d, track_memory=track_memory)
