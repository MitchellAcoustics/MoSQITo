#!/usr/bin/env python3
"""Export SciPy reference outputs for the Rust DSP tests.

The Rust primitives in ``mosqito-core::dsp`` deliberately reproduce SciPy's
semantics, because MoSQITo's published validation results were produced through
SciPy. This script captures those semantics as concrete numbers so the Rust side
can assert against them instead of against a re-reading of the documentation.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden.py

It writes ``crates/mosqito-core/tests/golden.json`` plus the versions used, so a
future mismatch can be attributed to a SciPy change rather than a port bug.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import scipy
import scipy.signal as ss
from scipy.interpolate import pchip_interpolate

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/mosqito-core/tests/golden.json"

# Fixed seed: the vectors are committed, so they must not move between runs.
RNG = np.random.default_rng(20240918)


def chirp(n: int) -> np.ndarray:
    """A broadband test signal that exercises the whole frequency range."""
    t = np.arange(n) / n
    return np.sin(2 * np.pi * (5 + 200 * t) * t) + 0.1 * RNG.standard_normal(n)


def main() -> None:
    cases: dict[str, object] = {}

    cases["versions"] = {"numpy": np.__version__, "scipy": scipy.__version__}

    # --- lfilter -----------------------------------------------------------
    x = chirp(512)
    b, a = [0.5, 0.2, -0.1], [1.0, 0.3, 0.05]
    cases["lfilter"] = {"b": b, "a": a, "x": x.tolist(), "y": ss.lfilter(b, a, x).tolist()}

    # lfilter with complex coefficients: the ECMA-418-2 gammatone case.
    bc = np.array([0.1 + 0.2j, 0.3 - 0.1j, 0.05 + 0.0j])
    ac = np.array([1.0 + 0.0j, -0.4 + 0.3j, 0.1 - 0.05j])
    yc = ss.lfilter(bc, ac, x)
    cases["lfilter_complex"] = {
        "b_re": bc.real.tolist(), "b_im": bc.imag.tolist(),
        "a_re": ac.real.tolist(), "a_im": ac.imag.tolist(),
        "x": x.tolist(),
        "y_re": yc.real.tolist(), "y_im": yc.imag.tolist(),
    }

    # --- filter design -----------------------------------------------------
    designs = []
    for order, wn in [(2, 0.25), (4, 0.1), (8, 0.3), (8, 0.01)]:
        designs.append({
            "kind": "butter_lowpass", "order": order, "wn": wn,
            "sos": ss.butter(order, wn, "lowpass", output="sos").tolist(),
        })
    for order, lo, hi in [(3, 0.2, 0.4), (3, 0.01, 0.02), (2, 0.6, 0.8)]:
        designs.append({
            "kind": "butter_bandpass", "order": order, "low": lo, "high": hi,
            "sos": ss.butter(order, [lo, hi], "bandpass", output="sos").tolist(),
        })
    # The third-octave bands MoSQITo designs at 48 kHz, including the 25 Hz
    # band where the poles crowd the unit circle.
    for fc in [25.0, 1000.0, 12500.0]:
        fs = 48000.0
        lo = fc / 2 ** (1 / 6) / (fs / 2)
        hi = fc * 2 ** (1 / 6) / (fs / 2)
        designs.append({
            "kind": "butter_bandpass", "order": 3, "low": lo, "high": hi,
            "sos": ss.butter(3, [lo, hi], "bandpass", output="sos").tolist(),
        })
    # decimate's anti-alias prototype, for each factor MoSQITo actually uses.
    for q in [2, 4, 8, 32]:
        designs.append({
            "kind": "cheby1_lowpass", "order": 8, "rp": 0.05, "wn": 0.8 / q,
            "sos": ss.cheby1(8, 0.05, 0.8 / q, output="sos").tolist(),
        })
    cases["designs"] = designs

    # --- sosfilt / filtfilt / sosfiltfilt ----------------------------------
    sos = ss.butter(4, 0.2, "lowpass", output="sos")
    cases["sosfilt"] = {"sos": sos.tolist(), "x": x.tolist(), "y": ss.sosfilt(sos, x).tolist()}
    cases["filtfilt"] = {
        "b": [0.2, 0.2], "a": [1.0, -0.6], "x": x.tolist(),
        "y": ss.filtfilt([0.2, 0.2], [1.0, -0.6], x).tolist(),
    }
    cases["sosfiltfilt"] = {
        "sos": sos.tolist(), "x": x.tolist(), "y": ss.sosfiltfilt(sos, x).tolist(),
    }

    # --- decimate ----------------------------------------------------------
    long_x = chirp(4096)
    cases["decimate"] = [
        {"q": q, "x": long_x.tolist(), "y": ss.decimate(long_x, q).tolist()}
        for q in (2, 4, 8)
    ]

    # --- hilbert envelope --------------------------------------------------
    cases["hilbert"] = [
        {"x": sig.tolist(), "env": np.abs(ss.hilbert(sig)).tolist()}
        for sig in (chirp(512), chirp(511))  # even and odd lengths
    ]

    # --- resample ----------------------------------------------------------
    cases["resample"] = []
    for n, num in [(256, 512), (512, 256), (441, 480), (300, 450), (300, 151)]:
        sig = chirp(n)
        cases["resample"].append(
            {"x": sig.tolist(), "num": num, "y": ss.resample(sig, num).tolist()}
        )

    # --- sosfreqz ----------------------------------------------------------
    w, h = ss.sosfreqz(sos, worN=64)
    cases["sosfreqz"] = {
        "sos": sos.tolist(), "n": 64,
        "h_re": h.real.tolist(), "h_im": h.imag.tolist(),
    }

    # --- find_peaks prominence --------------------------------------------
    cases["find_peaks"] = []
    for sig in (
        np.array([0.0, 1.0, 0.0, 2.0, 0.0, 10.0, 2.0, 6.0, 1.0, 10.0, 0.0]),
        np.abs(chirp(256)),
        np.array([0.0, 1.0, 1.0, 1.0, 0.0, 3.0, 3.0, 0.0]),
    ):
        idx, props = ss.find_peaks(sig, prominence=[None, None])
        cases["find_peaks"].append({
            "x": sig.tolist(),
            "indices": idx.tolist(),
            "prominences": props["prominences"].tolist(),
        })

    # --- pchip -------------------------------------------------------------
    cases["pchip"] = []
    for xs, ys in [
        ([0.0, 1.0, 2.0, 3.0, 4.5], [1.0, 3.0, 2.0, 5.0, 4.0]),
        ([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0, 1.0]),
        ([0.0, 2.0], [1.0, 5.0]),
    ]:
        xq = np.linspace(xs[0], xs[-1], 37)
        cases["pchip"].append({
            "x": xs, "y": ys, "xq": xq.tolist(),
            "yq": pchip_interpolate(xs, ys, xq).tolist(),
        })

    # --- interp / percentile / median --------------------------------------
    xp = [0.0, 1.0, 2.0, 4.0, 8.0]
    fp = [0.0, 10.0, 20.0, 40.0, 15.0]
    xq = np.linspace(-1.0, 9.0, 41)
    cases["interp"] = {
        "xp": xp, "fp": fp, "xq": xq.tolist(),
        "yq": np.interp(xq, xp, fp).tolist(),
    }

    sample = chirp(97)
    cases["percentile"] = {
        "x": sample.tolist(),
        "q": [0.0, 5.0, 25.0, 50.0, 90.0, 99.0, 100.0],
        "values": [float(np.percentile(sample, q))
                   for q in (0.0, 5.0, 25.0, 50.0, 90.0, 99.0, 100.0)],
        "median": float(np.median(sample)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases))
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.0f} KiB)")
    print(f"numpy {np.__version__}, scipy {scipy.__version__}")


if __name__ == "__main__":
    main()
