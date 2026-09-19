#!/usr/bin/env python3
"""Export reference outputs for mosqito-core's n-th octave band analysis.

`center_freq` and `filter_bandwidth` are transcribed directly from
`_center_freq.py` and `_filter_bandwidth.py` (using only numpy) rather than
imported, since importing the `mosqito` package pulls in matplotlib, pyuff,
pandas and openpyxl transitively (`mosqito/__init__.py` -> `utils.isoclose` ->
`matplotlib.pyplot`, `utils.load` -> `pyuff`) — the transcription is verified
against the installed `mosqito` package once below when it's available, so a
slip would be caught rather than silently trusted.

Where `mosqito` *is* importable (this repository, with its full dev
dependencies), this script also exports end-to-end reference vectors from the
real `noct_spectrum`/`noct_synthesis` on MoSQITo's own reference corpus wav —
a strictly stronger conformance gate than the self-consistency check alone.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden_noct.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/mosqito-core/tests/golden_noct.json"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

NOMINAL_OCTAVE_CENTER_FREQUENCIES = np.array(
    [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
)
NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES = np.array(
    [
        25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0,
        400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0,
        4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
    ]
)


def center_freq(fmin, fmax, n=3, g=10, fr=1000):
    """Transcribed from mosqito/sound_level_meter/noct_spectrum/_center_freq.py."""
    b = 1 / n
    if g == 2:
        u = 2**b
    elif g == 10:
        u = 10 ** (3 * b / 10)
    else:
        raise ValueError("g must be 2 or 10")
    kmin, kmax = np.round(np.log10(np.array([fmin, fmax]) / fr) / np.log10(u), 0)
    k = np.arange(kmin, kmax + 1).astype(int)
    f_exact = fr * u**k

    f_nom = f_exact.copy()
    if n == 1:
        freq = NOMINAL_OCTAVE_CENTER_FREQUENCIES
    elif n == 3:
        freq = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES
    if n in (1, 3):
        i_ref = np.where(freq == fr)[0][0]
        ind = np.where((k >= -i_ref) & (k < (len(freq) - i_ref)))
        f_nom = freq[k[ind] + i_ref]

    return f_exact, f_nom


def filter_bandwidth(fc, n=3, order=3):
    """Transcribed from mosqito/sound_level_meter/noct_spectrum/_filter_bandwidth.py."""
    b = 1 / n
    f1 = fc / (2 ** (b / 2))
    f2 = fc * (2 ** (b / 2))
    qr = fc / (f2 - f1)
    qd = (np.pi / 2 / order) / (np.sin(np.pi / 2 / order)) * qr
    alpha = (1 + np.sqrt(1 + 4 * qd**2)) / 2 / qd
    return alpha, f1, f2


def main() -> None:
    cases = {}

    # center_freq: every (fmin, fmax, n, g, fr) combination MoSQITo's Python
    # calls with, plus a below-the-table case exercising the fallback this
    # port adds (see DEVIATIONS.md) — checked here only for the in-range
    # cases, where the port's fallback is provably inert.
    cf_cases = []
    for fmin, fmax, n, g, fr in [
        (24.0, 12600.0, 3, 10, 1000.0),  # loudness_zwst / noct_spectrum default
        (24.0, 12600.0, 1, 10, 1000.0),
        (25.0, 20000.0, 3, 10, 1000.0),  # full third-octave table
        (31.5, 16000.0, 1, 10, 1000.0),  # full octave table
        (500.0, 2000.0, 3, 10, 1000.0),
        (100.0, 10000.0, 3, 10, 1000.0),
    ]:
        f_exact, f_nom = center_freq(fmin, fmax, n, g, fr)
        cf_cases.append({
            "fmin": fmin, "fmax": fmax, "n": n, "g": g, "fr": fr,
            "f_exact": f_exact.tolist(), "f_nom": f_nom.tolist(),
        })
    cases["center_freq"] = cf_cases

    # filter_bandwidth
    fb_cases = []
    for fc, n in [
        ([1000.0], 3),
        (center_freq(24.0, 12600.0, 3, 10, 1000.0)[0].tolist(), 3),
        (center_freq(24.0, 12600.0, 1, 10, 1000.0)[0].tolist(), 1),
    ]:
        alpha, f1, f2 = filter_bandwidth(np.array(fc), n)
        fb_cases.append({
            "fc": fc, "n": n,
            "alpha": alpha.tolist(), "f1": f1.tolist(), "f2": f2.tolist(),
        })
    cases["filter_bandwidth"] = fb_cases

    # End-to-end vectors from the real mosqito package on its own reference
    # corpus, when the full dev environment (matplotlib, pyuff, pandas,
    # openpyxl) is installed. This both verifies the transcriptions above
    # against the actual source and gives mosqito-core a direct MoSQITo
    # conformance gate, not just internal self-consistency.
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from mosqito.sound_level_meter import noct_spectrum, noct_synthesis
        from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq
        from mosqito.sound_level_meter.noct_spectrum._filter_bandwidth import (
            _filter_bandwidth,
        )
        from mosqito.utils import load
        from numpy.fft import fft, fftfreq
    except ImportError as exc:
        print(f"[skip] mosqito not importable ({exc}); end-to-end vectors not written")
    else:
        for case in cf_cases:
            fe, fn = _center_freq(
                case["fmin"], case["fmax"], n=case["n"], G=case["g"], fr=case["fr"]
            )
            assert np.allclose(fe, case["f_exact"]) and np.allclose(fn, case["f_nom"]), (
                "center_freq transcription diverges from mosqito's source"
            )
        for case in fb_cases:
            a, f1, f2 = _filter_bandwidth(np.array(case["fc"]), n=case["n"])
            assert (
                np.allclose(a, case["alpha"])
                and np.allclose(f1, case["f1"])
                and np.allclose(f2, case["f2"])
            ), "filter_bandwidth transcription diverges from mosqito's source"
        print("transcriptions verified against the installed mosqito package")

        # The wav itself is not embedded here: it already lives in the repo
        # at the path recorded below, and the Rust test reads it directly
        # (with its own already golden-tested WAV/FFT primitives) rather than
        # duplicating a 480,000-sample signal into this JSON file. Only the
        # small per-band outputs (<=30 floats each) are captured.
        wav = REPO_ROOT / "tests/input/Test signal 5 (pinknoise 60 dB).wav"
        sig, fs = load(str(wav), wav_calib=2 * 2**0.5)
        n = len(sig)

        e2e = {
            "wav_path": "tests/input/Test signal 5 (pinknoise 60 dB).wav",
            "wav_calib": 2 * 2**0.5,
            "fs": fs,
            "n": n,
        }
        for order in (1, 3):
            spec_t, freq_t = noct_spectrum(sig, fs, fmin=24, fmax=12600, n=order)
            e2e[f"noct_spectrum_n{order}"] = {
                "freq": freq_t.tolist(),
                "spec": spec_t.tolist(),
            }

        spectrum = 2 / np.sqrt(2) / n * fft(sig)[0 : n // 2]
        freqs = fftfreq(n, 1 / fs)[0 : n // 2]
        for order in (1, 3):
            spec_f, freq_f = noct_synthesis(
                np.abs(spectrum), freqs, fmin=24, fmax=12600, n=order
            )
            e2e[f"noct_synthesis_n{order}"] = {
                "freq": freq_f.tolist(),
                "spec": spec_f.tolist(),
            }
        cases["end_to_end"] = e2e

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
