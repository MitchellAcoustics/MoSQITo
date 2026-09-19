#!/usr/bin/env python3
"""Export golden vectors for SII (ANSI S3.5) against real MoSQITo.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden_sii.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from mosqito.sq_metrics.speech_intelligibility.sii_ansi._main_sii import _main_sii
from mosqito.sq_metrics.speech_intelligibility.sii_ansi._band_procedure_data import (
    _get_critical_band_data,
    _get_equal_critical_band_data,
    _get_octave_band_data,
    _get_third_octave_band_data,
)
from mosqito.sq_metrics.speech_intelligibility.sii_ansi._speech_data import (
    _get_critical_band_speech_data,
    _get_third_octave_band_speech_data,
)
from mosqito.sq_metrics.speech_intelligibility.sii_ansi.sii_ansi import sii_ansi
from mosqito.sq_metrics.speech_intelligibility.sii_ansi.sii_ansi_freq import sii_ansi_freq
from mosqito.sq_metrics.speech_intelligibility.sii_ansi.sii_ansi_level import sii_ansi_level

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/mosqito-core/tests/golden_sii.json"

RNG = np.random.default_rng(20250202)


def main() -> None:
    cases: dict[str, object] = {}

    # --- _main_sii, all four band procedures, real (not synthetic) speech
    # spectra with a randomised noise spectrum -------------------------------
    methods = {
        "critical": _get_critical_band_data,
        "equally_critical": _get_equal_critical_band_data,
        "third_octave": _get_third_octave_band_data,
        "octave": _get_octave_band_data,
    }
    main_sii_cases = {}
    for method, data_fn in methods.items():
        data = data_fn()
        center = data[0]
        nbands = len(center)
        if method == "critical":
            speech, _ = _get_critical_band_speech_data("normal")
        elif method == "third_octave":
            speech, _ = _get_third_octave_band_speech_data("raised")
        else:
            # equally_critical / octave: use the band data's own normal
            # speech spectrum (last tuple element).
            speech = data[-1]
        noise = speech - 10.0 + RNG.uniform(-5.0, 5.0, nbands)
        sii, sii_spec, freq_axis = _main_sii(method, speech.copy(), noise.copy(), threshold=None)
        sii_z, sii_spec_z, _ = _main_sii(method, speech.copy(), noise.copy(), threshold="zwicker")
        main_sii_cases[method] = {
            "speech": speech.tolist(),
            "noise": noise.tolist(),
            "sii": float(sii),
            "sii_spec": np.asarray(sii_spec).tolist(),
            "freq_axis": np.asarray(freq_axis).tolist(),
            "sii_zwicker": float(sii_z),
            "sii_spec_zwicker": np.asarray(sii_spec_z).tolist(),
        }
    cases["main_sii"] = main_sii_cases

    # --- sii_ansi_level ------------------------------------------------------
    sii, sii_spec, freq_axis = sii_ansi_level(65.0, method="third_octave", speech_level="raised")
    cases["sii_ansi_level"] = {
        "noise_level": 65.0, "method": "third_octave", "speech_level": "raised",
        "sii": float(sii), "sii_spec": np.asarray(sii_spec).tolist(),
        "freq_axis": np.asarray(freq_axis).tolist(),
    }

    # --- sii_ansi_freq -------------------------------------------------------
    fs = 48000
    nfft = 4800
    freqs = np.arange(1, nfft // 2 + 1) * (fs / nfft)
    spectrum_db = 40.0 - 0.002 * freqs + 5.0 * np.sin(freqs / 500.0)
    sii, sii_spec, freq_axis = sii_ansi_freq(
        spectrum_db, freqs, method="critical", speech_level="normal"
    )
    cases["sii_ansi_freq"] = {
        "spectrum_db": spectrum_db.tolist(), "freqs": freqs.tolist(),
        "method": "critical", "speech_level": "normal",
        "sii": float(sii), "sii_spec": np.asarray(sii_spec).tolist(),
        "freq_axis": np.asarray(freq_axis).tolist(),
    }

    # --- sii_ansi (time signal) -----------------------------------------------
    n = nfft
    t = np.arange(n) / fs
    noise_sig = (
        0.05 * RNG.standard_normal(n)
        + 0.02 * np.sin(2 * np.pi * 500 * t)
        + 0.01 * np.sin(2 * np.pi * 2000 * t)
    )
    sii, sii_spec, freq_axis = sii_ansi(noise_sig, fs, method="octave", speech_level="loud")
    cases["sii_ansi"] = {
        "noise": noise_sig.tolist(), "fs": fs, "method": "octave", "speech_level": "loud",
        "sii": float(sii), "sii_spec": np.asarray(sii_spec).tolist(),
        "freq_axis": np.asarray(freq_axis).tolist(),
    }

    OUT.write_text(json.dumps(cases))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
