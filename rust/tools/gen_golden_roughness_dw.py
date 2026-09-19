#!/usr/bin/env python3
"""Export golden vectors for roughness_dw (Daniel & Weber) against real MoSQITo.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden_roughness_dw.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from mosqito.sq_metrics.roughness.roughness_dw._H_weighting import _H_weighting
from mosqito.sq_metrics.roughness.roughness_dw._gzi_weighting import _gzi_weighting
from mosqito.sq_metrics.roughness.roughness_dw._ear_filter_coeff import _ear_filter_coeff
from mosqito.sq_metrics.roughness.roughness_dw._roughness_dw_main_calc import (
    _roughness_dw_main_calc,
)
from mosqito.sq_metrics.roughness.roughness_dw.roughness_dw import roughness_dw
from mosqito.sq_metrics.roughness.roughness_dw.roughness_dw_freq import roughness_dw_freq
from mosqito.sound_level_meter.comp_spectrum import comp_spectrum
from mosqito.utils.am_sine_generator import am_sine_generator

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/mosqito-core/tests/golden_roughness_dw.json"


def am_tone(fs, fc, fmod, dB, duration):
    time = np.arange(0, duration, 1 / fs)
    stimulus = 0.5 * (1 + np.sin(2 * np.pi * fmod * time)) * np.sin(2 * np.pi * fc * time)
    rms = np.sqrt(np.mean(np.power(stimulus, 2)))
    ampl = 0.00002 * np.power(10, dB / 20) / rms
    return stimulus * ampl


def main() -> None:
    cases: dict[str, object] = {}

    # --- _H_weighting, _gzi_weighting, _ear_filter_coeff -------------------
    # A much smaller `n` than the real 200 ms/48 kHz block (9600 samples)
    # suffices to validate `_H_weighting`'s interpolation/broadcast logic —
    # keeps the golden file a reasonable size without weakening the check
    # (the real pipeline's own n=9600 case is exercised end to end by the
    # `main_calc`/`roughness_dw` cases below instead).
    fs = 48000
    nperseg = 960
    h = _H_weighting(nperseg, fs)
    cases["h_weighting"] = {"n": nperseg, "fs": fs, "h": h.tolist()}

    zi = np.arange(1, 48, 1) / 2
    gzi = _gzi_weighting(zi)
    cases["gzi_weighting"] = {"zi": zi.tolist(), "gzi": gzi.tolist()}

    bark_axis = np.linspace(0.0, 24.0, 50)
    a0 = _ear_filter_coeff(bark_axis)
    cases["ear_filter_coeff"] = {"bark_axis": bark_axis.tolist(), "a0": a0.tolist()}

    # A reduced sampling rate keeps these end-to-end cases' arrays small
    # without changing what's being validated — `_roughness_dw_main_calc`
    # and `roughness_dw`/`roughness_dw_freq` have no dependence on the
    # absolute value of `fs`, only on the fc/fmod/fs ratios (fc=1000 Hz at
    # fs=4800 Hz is unrealistic as audio, but exercises exactly the same
    # code paths as fc=1000 Hz at fs=48000 Hz would).
    fs_small = 4800

    # --- _roughness_dw_main_calc on a real am-tone spectrum -----------------
    stimulus = am_tone(fs_small, fc=1000, fmod=70, dB=60, duration=0.2)
    spec, freq_axis = comp_spectrum(stimulus, fs_small, nfft="default", window="blackman", db=False)
    hWeight_small = _H_weighting(len(stimulus), fs_small)
    R, R_spec, bark = _roughness_dw_main_calc(spec, freq_axis, fs_small, gzi, hWeight_small)
    cases["main_calc"] = {
        "spec_re": spec.real.tolist(), "spec_im": spec.imag.tolist(),
        "freq_axis": freq_axis.tolist(), "fs": fs_small,
        "R": float(R), "R_spec": np.asarray(R_spec).tolist(), "bark": np.asarray(bark).tolist(),
    }

    # --- roughness_dw (time signal, multi-segment) ---------------------------
    stimulus_long = am_tone(fs_small, fc=1000, fmod=70, dB=60, duration=0.6)
    R, R_spec, bark, time = roughness_dw(stimulus_long, fs_small, overlap=0.5)
    cases["roughness_dw"] = {
        "signal": stimulus_long.tolist(), "fs": fs_small, "overlap": 0.5,
        "R": np.asarray(R).tolist(), "R_spec": np.asarray(R_spec).tolist(),
        "bark": np.asarray(bark).tolist(), "time": np.asarray(time).tolist(),
    }

    # --- roughness_dw_freq (1-D spectrum) -------------------------------------
    spec2, freqs2 = comp_spectrum(stimulus, fs_small, db=False)
    R, R_spec, bark = roughness_dw_freq(np.abs(spec2), freqs2)
    cases["roughness_dw_freq"] = {
        "spectrum": np.abs(spec2).tolist(), "freqs": freqs2.tolist(),
        "R": float(R), "R_spec": np.asarray(R_spec).tolist(), "bark": np.asarray(bark).tolist(),
    }

    # --- reference curves (Zwicker & Fastl / Daniel & Weber) for the
    # conformance gate: digitised curves from `validations/.../input/references.py`,
    # re-exported here as (fc, fmod) -> R lookups so the Rust test doesn't need
    # to re-transcribe the underlying digitised arrays.
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent
                           / "validations/sq_metrics/roughness_dw/input"))
    from references import ref_zf, ref_dw

    fc_list = [125, 250, 500, 1000, 2000, 4000, 8000]
    fmod_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160]
    ref_table = []
    for fc in fc_list:
        for fmod in fmod_list:
            ref_table.append(
                {
                    "fc": fc, "fmod": fmod,
                    "ref_zf": float(ref_zf(fc, fmod)),
                    "ref_dw": float(ref_dw(fc, fmod)),
                }
            )
    cases["reference_curve"] = ref_table

    OUT.write_text(json.dumps(cases))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
