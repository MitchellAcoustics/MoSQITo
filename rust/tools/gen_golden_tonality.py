#!/usr/bin/env python3
"""Export golden vectors for tonality (TNR/PR) against real MoSQITo.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden_tonality.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from mosqito.sound_level_meter.noct_spectrum._getFrequencies import _getFrequencies
from mosqito.sound_level_meter.comp_spectrum import comp_spectrum
from mosqito.sq_metrics.tonality.tone_to_noise_ecma._critical_band import (
    _critical_band,
    _lower_critical_band,
    _upper_critical_band,
)
from mosqito.sq_metrics.tonality.tone_to_noise_ecma._LTH import _LTH
from mosqito.sq_metrics.tonality.tone_to_noise_ecma._spectrum_smoothing import (
    _spectrum_smoothing,
)
from mosqito.sq_metrics.tonality.tone_to_noise_ecma._screening_for_tones import (
    _screening_for_tones,
)
from mosqito.sq_metrics.tonality.tone_to_noise_ecma._tnr_main_calc import _tnr_main_calc
from mosqito.sq_metrics.tonality.prominence_ratio_ecma._pr_main_calc import _pr_main_calc
from mosqito.sq_metrics.tonality.tone_to_noise_ecma.tnr_ecma_st import tnr_ecma_st
from mosqito.sq_metrics.tonality.tone_to_noise_ecma.tnr_ecma_freq import tnr_ecma_freq
from mosqito.sq_metrics.tonality.tone_to_noise_ecma.tnr_ecma_perseg import tnr_ecma_perseg
from mosqito.sq_metrics.tonality.prominence_ratio_ecma.pr_ecma_st import pr_ecma_st
from mosqito.sq_metrics.tonality.prominence_ratio_ecma.pr_ecma_freq import pr_ecma_freq
from mosqito.sq_metrics.tonality.prominence_ratio_ecma.pr_ecma_perseg import pr_ecma_perseg

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/mosqito-core/tests/golden_tonality.json"


def two_tone_stimulus(fs, duration, dB=60, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(0, duration, 1 / fs)
    stimulus = (
        np.sin(2 * np.pi * 1000 * time)
        + 0.5 * np.sin(2 * np.pi * 3000 * time)
        + rng.normal(0, 0.3, len(time))
    )
    rms = np.sqrt(np.mean(stimulus**2))
    ampl = 0.00002 * 10 ** (dB / 20) / rms
    return stimulus * ampl


def main() -> None:
    cases: dict[str, object] = {}

    # --- critical_band / lower / upper --------------------------------------
    f0_list = [100.0, 300.0, 500.0, 1000.0, 1600.0, 2000.0, 5000.0, 10000.0]
    cb_cases = []
    for f0 in f0_list:
        f1, f2 = _critical_band(f0)
        lf1, lf2 = _lower_critical_band(f0)
        uf1, uf2 = _upper_critical_band(f0)
        cb_cases.append(
            {"f0": f0, "f1": f1, "f2": f2, "lf1": lf1, "lf2": lf2, "uf1": uf1, "uf2": uf2}
        )
    cases["critical_band"] = cb_cases

    # --- LTH -----------------------------------------------------------------
    lth_freqs = np.array([89.2, 100.0, 304.9, 305.0, 1000.0, 2229.9, 2230.0, 7000.0, 11199.9])
    cases["lth"] = {"freqs": lth_freqs.tolist(), "values": _LTH(lth_freqs).tolist()}

    # --- get_frequencies (_getFrequencies) ------------------------------------
    gf = _getFrequencies(90.0, 11200.0, 24, G=10, fr=1000)["f"]
    cases["get_frequencies"] = {"f1": gf[:, 0].tolist(), "fm": gf[:, 1].tolist(), "f2": gf[:, 2].tolist()}

    # --- spectrum_smoothing / screening_for_tones, 1-D and multi-segment -----
    # A moderate resolution: fine enough (df ~ 2.5 Hz) that 1/24-octave bands
    # near 90 Hz still span several bins, without the array sizes of a full
    # 48 kHz/several-hundred-ms spectrum.
    fs_small = 9600
    duration_small = 0.2
    stim1 = two_tone_stimulus(fs_small, duration_small, seed=1)
    spec_db1, freq_axis1 = comp_spectrum(stim1, fs_small, db=True)
    freq_index1 = np.where((freq_axis1 > 89.1) & (freq_axis1 < 11200))[0]
    freqs1 = freq_axis1[freq_index1]
    spec1 = spec_db1[freq_index1]

    smooth1 = _spectrum_smoothing(freqs1, spec1, 24, 90.0, 11200.0, freqs1)
    tones1 = _screening_for_tones(freqs1, spec1, "smoothed", 90.0, 11200.0)
    cases["smoothing_1d"] = {
        "freqs": freqs1.tolist(),
        "spec_db": spec1.tolist(),
        "smooth_spec": np.asarray(smooth1).tolist(),
        "tones": np.asarray(tones1, dtype=float).tolist(),
    }

    # Multi-segment: 3 identical-shape segments (as tnr_ecma_perseg's
    # internal call always produces), built from three different stimuli so
    # the segments are not literally identical.
    nseg = 3
    spec_db_2d = np.empty((nseg, len(freq_index1)))
    freqs_2d = np.empty((nseg, len(freq_index1)))
    for i in range(nseg):
        stim_i = two_tone_stimulus(fs_small, duration_small, seed=10 + i)
        spec_db_i, freq_axis_i = comp_spectrum(stim_i, fs_small, db=True)
        spec_db_2d[i, :] = spec_db_i[freq_index1]
        freqs_2d[i, :] = freq_axis_i[freq_index1]

    smooth2 = _spectrum_smoothing(freqs_2d, spec_db_2d.T, 24, 90.0, 11200.0, freqs_2d)
    tones2 = _screening_for_tones(freqs_2d, spec_db_2d, "smoothed", 90.0, 11200.0)
    cases["smoothing_2d"] = {
        "freqs": freqs_2d.tolist(),
        "spec_db": spec_db_2d.tolist(),
        "smooth_spec": np.asarray(smooth2).tolist(),
        "tones": [np.asarray(t, dtype=float).tolist() for t in tones2],
    }

    # --- _tnr_main_calc / _pr_main_calc, 1-D and (2-D spec, 1-D freq) --------
    tf1, tnr1, prom1, t_tnr1 = _tnr_main_calc(spec_db1, freq_axis1)
    cases["tnr_main_calc_1d"] = {
        "spectrum_db": spec_db1.tolist(),
        "freq_axis": freq_axis1.tolist(),
        "tones_freqs": np.asarray(tf1, dtype=float).tolist(),
        "tnr": np.asarray(tnr1, dtype=float).tolist(),
        "prominence": np.asarray(prom1, dtype=bool).tolist(),
        "t_tnr": float(np.ravel(t_tnr1)[0]) if np.ravel(t_tnr1).size else 0.0,
    }
    tfp1, pr1, promp1, t_pr1 = _pr_main_calc(spec_db1, freq_axis1)
    cases["pr_main_calc_1d"] = {
        "tones_freqs": np.asarray(tfp1, dtype=float).tolist(),
        "pr": np.asarray(pr1, dtype=float).tolist(),
        "prominence": np.asarray(promp1, dtype=bool).tolist(),
        "t_pr": float(np.ravel(t_pr1)[0]) if np.ravel(t_pr1).size else 0.0,
    }

    # (2-D spectrum_db, 1-D freq_axis) branch: build spectrum_db as
    # (nperseg, nseg) with a *shared* 1-D freq_axis, matching what
    # tnr_ecma_perseg's internal comp_spectrum call actually produces.
    spec_db_full_2d = np.empty((len(freq_axis1), nseg))
    for i in range(nseg):
        stim_i = two_tone_stimulus(fs_small, duration_small, seed=20 + i)
        spec_db_i, _ = comp_spectrum(stim_i, fs_small, db=True)
        spec_db_full_2d[:, i] = spec_db_i

    tf2, tnr2, prom2, t_tnr2 = _tnr_main_calc(spec_db_full_2d, freq_axis1)
    cases["tnr_main_calc_2d"] = {
        "spectrum_db": spec_db_full_2d.tolist(),
        "freq_axis": freq_axis1.tolist(),
        "tones_freqs": [np.asarray(t, dtype=float).tolist() for t in tf2],
        "tnr": [np.asarray(t, dtype=float).tolist() for t in tnr2],
        "prominence": [np.asarray(t, dtype=bool).tolist() for t in prom2],
        "t_tnr": np.asarray(t_tnr2, dtype=float).tolist(),
    }
    tfp2, pr2, promp2, t_pr2 = _pr_main_calc(spec_db_full_2d, freq_axis1)
    cases["pr_main_calc_2d"] = {
        "tones_freqs": [np.asarray(t, dtype=float).tolist() for t in tfp2],
        "pr": [np.asarray(t, dtype=float).tolist() for t in pr2],
        "prominence": [np.asarray(t, dtype=bool).tolist() for t in promp2],
        "t_pr": np.asarray(t_pr2, dtype=float).tolist(),
    }

    # --- Full entry points, realistic 48 kHz signal --------------------------
    # 24 kHz keeps Nyquist comfortably above the 11200 Hz TNR/PR range of
    # interest (fixed, absolute Hz values per ECMA-74 — unlike roughness_dw,
    # this can't be rescaled to an arbitrary fs) while keeping the golden
    # file's arrays much smaller than a full 48 kHz capture would.
    fs = 24000
    duration = 1.0
    stim = two_tone_stimulus(fs, duration, seed=2)
    # Stored once — `tnr_ecma_freq`'s spectrum and every other case's signal
    # are re-derived from this same stimulus in the Rust test (comp_spectrum
    # itself is already validated bit-for-bit in `golden_utils.rs`), instead
    # of duplicating multi-thousand-sample arrays per case.
    cases["stimulus"] = {"signal": stim.tolist(), "fs": fs}

    t_tnr, tnr, prom, tf = tnr_ecma_st(stim, fs, prominence=False)
    cases["tnr_ecma_st"] = {
        "t_tnr": float(np.ravel(t_tnr)[0]),
        "tnr": np.asarray(tnr, dtype=float).tolist(),
        "prominence": np.asarray(prom, dtype=bool).tolist(),
        "tones_freqs": np.asarray(tf, dtype=float).tolist(),
    }

    spec, freq_axis = comp_spectrum(stim, fs, db=False)
    t_tnr_f, tnr_f, prom_f, tf_f = tnr_ecma_freq(np.abs(spec), freq_axis, prominence=False)
    cases["tnr_ecma_freq"] = {
        "t_tnr": float(np.ravel(t_tnr_f)[0]),
        "tnr": np.asarray(tnr_f, dtype=float).tolist(),
        "prominence": np.asarray(prom_f, dtype=bool).tolist(),
        "tones_freqs": np.asarray(tf_f, dtype=float).tolist(),
    }

    t_tnr_p, tnr_p, promi_p, freqs_p, time_p = tnr_ecma_perseg(stim, fs, prominence=False, overlap=0.5)
    cases["tnr_ecma_perseg"] = {
        "overlap": 0.5,
        "t_tnr": np.asarray(t_tnr_p, dtype=float).tolist(),
        "tnr": np.nan_to_num(np.asarray(tnr_p, dtype=float), nan=-999.0).tolist(),
        "prominence": np.asarray(promi_p, dtype=bool).tolist(),
        "freqs": np.asarray(freqs_p, dtype=float).tolist(),
        "time": np.asarray(time_p, dtype=float).tolist(),
    }

    t_pr, pr, promp, tfp = pr_ecma_st(stim, fs, prominence=False)
    cases["pr_ecma_st"] = {
        "t_pr": float(np.ravel(t_pr)[0]),
        "pr": np.asarray(pr, dtype=float).tolist(),
        "prominence": np.asarray(promp, dtype=bool).tolist(),
        "tones_freqs": np.asarray(tfp, dtype=float).tolist(),
    }

    t_pr_f, pr_f, promp_f, tfp_f = pr_ecma_freq(np.abs(spec), freq_axis, prominence=False)
    cases["pr_ecma_freq"] = {
        "t_pr": float(np.ravel(t_pr_f)[0]),
        "pr": np.asarray(pr_f, dtype=float).tolist(),
        "prominence": np.asarray(promp_f, dtype=bool).tolist(),
        "tones_freqs": np.asarray(tfp_f, dtype=float).tolist(),
    }

    t_pr_p, pr_p, promi_pp, freqs_pp, time_pp = pr_ecma_perseg(stim, fs, prominence=False, overlap=0.5)
    cases["pr_ecma_perseg"] = {
        "t_pr": np.asarray(t_pr_p, dtype=float).tolist(),
        "pr": np.nan_to_num(np.asarray(pr_p, dtype=float), nan=-999.0).tolist(),
        "prominence": np.asarray(promi_pp, dtype=bool).tolist(),
        "freqs": np.asarray(freqs_pp, dtype=float).tolist(),
        "time": np.asarray(time_pp, dtype=float).tolist(),
    }

    OUT.write_text(json.dumps(cases))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
