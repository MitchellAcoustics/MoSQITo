#!/usr/bin/env python3
"""Export golden vectors for Phase 2's foundation utilities against real MoSQITo.

Covers `mosqito.utils.conversion.{bark2freq,freq2bark,db2amp,spectrum2dBA}`,
`mosqito.utils.LTQ`, `mosqito.sound_level_meter.comp_spectrum`, and
`mosqito.sound_level_meter.freq_band_synthesis`.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden_utils.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from mosqito.utils.conversion.bark2freq import bark2freq
from mosqito.utils.conversion.freq2bark import freq2bark
from mosqito.utils.conversion.db2amp import db2amp
from mosqito.utils.conversion.spectrum2dBA import spectrum2dBA
from mosqito.utils.LTQ import LTQ
from mosqito.sound_level_meter.comp_spectrum import comp_spectrum
from mosqito.sound_level_meter.freq_band_synthesis import freq_band_synthesis
from mosqito.utils.sine_wave_generator import sine_wave_generator
from mosqito.utils.am_sine_generator import am_sine_generator
from mosqito.utils.fm_sine_generator import fm_sine_generator
from mosqito.sq_metrics.loudness.utils.sone_to_phon import sone_to_phon
from mosqito.sq_metrics.loudness.utils.equal_loudness_contours import equal_loudness_contours

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/mosqito-core/tests/golden_utils.json"

RNG = np.random.default_rng(20250101)


def main() -> None:
    cases: dict[str, object] = {}
    cases["versions"] = {"numpy": np.__version__}

    # --- bark2freq / freq2bark ---------------------------------------------
    bark_axis = np.arange(0.0, 24.55, 0.37)
    cases["bark2freq"] = {
        "bark": bark_axis.tolist(),
        "freq": bark2freq(bark_axis).tolist(),
    }
    freq_axis = np.arange(0.0, 20500.0, 137.0)
    cases["freq2bark"] = {
        "freq": freq_axis.tolist(),
        "bark": freq2bark(freq_axis).tolist(),
    }

    # --- db2amp --------------------------------------------------------------
    db_vals = np.array([-40.0, -6.0, 0.0, 6.0, 20.0, 94.0])
    cases["db2amp"] = {
        "db": db_vals.tolist(),
        "ref1": [float(db2amp(d, ref=1)) for d in db_vals],
        "ref2e-5": [float(db2amp(d, ref=2e-5)) for d in db_vals],
    }

    # --- spectrum2dBA --------------------------------------------------------
    fs = 48000
    n = 256
    spectrum_db = 60.0 + 10.0 * np.sin(np.linspace(0, 6.0, n))
    cases["spectrum2dBA"] = {
        "spectrum_db": spectrum_db.tolist(),
        "fs": fs,
        "dba": spectrum2dBA(spectrum_db, fs).tolist(),
    }

    # --- LTQ -------------------------------------------------------------
    bark_query = np.arange(0.0, 24.6, 0.83)
    cases["ltq"] = {
        "bark": bark_query.tolist(),
        "zwicker": LTQ(bark_query, reference="zwicker").tolist(),
        "roughness": LTQ(bark_query, reference="roughness").tolist(),
    }

    # --- comp_spectrum -------------------------------------------------------
    def am_tone(n, fs, fc, fm):
        t = np.arange(n) / fs
        return (0.5 * (1 + np.sin(2 * np.pi * fm * t))) * np.sin(2 * np.pi * fc * t)

    nfft = 4800
    sig1d = am_tone(nfft, fs, 1000.0, 70.0)
    spec_h_db, freq_h = comp_spectrum(sig1d, fs, window="hanning", db=True)
    spec_b_cplx, freq_b = comp_spectrum(sig1d, fs, window="blackman", db=False)
    cases["comp_spectrum_1d"] = {
        "signal": sig1d.tolist(),
        "fs": fs,
        "hanning_db": spec_h_db.tolist(),
        "hanning_freq": freq_h.tolist(),
        "blackman_re": spec_b_cplx.real.tolist(),
        "blackman_im": spec_b_cplx.imag.tolist(),
        "blackman_freq": freq_b.tolist(),
    }

    sig2d = np.stack([am_tone(nfft, fs, 1000.0, 70.0), am_tone(nfft, fs, 2000.0, 40.0)], axis=1)
    spec2d_db, freq2d = comp_spectrum(sig2d, fs, window="hanning", db=True)
    cases["comp_spectrum_2d"] = {
        "signal": sig2d.tolist(),
        "fs": fs,
        "hanning_db": spec2d_db.tolist(),
        "hanning_freq": freq2d.tolist(),
    }

    # --- freq_band_synthesis --------------------------------------------------
    freqs = np.arange(1, nfft // 2 + 1) * (fs / nfft)
    fmin = np.array([100.0, 300.0, 630.0, 1250.0])
    fmax = np.array([300.0, 630.0, 1250.0, 2500.0])
    band_levels, band_centers = freq_band_synthesis(spec_h_db, freq_h, fmin, fmax)
    cases["freq_band_synthesis"] = {
        "spectrum_db": spec_h_db.tolist(),
        "freqs": freq_h.tolist(),
        "fmin": fmin.tolist(),
        "fmax": fmax.tolist(),
        "band_levels": band_levels.tolist(),
        "band_centers": band_centers.tolist(),
    }

    # --- generators ------------------------------------------------------
    sig, time = sine_wave_generator(fs=48000, d=0.05, freq=200.0, spl_level=65.0)
    cases["sine_wave_generator"] = {
        "fs": 48000, "d": 0.05, "freq": 200.0, "spl_level": 65.0,
        "signal": sig.tolist(), "time": time.tolist(),
    }

    n_mod = 2400
    t_mod = np.arange(n_mod) / 48000
    xmod = 0.6 * np.sin(2 * np.pi * 10.0 * t_mod)
    y_am, m_am = am_sine_generator(xmod, fs=48000, fc=1000.0, spl_level=70.0)
    cases["am_sine_generator"] = {
        "xmod": xmod.tolist(), "fs": 48000, "fc": 1000.0, "spl_level": 70.0,
        "y_am": y_am.tolist(), "m": float(m_am),
    }

    y_fm, inst_freq, f_delta, m_fm = fm_sine_generator(
        xmod, fs=48000, fc=500.0, k=200.0, spl_level=70.0
    )
    cases["fm_sine_generator"] = {
        "xmod": xmod.tolist(), "fs": 48000, "fc": 500.0, "k": 200.0, "spl_level": 70.0,
        "y_fm": y_fm.tolist(), "inst_freq": inst_freq.tolist(),
        "f_delta": float(f_delta), "m": float(m_fm),
    }

    # --- sone_to_phon / equal_loudness_contours ---------------------------
    sones = [0.001, 0.5, 0.999, 1.0, 1.5, 2.0, 4.0, 10.0, 40.0]
    cases["sone_to_phon"] = {
        "sones": sones,
        "phons": [float(sone_to_phon(s)) for s in sones],
    }

    spl40, freq40 = equal_loudness_contours(40.0)
    spl80, freq80 = equal_loudness_contours(80.0)
    cases["equal_loudness_contours"] = {
        "phon_40": {"spl": spl40.tolist(), "freq": freq40.tolist()},
        "phon_80": {"spl": spl80.tolist(), "freq": freq80.tolist()},
    }

    OUT.write_text(json.dumps(cases))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
