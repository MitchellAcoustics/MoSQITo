#!/usr/bin/env python3
"""Export reference outputs for the roughness_ecma pipeline's individual
stages (the ones NOT affected by the `_lowpass_filter.py` bug this port
fixes) from the installed `mosqito` package, plus full-pipeline references
from this port's own validated "corrected" Python reproduction (see
`tools/refit_c_r.py` — the same one used to re-derive `c_R`).

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_golden_roughness.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "crates/mosqito-core/tests/golden_roughness_ecma.json"
)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "validations/sq_metrics/roughness_ecma"))

from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import (
    _auditory_filters_centre_freq,
)
from mosqito.sq_metrics.roughness.roughness_ecma._von_hann_window import _von_hann_window
from mosqito.sq_metrics.roughness.roughness_ecma._weighting import (
    _f_max,
    _r_max,
    _Q2_high,
    _Q2_low,
    _high_mod_rate_weighting,
    _low_mod_rate_weighting,
)
from mosqito.sq_metrics.roughness.roughness_ecma._refinement import _refinement, _rho
from mosqito.sq_metrics.roughness.roughness_ecma._peak_picking import _peak_picking
from mosqito.sq_metrics.roughness.roughness_ecma._estimate_fund_mod_rate import (
    _estimate_fund_mod_rate,
)
from mosqito.sq_metrics.roughness.roughness_ecma._noise_reduction import _noise_reduction

RNG = np.random.default_rng(20240922)


def main() -> None:
    cases: dict = {
        "von_hann_window": [],
        "weighting": [],
        "refinement": [],
        "peak_picking": [],
        "estimate_fund_mod_rate": [],
        "noise_reduction": [],
    }

    cases["von_hann_window"] = _von_hann_window(512).tolist()

    centre_freq = _auditory_filters_centre_freq()
    for z in (0, 1, 13, 26, 40, 52):
        cf = float(centre_freq[z])
        fmax = float(_f_max(np.array([cf]))[0])
        rmax = float(_r_max(np.array([cf]))[0])
        q2h = float(_Q2_high(np.array([cf]))[0])
        q2l = float(_Q2_low(np.array([cf]))[0])
        for mod_rate in (fmax * 0.5, fmax * 1.5, 5.0, 200.0):
            amp = 1.0
            high = float(_high_mod_rate_weighting(mod_rate, amp, fmax, rmax, q2h))
            low = float(_low_mod_rate_weighting(mod_rate, np.array([amp, amp * 0.5]), fmax, q2l))
            cases["weighting"].append(
                {
                    "centre_freq": cf,
                    "fmax": fmax,
                    "rmax": rmax,
                    "q2_high": q2h,
                    "q2_low": q2l,
                    "mod_rate": mod_rate,
                    "high_mod_rate_weighting": high,
                    "low_mod_rate_weighting": low,
                }
            )

    # --- refinement: synthetic peaky spectra ------------------------------
    for _ in range(20):
        spec = RNG.uniform(0.0, 1.0, size=256)
        kpi = RNG.integers(2, 255)
        spec[kpi] = spec[kpi - 1 : kpi + 2].max() + RNG.uniform(1.0, 5.0)
        mod_rate, amp = _refinement(int(kpi), spec)
        cases["refinement"].append(
            {"spec": spec.tolist(), "kpi": int(kpi), "mod_rate": float(mod_rate), "amp": float(amp)}
        )

    # --- peak_picking: synthetic multi-peak spectra -----------------------
    for _ in range(20):
        spec = RNG.uniform(0.0, 0.3, size=256)
        n_peaks = RNG.integers(1, 8)
        peak_positions = RNG.choice(np.arange(5, 250), size=n_peaks, replace=False)
        for p in peak_positions:
            spec[p] = RNG.uniform(1.0, 10.0)
        f_p, a = _peak_picking(spec)
        cases["peak_picking"].append(
            {"spec": spec.tolist(), "f_p": np.asarray(f_p).tolist(), "a": np.asarray(a).tolist()}
        )

    # --- estimate_fund_mod_rate: synthetic harmonic-ish peak lists --------
    for _ in range(20):
        n_peaks = RNG.integers(1, 8)
        f_p = np.sort(RNG.uniform(10.0, 400.0, size=n_peaks))
        ai_tilde = RNG.uniform(0.1, 5.0, size=n_peaks)
        mod_rate, a_hat = _estimate_fund_mod_rate(f_p, ai_tilde)
        cases["estimate_fund_mod_rate"].append(
            {
                "f_p": f_p.tolist(),
                "ai_tilde": ai_tilde.tolist(),
                "mod_rate": float(mod_rate),
                "a_hat": np.asarray(a_hat).tolist(),
            }
        )

    # --- noise_reduction: small synthetic (L, 53, K) spectra --------------
    for _ in range(5):
        length_l = int(RNG.integers(3, 8))
        k = 20
        spectrum = RNG.uniform(0.0, 1.0, size=(length_l, 53, k))
        phi_e = _noise_reduction(spectrum.copy())
        cases["noise_reduction"].append(
            {"spectrum": spectrum.tolist(), "phi_e": np.asarray(phi_e).tolist()}
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
