#!/usr/bin/env python3
"""Re-derives ECMA-418-2 roughness's calibration factor `c_R` after fixing
`_lowpass_filter.py`'s bug (see `DEVIATIONS.md`'s D-ecma-5 and the
"Bug fix (unsanctioned by the paper)" entry, and
`rust/crates/mosqito-core/src/roughness/ecma/lowpass_filter.rs`).

Reproduces `roughness_ecma`'s pipeline directly (calling MoSQITo's own,
unmodified helper functions for every stage except the two under
correction: `_lowpass_filter` and `_non_linear_transform`'s `c_R`), so a
corrected lowpass filter and an arbitrary `c_R` can be substituted in
without needing to monkeypatch compiled bytecode.

With `tau`/the rising-falling classification fixed, `R` is exactly linear
in `c_R` (the classification only compares `R_hat` values scaled by the
same positive `c_R`, so scale never flips a comparison; everything
downstream is then a scale-invariant combination). That means the
expensive part of this fit — the full pipeline up to `non_linear_transform`
— only needs to run once per test point, at `c_R = 1`; `c_R` itself is then
recovered in closed form as a least-squares fit against the reference
values: `c_R = sum(R1 * ref) / sum(R1 ** 2)`.

Run from the ``rust/`` directory (takes several minutes: 105 points x a
1.5 s ECMA-418-2 roughness computation each, in pure Python)::

    ../.venv/bin/python tools/refit_c_r.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "validations/sq_metrics/roughness_ecma"))

from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import (
    _auditory_filters_centre_freq,
)
from mosqito.sq_metrics.loudness.loudness_ecma._band_pass_signals import _band_pass_signals
from mosqito.sq_metrics.loudness.loudness_ecma._ecma_time_segmentation import (
    _ecma_time_segmentation,
)
from mosqito.sq_metrics.loudness.loudness_ecma._loudness_from_bandpass import (
    _loudness_from_bandpass,
)
from mosqito.sq_metrics.loudness.loudness_ecma._preprocessing import _preprocessing
from mosqito.sq_metrics.roughness.roughness_ecma._estimate_fund_mod_rate import (
    _estimate_fund_mod_rate,
)
from mosqito.sq_metrics.roughness.roughness_ecma._interpolation_50 import _interpolation_50
from mosqito.sq_metrics.roughness.roughness_ecma._noise_reduction import _noise_reduction
from mosqito.sq_metrics.roughness.roughness_ecma._peak_picking import _peak_picking
from mosqito.sq_metrics.roughness.roughness_ecma._von_hann_window import _von_hann_window
from mosqito.sq_metrics.roughness.roughness_ecma._weighting import (
    _f_max,
    _high_mod_rate_weighting,
    _low_mod_rate_weighting,
    _Q2_high,
    _Q2_low,
    _r_max,
)
from mosqito.utils.am_sine_generator import am_sine_generator
from numpy.fft import fft
from scipy.signal import decimate, hilbert

from input.references import ref_ecma, ref_zf  # noqa: E402

FC_LIST = [125, 250, 500, 1000, 2000, 4000, 8000]
FMOD_VECTOR = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 300, 400]


def lowpass_filter_correct(r_hat: np.ndarray) -> np.ndarray:
    """The corrected ECMA-418-2 Eq. 109/110 lowpass: a genuine first-order
    recursive filter over time (axis 0), per band (axis 1) independently."""
    n50, cbf = r_hat.shape
    r_time_spec = np.zeros((n50, cbf))
    r_time_spec[0, :] = r_hat[0, :]
    prev = r_hat[0, :].copy()
    for l in range(1, n50):
        rising = r_hat[l, :] >= prev
        tau = np.where(rising, 0.0625, 0.5)
        e = np.exp(-1 / (50 * tau))
        cur = r_hat[l, :] * (1 - e) + prev * e
        r_time_spec[l, :] = cur
        prev = cur
    return r_time_spec


def non_linear_transform(r_est: np.ndarray, c_r: float) -> np.ndarray:
    """`_non_linear_transform.py` with `c_R` as a parameter instead of its
    hardcoded constant."""
    n50, cbf = r_est.shape
    r_sq_mean = np.sqrt(np.sum(r_est**2, axis=1) / cbf)
    r_lin_mean = np.mean(r_est, axis=1)
    b = np.zeros(n50)
    b[r_lin_mean != 0] = r_sq_mean[r_lin_mean != 0] / r_lin_mean[r_lin_mean != 0]
    e = 0.25 * np.tanh(1.75 * (b - 2.5)) + 0.7
    return c_r * np.power(r_est, e[..., np.newaxis])


def roughness_ecma_variant(signal, fs, c_r, lowpass_fn):
    """Reproduces `roughness_ecma.py` end to end, with `_lowpass_filter` and
    `c_R` swapped for the arguments above; every other stage calls
    MoSQITo's own unmodified functions directly."""
    cbf = 53
    center_freq = _auditory_filters_centre_freq()
    sb = 16384
    sh = 4096
    duration = len(signal) / fs

    signal, n_new = _preprocessing(signal, sb, sh)
    bandpass_signals = _band_pass_signals(signal, sb, sh)
    block_array, time_array = _ecma_time_segmentation(bandpass_signals, sb, sh, n_new)
    time_axis = np.array(time_array)[0]
    block_array = np.asarray(block_array)

    n_specific, bark_axis = _loudness_from_bandpass(block_array)
    n_specific = np.array(n_specific).T
    length_l = n_specific.shape[0]

    envelopes = abs(hilbert(block_array))
    envelopes = np.transpose(np.asarray(envelopes), (1, 0, 2))

    sbb = 512
    downsampling_factor = 32
    envelopes_downsampled_ = decimate(envelopes, downsampling_factor // 4, axis=2)
    envelopes_downsampled = decimate(envelopes_downsampled_, 4, axis=2)

    n_specific_max = np.asarray(n_specific).max(axis=1)
    hann_window = _von_hann_window(sbb)
    phi_e0 = np.sum(np.power(envelopes_downsampled * hann_window, 2), axis=2)
    den = n_specific_max[:, np.newaxis] * phi_e0

    dft = (
        abs(fft((envelopes_downsampled * hann_window), axis=2)[:, :, : sbb // 2]) / 2 * np.sqrt(2)
    ) ** 2
    scaling = np.zeros((length_l, cbf))
    scaling[den != 0] = np.power(n_specific[den != 0], 2) / den[den != 0]
    phi_e = scaling[:, :, np.newaxis] * dft

    phi_e = _noise_reduction(phi_e)

    fmax = _f_max(center_freq)
    rmax = _r_max(center_freq)
    q2_high = _Q2_high(center_freq)
    q2_low = _Q2_low(center_freq)

    amplitude = np.zeros((length_l, cbf))
    for l in range(length_l):
        for z in range(cbf):
            f_p, ai = _peak_picking(phi_e[l, z, :])
            n_peak = len(f_p)
            if n_peak == 0:
                amplitude[l, z] = 0
            else:
                ai_tilde = np.empty(n_peak)
                for i0 in range(n_peak):
                    ai_tilde[i0] = _high_mod_rate_weighting(
                        f_p[i0], ai[i0], fmax[z], rmax[z], q2_high[z]
                    )
                mod_rate, a_hat = _estimate_fund_mod_rate(f_p, ai_tilde)
                amplitude[l, z] = _low_mod_rate_weighting(mod_rate, a_hat, fmax[z], q2_low[z])

    amplitude[amplitude < 0.074376] = 0

    amplitude_50, t_50 = _interpolation_50(amplitude, time_axis, duration)
    r_est = np.clip(amplitude_50, 0, None)
    r_time_spec_temp = non_linear_transform(r_est, c_r)
    r_time_spec = lowpass_fn(r_time_spec_temp)

    r_spec = np.mean(r_time_spec[10:, :], axis=0)
    r_time = 0.5 * np.sum(r_time_spec, axis=1)
    r = np.percentile(r_time, 90)

    return r, r_time, r_spec, bark_axis, t_50


def main() -> None:
    duration = 1.5
    fs = 48000
    time = np.linspace(0, duration, int(duration * fs))
    level = 60

    r1 = []
    ref_ecma_vals = []
    ref_zf_vals = []
    for fc in FC_LIST:
        for fmod in FMOD_VECTOR:
            xmod = np.sin(2 * np.pi * fmod * time)
            stimulus, _ = am_sine_generator(xmod, fs, fc, level)
            r, *_ = roughness_ecma_variant(stimulus, fs, 1.0, lowpass_filter_correct)
            r1.append(r)
            ref_ecma_vals.append(ref_ecma(fc, fmod))
            ref_zf_vals.append(ref_zf(fc, fmod))
            print(f"fc={fc} fmod={fmod} R(c_R=1)={r:.6f} ref_ecma={ref_ecma(fc, fmod):.4f}")

    r1 = np.array(r1)
    ref_ecma_vals = np.array(ref_ecma_vals)
    ref_zf_vals = np.array(ref_zf_vals)

    c_r = np.sum(r1 * ref_ecma_vals) / np.sum(r1 * r1)
    r = c_r * r1

    within_zf = np.abs(r - ref_zf_vals) <= 0.1
    rel_err_ecma = np.abs(r - ref_ecma_vals) / ref_ecma_vals

    print(f"\nfit c_R = {c_r!r}")
    print(f"within +/-0.1 asper of Zwicker-Fastl: {within_zf.sum()}/{len(r)}")
    print(f"mean relative error vs ECMA Annex C: {rel_err_ecma.mean():.4f}")
    print(f"max relative error vs ECMA Annex C: {rel_err_ecma.max():.4f}")
    print(f"count within 30% of ECMA Annex C: {(rel_err_ecma <= 0.30).sum()}/{len(r)}")


if __name__ == "__main__":
    main()
