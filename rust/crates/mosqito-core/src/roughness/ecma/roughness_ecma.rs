//! Top-level ECMA-418-2 (2nd Ed, 2022) §7.1 roughness entry point, matching
//! `roughness_ecma.py`, with the corrections detailed in this module's
//! parent doc comment and in `lowpass_filter.rs`/`non_linear_transform.rs`.

use ndarray::Array2;
use rayon::prelude::*;

use super::envelope_spectrum::{band_spectrum, N_BINS};
use super::estimate_fund_mod_rate::estimate_fund_mod_rate;
use super::lowpass_filter::lowpass_filter;
use super::noise_reduction::noise_reduction;
use super::non_linear_transform::non_linear_transform;
use super::peak_picking::peak_picking;
use super::weighting::{
    f_max, high_mod_rate_weighting, low_mod_rate_weighting, q2_high, q2_low, r_max,
};
use crate::dsp::{pchip, percentile_linear, resample};
use crate::loudness::ecma::{
    auditory_filters_centre_freq, band_pass_signals, n_blocks, preprocess,
    specific_loudness_for_band, LTQ_Z,
};

const CBF: usize = 53;
const SB: usize = 16384;
const SH: usize = 4096;
const FS_ECMA: f64 = 48000.0;

/// Amplitude threshold below which a block/band's modulation amplitude is
/// discarded, ECMA-418-2 §7.1.5.4.
const AMPLITUDE_FLOOR: f64 = 0.074376;

/// Calibration factor `c_R` [asper/Bark_HMS], re-fit against the ECMA-418-2
/// Annex C reference values after correcting `_lowpass_filter.py`'s bug —
/// see `lowpass_filter.rs`'s module doc and `DEVIATIONS.md`'s D-ecma-5.
const C_R: f64 = 0.03288;

/// `(R, R_time, R_specific, bark_axis, time_axis)` — [`roughness_ecma`]'s
/// result.
type RoughnessEcmaResult = (f64, Vec<f64>, [f64; CBF], [f64; CBF], Vec<f64>);

/// Computes the specific and total roughness per ECMA-418-2 (2nd Ed, 2022)
/// §7.1, for stationary signals. Matches `roughness_ecma(signal, fs)`.
///
/// Resamples to 48 kHz first if `fs != 48000`.
pub fn roughness_ecma(signal: &[f64], fs: f64) -> RoughnessEcmaResult {
    let signal_orig_len = signal.len();
    let signal = if fs != FS_ECMA {
        resample(signal, (FS_ECMA * signal.len() as f64 / fs) as usize)
    } else {
        signal.to_vec()
    };
    let duration = signal_orig_len as f64 / fs;

    let centre_freq = auditory_filters_centre_freq();
    let (padded, n_new) = preprocess(&signal, SB, SH);
    let bandpass = band_pass_signals(&padded, FS_ECMA);
    let l = n_blocks(n_new, SH);

    // Specific loudness per band (parallel; no cross-band coupling), and
    // the shared time axis (identical across bands for scalar sb/sh).
    let per_band_loudness: Vec<(Vec<f64>, Vec<f64>)> = (0..CBF)
        .into_par_iter()
        .map(|z| specific_loudness_for_band(&bandpass[z], SB, SH, n_new, LTQ_Z[z]))
        .collect();
    let time_axis = per_band_loudness[0].1.clone();

    let mut n_specific = Array2::<f64>::zeros((l, CBF));
    for (z, (spec, _time)) in per_band_loudness.iter().enumerate() {
        for (t, &v) in spec.iter().enumerate() {
            n_specific[[t, z]] = v;
        }
    }
    // Max across bands, per time block (`N_specific.max(axis=1)` in Python
    // — axis 1 there is the *band* axis after transposition).
    let n_specific_max: Vec<f64> = (0..l)
        .map(|t| {
            (0..CBF)
                .map(|z| n_specific[[t, z]])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    // Envelope power spectrum per band (parallel; no cross-band coupling).
    let band_spectra: Vec<_> = (0..CBF)
        .into_par_iter()
        .map(|z| band_spectrum(&bandpass[z], SB, SH, n_new))
        .collect();

    // Cross-band-dependent scaling (Eq. 85), assembled into [time][band][bin].
    let mut phi_e_raw = vec![vec![vec![0.0f64; N_BINS]; CBF]; l];
    for (z, bs) in band_spectra.iter().enumerate() {
        for t in 0..l {
            let den = n_specific_max[t] * bs.phi_e0[t];
            let scaling = if den != 0.0 {
                n_specific[[t, z]].powi(2) / den
            } else {
                0.0
            };
            for (k, dst) in phi_e_raw[t][z].iter_mut().enumerate() {
                *dst = scaling * bs.dft[[t, k]];
            }
        }
    }
    let phi_e = noise_reduction(&phi_e_raw);

    let fmax: Vec<f64> = centre_freq.iter().map(|&f| f_max(f)).collect();
    let rmax: Vec<f64> = centre_freq.iter().map(|&f| r_max(f)).collect();
    let q2h: Vec<f64> = centre_freq.iter().map(|&f| q2_high(f)).collect();
    let q2l: Vec<f64> = centre_freq.iter().map(|&f| q2_low(f)).collect();

    // Spectral weighting and modulation-rate estimation, per (l, z) — the
    // ECMA-418-2 stage the plan flags as the other major hot spot; run in
    // parallel across every (time, band) pair.
    let mut amplitude = Array2::<f64>::zeros((l, CBF));
    let amps: Vec<((usize, usize), f64)> = (0..l)
        .into_par_iter()
        .flat_map(|t| {
            let (phi_e, fmax, rmax, q2h, q2l) = (&phi_e, &fmax, &rmax, &q2h, &q2l);
            (0..CBF).into_par_iter().map(move |z| {
                let (f_p, ai) = peak_picking(&phi_e[t][z]);
                let amp = if f_p.is_empty() {
                    0.0
                } else {
                    let ai_tilde: Vec<f64> = f_p
                        .iter()
                        .zip(&ai)
                        .map(|(&fp, &a)| high_mod_rate_weighting(fp, a, fmax[z], rmax[z], q2h[z]))
                        .collect();
                    let (mod_rate, a_hat) = estimate_fund_mod_rate(&f_p, &ai_tilde);
                    low_mod_rate_weighting(mod_rate, &a_hat, fmax[z], q2l[z])
                };
                ((t, z), amp)
            })
        })
        .collect();
    for ((t, z), amp) in amps {
        amplitude[[t, z]] = if amp < AMPLITUDE_FLOOR { 0.0 } else { amp };
    }
    // Interpolation to 50 Hz (Eq. 103), per band.
    let n50 = (duration * 50.0) as usize;
    let t_50: Vec<f64> = (0..n50).map(|i| i as f64 / 50.0).collect();

    let mut r_est = vec![vec![0.0f64; CBF]; n50];
    for z in 0..CBF {
        let y: Vec<f64> = (0..l).map(|t| amplitude[[t, z]]).collect();
        let interpolated = pchip(&time_axis, &y, &t_50);
        for (t, &v) in interpolated.iter().enumerate() {
            r_est[t][z] = v.max(0.0);
        }
    }

    let r_time_spec_temp = non_linear_transform(&r_est, C_R);
    let r_time_spec = lowpass_filter(&r_time_spec_temp);

    // Representative values (§7.1.8): skip the first 10 (50 Hz) blocks
    // (0.2 s) as a transient guard for the band-mean specific roughness.
    let mut r_spec = [0.0f64; CBF];
    let n_after_transient = r_time_spec.len().saturating_sub(10);
    for z in 0..CBF {
        let sum: f64 = r_time_spec.iter().skip(10).map(|row| row[z]).sum();
        r_spec[z] = sum / n_after_transient as f64;
    }

    let r_time: Vec<f64> = r_time_spec
        .iter()
        .map(|row| 0.5 * row.iter().sum::<f64>())
        .collect();
    let r = if r_time.is_empty() {
        0.0
    } else {
        percentile_linear(&r_time, 90.0)
    };

    let bark_axis: [f64; CBF] = std::array::from_fn(|i| 0.5 + i as f64 * 0.5);

    (r, r_time, r_spec, bark_axis, t_50)
}
