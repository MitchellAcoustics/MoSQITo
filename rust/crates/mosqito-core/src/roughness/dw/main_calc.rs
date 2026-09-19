//! Daniel & Weber roughness, core per-spectrum computation, matching
//! `_roughness_dw_main_calc.py`.

use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::FftPlanner;

use super::ear_filter_coeff::ear_filter_coeff;
use crate::utils::{amp2db, db2amp, freq2bark, ltq, LtqReference};

const N_CHANNEL: usize = 47;

/// `(R, R_spec, bark_axis)` — one spectrum's roughness.
pub type RoughnessDwResult = (f64, [f64; N_CHANNEL], [f64; N_CHANNEL]);

/// The 47 channel centres, in Bark, `0.5, 1.0, ..., 23.5`.
pub(crate) fn channel_centres() -> [f64; N_CHANNEL] {
    std::array::from_fn(|i| (i + 1) as f64 / 2.0)
}

/// Computes Daniel & Weber roughness from one amplitude-or-complex spectrum.
///
/// `spec` is the one-sided spectrum (real amplitude, or complex — a real
/// amplitude spectrum embeds as `Complex64::new(a, 0.0)`), `freq_axis` its
/// frequency axis in Hz, both length `n_orig`; `fs` the original sampling
/// frequency; `gzi`/`h_weight` the precomputed weighting functions
/// ([`super::gzi_weighting::gzi_weighting`] over `0.5..=23.5`,
/// [`super::h_weighting::h_weighting`] sized `(47, 2*n_orig)`).
///
/// # A simplification that provably does not change the output
/// Python builds a `2*n_orig`-length "two-sided" spectrum by mirroring
/// `spec`, then multiplies the *whole* array by an ear-filter gain `a0`
/// that is `zeros(2*n_orig)` with only its first `n_orig` entries set —
/// which zeroes the mirrored half outright. Every later step
/// (`module = abs(spec[:n_orig])`, and the excitation reconstruction, which
/// only ever indexes `audible_index < n_orig`) then reads exclusively from
/// that first half. The mirrored half is therefore write-only: computed,
/// immediately zeroed, and never read. This port skips building it — `n_orig`
/// -length arrays throughout, scaled by `ear_filter_coeff` directly — which
/// is bit-identical to Python's result, not a behavioural deviation (see
/// `DEVIATIONS.md` for the deviations that *do* change output).
///
/// # A reproduced bug: `hBP[i].all() != 0`
/// See `DEVIATIONS.md` — the guard on which channel pairs get a
/// cross-correlation is reproduced exactly, including its edge-case
/// fragility (it excludes a channel pair only when *every* sample of one
/// row is exactly zero, not when the channel merely has low excitation).
pub fn roughness_dw_main_calc(
    spec: &[Complex64],
    freq_axis: &[f64],
    fs: f64,
    gzi: &[f64],
    h_weight: &[Vec<f64>],
) -> RoughnessDwResult {
    assert_eq!(spec.len(), freq_axis.len());
    let n_orig = spec.len();
    let n = 2 * n_orig;

    let bark_axis = freq2bark(freq_axis);
    let a0: Vec<f64> = ear_filter_coeff(&bark_axis)
        .iter()
        .map(|&c| db2amp(c, 1.0))
        .collect();

    let scaled: Vec<Complex64> = spec.iter().zip(&a0).map(|(&s, &a)| s * a).collect();
    let module: Vec<f64> = scaled.iter().map(|c: &Complex64| c.norm()).collect();
    let spec_db = amp2db(&module, 2e-5);

    let threshold = ltq(&bark_axis, LtqReference::Roughness);
    let audible_index: Vec<usize> = (0..n_orig).filter(|&i| spec_db[i] > threshold[i]).collect();
    let n_aud = audible_index.len();

    // ------------------------------- stage 1 ---------------------------
    let s1 = -27.0f64;
    let s2: Vec<f64> = audible_index
        .iter()
        .map(|&ind| (-24.0 - 230.0 / freq_axis[ind] + 0.2 * spec_db[ind]).min(0.0))
        .collect();

    let zi: [f64; N_CHANNEL] = channel_centres();
    // Channel centres, expressed on the same 1-indexed sample-position scale
    // `threshold` (length n_orig) is defined over, matching
    // `bark2freq(zi) * n / fs` then `interp(zb, nZ, threshold)` with
    // `nZ = 1..=n_orig`.
    let zb: Vec<f64> = crate::utils::bark2freq(&zi)
        .iter()
        .map(|&f| f * n as f64 / fs)
        .collect();
    let n_z: Vec<f64> = (1..=n_orig).map(|i| i as f64).collect();
    let min_excit_db = crate::dsp::interp(&zb, &n_z, &threshold);

    let ch_low: Vec<i64> = audible_index
        .iter()
        .map(|&ind| (2.0 * bark_axis[ind]).floor() as i64 - 1)
        .collect();
    let ch_high: Vec<i64> = audible_index
        .iter()
        .map(|&ind| (2.0 * bark_axis[ind]).ceil() as i64 - 1)
        .collect();

    // slopes[k][j]: excitation amplitude channel j gets from audible
    // component k.
    let mut slopes = vec![vec![0.0f64; N_CHANNEL]; n_aud];
    for k in 0..n_aud {
        let lev_db = spec_db[audible_index[k]];
        let b = bark_axis[audible_index[k]];
        let low_end = (ch_low[k] + 1).max(0) as usize;
        for j in 0..low_end.min(N_CHANNEL) {
            let sl = s1 * (b - (j + 1) as f64 * 0.5) + lev_db;
            if sl > min_excit_db[j] {
                slopes[k][j] = db2amp(sl, 2e-5);
            }
        }
        let high_start = ch_high[k].max(0) as usize;
        for j in high_start.min(N_CHANNEL)..N_CHANNEL {
            let sl = s2[k] * ((j + 1) as f64 * 0.5 - b) + lev_db;
            if sl > min_excit_db[j] {
                slopes[k][j] = db2amp(sl, 2e-5);
            }
        }
    }

    // ------------------------------- stage 2 ---------------------------
    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);

    let results: Vec<(Vec<f64>, f64)> = (0..N_CHANNEL)
        .into_par_iter()
        .map(|i| {
            let mut exc = vec![Complex64::new(0.0, 0.0); n];
            for k in 0..n_aud {
                let ind = audible_index[k];
                let ampl = if ch_low[k] == i as i64 || ch_high[k] == i as i64 {
                    1.0
                } else if ch_high[k] > i as i64 {
                    // Python indexes `slopes[j, i+1]` unguarded; `i+1` can
                    // only reach `N_CHANNEL` for a component beyond ~23.5
                    // Bark, which crashes the Python with an IndexError and
                    // is not reachable through this crate's public API
                    // (`comp_spectrum`'s frequency axis tops out at `fs/2`,
                    // and Nyquist for any supported `fs` sits below that).
                    // Clamped here instead of panicking.
                    slopes[k][(i + 1).min(N_CHANNEL - 1)] / module[ind]
                } else {
                    // Python's `slopes[j, i-1]` wraps to the *last* channel
                    // via numpy negative indexing when `i == 0` — reachable
                    // only when `ch_high[k] == -1`, itself only possible for
                    // a component at exactly 0 Bark, which likewise never
                    // arises through this crate's public API (`freq_axis`
                    // never contains an exact 0 Hz bin). Reproduced anyway
                    // for whatever input could reach it. See `DEVIATIONS.md`.
                    let idx = if i == 0 { N_CHANNEL - 1 } else { i - 1 };
                    slopes[k][idx] / module[ind]
                };
                exc[ind] = scaled[ind] * ampl;
            }

            // `n * real(ifft(exc))`: rustfft's inverse is unnormalised
            // (numpy's divides by `n`), so the `n *` and the `/n` cancel —
            // this is exactly `real(raw inverse fft)`, no extra scaling.
            let mut buf = exc;
            inv.process(&mut buf);
            let temporal_excitation: Vec<f64> = buf.iter().map(|c| c.re.abs()).collect();

            let h0 = temporal_excitation.iter().sum::<f64>() / n as f64;
            let mut envelope_spec: Vec<Complex64> = temporal_excitation
                .iter()
                .map(|&v| Complex64::new(v - h0, 0.0))
                .collect();
            fwd.process(&mut envelope_spec);
            for (v, &w) in envelope_spec.iter_mut().zip(&h_weight[i]) {
                *v *= w;
            }
            inv.process(&mut envelope_spec);
            // `2 * real(ifft(envelope_spec))`, with the `/n` normalisation
            // that `n *` cancelled above now applying in full.
            let scale = 2.0 / n as f64;
            let h_bp: Vec<f64> = envelope_spec.iter().map(|c| scale * c.re).collect();

            let h_bp_rms = (h_bp.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
            let mod_depth = if h0 > 0.0 {
                (h_bp_rms / h0).min(1.0)
            } else {
                0.0
            };

            (h_bp, mod_depth)
        })
        .collect();

    let h_bp: Vec<Vec<f64>> = results.iter().map(|(h, _)| h.clone()).collect();
    let mod_depth: Vec<f64> = results.iter().map(|(_, m)| *m).collect();

    // ------------------------------- stage 3 ---------------------------
    let mut ki = [0.0f64; N_CHANNEL];
    for i in 0..45 {
        if h_bp[i].iter().all(|&v| v != 0.0) && h_bp[i + 2].iter().all(|&v| v != 0.0) {
            ki[i] = pearson_corrcoef(&h_bp[i], &h_bp[i + 2]);
        }
    }

    let mut r_spec = [0.0f64; N_CHANNEL];
    r_spec[0] = gzi[0] * (mod_depth[0] * ki[0]).powi(2);
    r_spec[1] = gzi[1] * (mod_depth[1] * ki[1]).powi(2);
    for i in 2..45 {
        r_spec[i] = gzi[i] * (mod_depth[i] * ki[i] * ki[i - 2]).powi(2);
    }
    r_spec[45] = gzi[45] * (mod_depth[45] * ki[43]).powi(2);
    r_spec[46] = gzi[46] * (mod_depth[46] * ki[44]).powi(2);

    let r: f64 = 0.25 * r_spec.iter().sum::<f64>();

    (r, r_spec, zi)
}

/// Pearson correlation coefficient, matching `numpy.corrcoef(x, y)[0, 1]`.
fn pearson_corrcoef(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    cov / (var_x * var_y).sqrt()
}
