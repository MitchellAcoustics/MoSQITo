//! ECMA-418-2 (2nd Ed, 2022) §7.1.4: noise-reduction weighting of the
//! envelope scaled power spectra. Matches `_noise_reduction.py`.

use rayon::prelude::*;

use crate::dsp::median;

/// Applies §7.1.4's noise-reduction weighting to `spectrum`, indexed
/// `[time][band][bin]` (`Ntime x 53 x sbb/2`).
///
/// Each time block is independent of every other (no cross-`t` coupling), so
/// this runs in parallel across `t`, matching how the rest of the
/// `roughness_ecma` pipeline is parallelized across its own independent axis.
pub fn noise_reduction(spectrum: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
    let l = spectrum.len();
    if l == 0 {
        return Vec::new();
    }
    let n_bands = spectrum[0].len();
    let k = spectrum[0][0].len();

    (0..l)
        .into_par_iter()
        .map(|t| {
            // Averaging with neighbouring bands.
            let mut spectrum_average = vec![vec![0.0f64; k]; n_bands];
            for z in 0..n_bands {
                for bin in 0..k {
                    spectrum_average[z][bin] = if z == 0 {
                        (spectrum[t][0][bin] + spectrum[t][1][bin]) / 2.0
                    } else if z == n_bands - 1 {
                        (spectrum[t][n_bands - 1][bin] + spectrum[t][n_bands - 2][bin]) / 2.0
                    } else {
                        (spectrum[t][z - 1][bin] + spectrum[t][z][bin] + spectrum[t][z + 1][bin])
                            / 3.0
                    };
                }
            }

            // Sum across bands, then median of that sum over bins 2.. (the
            // Python indexes `s[:, 2:]`, i.e. excludes the first two bins).
            let s: Vec<f64> = (0..k)
                .map(|bin| (0..n_bands).map(|z| spectrum_average[z][bin]).sum::<f64>())
                .collect();
            let s_tilde = median(&s[2..]);

            let w_tilde: Vec<f64> = (0..k)
                .map(|bin| {
                    let clip = (0.1891 * (0.0120 * bin as f64).exp()).clamp(0.0, 1.0);
                    // Preserved verbatim from `_noise_reduction.py:37`: `10e-10`
                    // (= 1e-9), likely intended as `1e-10` — see `DEVIATIONS.md`.
                    0.0856 * (s[bin] / (s_tilde + 10e-10)) * clip
                })
                .collect();
            let w_tilde_max = w_tilde[2..]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

            let mut noise_suppression_weighting = vec![0.0f64; k];
            for bin in 0..k {
                if w_tilde[bin] >= 0.05 * w_tilde_max {
                    noise_suppression_weighting[bin] = (w_tilde[bin] - 0.1407).clamp(0.0, 1.0);
                }
            }

            let mut phi_e_t = vec![vec![0.0f64; k]; n_bands];
            for z in 0..n_bands {
                for bin in 0..k {
                    phi_e_t[z][bin] = spectrum_average[z][bin] * noise_suppression_weighting[bin];
                }
            }
            phi_e_t
        })
        .collect()
}
