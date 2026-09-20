//! Top-level Daniel & Weber roughness entry points, matching
//! `roughness_dw.py`/`roughness_dw_freq.py`.

use ndarray::Array2;
use num_complex::Complex64;
use rayon::prelude::*;

use super::gzi_weighting::gzi_weighting;
use super::h_weighting::h_weighting;
use super::main_calc::{
    channel_centres, roughness_dw_main_calc, roughness_dw_main_calc_with_setup, RoughnessDwSetup,
};
use crate::slm::{comp_spectrum_complex, SpectrumWindow};
use crate::utils::time_segmentation;

const N_CHANNEL: usize = 47;

/// `(R, R_spec, bark_axis, time_axis)` — [`roughness_dw`]'s result.
type RoughnessDwSegResult = (Vec<f64>, Array2<f64>, [f64; N_CHANNEL], Vec<f64>);

/// Computes Daniel & Weber roughness from a time signal, matching
/// `roughness_dw(signal, fs, overlap)`.
///
/// The signal is segmented into 200 ms blocks (matching MoSQITo's fixed
/// time resolution for this method), each block's roughness computed
/// independently and in parallel — mirroring `roughness_ecma`'s and
/// `loudness_zwst_perseg`'s use of rayon across segments.
///
/// # Panics
/// Panics if `signal` is shorter than one 200 ms block at `fs`.
pub fn roughness_dw(signal: &[f64], fs: f64, overlap: f64) -> RoughnessDwSegResult {
    let nperseg = (0.2 * fs) as usize;
    let noverlap = (overlap * nperseg as f64) as usize;
    let (blocks, time) = time_segmentation(signal, fs, nperseg, Some(noverlap));
    let nseg = blocks.ncols();

    let (spec, freq_axis) = comp_spectrum_complex(blocks.view(), fs, SpectrumWindow::Blackman);

    let h_weight = h_weighting(nperseg, fs);
    let zi = channel_centres();
    let gzi = gzi_weighting(&zi);
    // Every segment shares this signal's freq_axis/fs/block length, so the
    // ear-filter/threshold tables and FFT plans are identical across all
    // `nseg` calls below — built once here instead of once per segment.
    // See `RoughnessDwSetup`'s own docs.
    let setup = RoughnessDwSetup::new(&freq_axis, fs);

    let results: Vec<(f64, [f64; N_CHANNEL], [f64; N_CHANNEL])> = (0..nseg)
        .into_par_iter()
        .map(|col| {
            let spec_col: Vec<Complex64> = spec.column(col).to_vec();
            roughness_dw_main_calc_with_setup(&setup, &spec_col, &freq_axis, &gzi, &h_weight)
        })
        .collect();

    let r: Vec<f64> = results.iter().map(|&(r, _, _)| r).collect();
    let mut r_spec = Array2::<f64>::zeros((N_CHANNEL, nseg));
    for (col, &(_, spec_col, _)) in results.iter().enumerate() {
        for (z, &v) in spec_col.iter().enumerate() {
            r_spec[[z, col]] = v;
        }
    }
    let bark_axis = results
        .first()
        .map(|&(_, _, b)| b)
        .unwrap_or_else(channel_centres);

    (r, r_spec, bark_axis, time)
}

/// Computes Daniel & Weber roughness from a fine-band amplitude spectrum,
/// matching `roughness_dw_freq(spectrum, freqs)` for a 1-D spectrum (the
/// 2-D per-segment case is not ported — no caller in this crate needs it;
/// same scope-narrowing as `loudness_zwst_freq`/`noct_synthesis`).
///
/// `spectrum` must be a non-negative amplitude spectrum (not complex — use
/// `.norm()` first if starting from a complex FFT output).
///
/// # Panics
/// Panics if `spectrum` and `freqs` differ in length, or if `freqs` has
/// fewer than 2 points (needed to recover the implied sampling rate).
pub fn roughness_dw_freq(
    spectrum: &[f64],
    freqs: &[f64],
) -> (f64, [f64; N_CHANNEL], [f64; N_CHANNEL]) {
    assert_eq!(
        spectrum.len(),
        freqs.len(),
        "spectrum and freqs must have the same length"
    );
    assert!(freqs.len() >= 2, "freqs must have at least 2 points");

    let nperseg = spectrum.len();
    let df: f64 = freqs.windows(2).map(|w| w[1] - w[0]).sum::<f64>() / (freqs.len() - 1) as f64;
    let fs = (2.0 * nperseg as f64 * df).trunc();

    let h_weight = h_weighting(2 * nperseg, fs);
    let zi = channel_centres();
    let gzi = gzi_weighting(&zi);

    let spec_complex: Vec<Complex64> = spectrum.iter().map(|&s| Complex64::new(s, 0.0)).collect();
    roughness_dw_main_calc(&spec_complex, freqs, fs, &gzi, &h_weight)
}
