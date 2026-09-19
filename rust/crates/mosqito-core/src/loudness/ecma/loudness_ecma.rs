//! Top-level ECMA-418-2 (2nd Ed, 2022) §5 loudness entry point, matching
//! `loudness_ecma.py`.

use ndarray::Array2;
use rayon::prelude::*;

use super::band_pass_signals::band_pass_signals;
use super::preprocessing::preprocess;
use super::specific_loudness::specific_loudness_for_band;
use super::tables::{bark_axis_53, LTQ_Z};
use crate::dsp::resample_to;

const FS_ECMA: f64 = 48000.0;

/// `(N, N_time, N_specific, bark_axis, time_axis)` — [`loudness_ecma`]'s
/// result. `N_specific` is (53, `n_blocks`); every band shares the same
/// `time_axis` since only a single scalar `sb`/`sh` is supported (see
/// `specific_loudness.rs`'s module doc).
type LoudnessEcmaResult = (f64, Vec<f64>, Array2<f64>, [f64; 53], Vec<f64>);

/// Computes the specific and total loudness per ECMA-418-2 (2nd Ed, 2022)
/// §5, for stationary signals. Matches `loudness_ecma(signal, fs, sb, sh)`
/// for scalar `sb`/`sh` (the per-band list case is not ported).
///
/// Resamples to 48 kHz first if `fs != 48000` (`loudness_ecma.py:93-100`).
pub fn loudness_ecma(signal: &[f64], fs: f64, sb: usize, sh: usize) -> LoudnessEcmaResult {
    let signal = resample_to(signal, fs, FS_ECMA);

    let (padded, n_new) = preprocess(&signal, sb, sh);
    let bandpass = band_pass_signals(&padded, FS_ECMA);

    let per_band: Vec<(Vec<f64>, Vec<f64>)> = (0..53)
        .into_par_iter()
        .map(|band| specific_loudness_for_band(&bandpass[band], sb, sh, n_new, LTQ_Z[band]))
        .collect();

    let n_blocks = per_band[0].0.len();
    let mut n_specific = Array2::<f64>::zeros((53, n_blocks));
    for (band, (spec, _time)) in per_band.iter().enumerate() {
        for (t, &v) in spec.iter().enumerate() {
            n_specific[[band, t]] = v;
        }
    }
    let time_axis = per_band[0].1.clone();

    // Time-dependent total loudness (Eq. 116).
    const DELTA_Z: f64 = 0.5;
    let n_time: Vec<f64> = (0..n_blocks)
        .map(|t| (0..53).map(|band| n_specific[[band, t]]).sum::<f64>() * DELTA_Z)
        .collect();

    // Single-value loudness (Eq. 117): power mean with exponent 1/log10(2).
    let exponent = 1.0 / (2.0f64).log10();
    let mean_pow = n_time.iter().map(|&v| v.powf(exponent)).sum::<f64>() / n_blocks as f64;
    let n = mean_pow.powf(1.0 / exponent);

    (n, n_time, n_specific, bark_axis_53(), time_axis)
}
