//! First-order low-pass filtering with linear interpolation for increased
//! precision. Matches `_lowpass_intp.py`.

use crate::dsp::lfilter;

const LP_ITER: usize = 24;

/// Filters `loudness` (a 1-D time series) with a first-order low-pass of
/// time constant `tau`, internally upsampled `LP_ITER`x by linear
/// interpolation between samples for precision, then downsampled back.
///
/// Unlike [`super::nonlinear_decay::nl_loudness`], this filter's initial
/// state is a plain zero (`scipy.signal.lfilter`'s default `zi`), so there
/// is no wraparound subtlety here.
pub fn lowpass_intp(loudness: &[f64], tau: f64, sample_rate: f64) -> Vec<f64> {
    let n = loudness.len();
    if n == 0 {
        return Vec::new();
    }

    let a1 = (-1.0 / (sample_rate * LP_ITER as f64 * tau)).exp();
    let b0 = 1.0 - a1;

    let mut ui = vec![0.0f64; n * LP_ITER];
    for t in 0..n {
        let next = if t + 1 < n { loudness[t + 1] } else { 0.0 };
        let delta = (next - loudness[t]) / LP_ITER as f64;
        for i_in in 0..LP_ITER {
            ui[t * LP_ITER + i_in] = loudness[t] + i_in as f64 * delta;
        }
    }

    let filtered = lfilter(&[b0], &[1.0, -a1], &ui);
    (0..n).map(|t| filtered[t * LP_ITER]).collect()
}
