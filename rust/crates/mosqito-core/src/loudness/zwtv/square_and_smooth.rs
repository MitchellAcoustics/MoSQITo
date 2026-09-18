//! Squaring plus a cascade of three first-order smoothing low-pass filters,
//! per ISO 532-1 section 6.3. Matches `_square_and_smooth.py`.

use crate::dsp::lfilter;

/// Squares `sig`, then applies the frequency-dependent smoothing filter
/// three times in cascade.
pub fn square_and_smooth(sig: &[f64], center_freq: f64, fs: f64) -> Vec<f64> {
    let tau = if center_freq <= 1000.0 {
        2.0 / (3.0 * center_freq)
    } else {
        2.0 / (3.0 * 1000.0)
    };

    let mut sig: Vec<f64> = sig.iter().map(|&s| s * s).collect();

    let a1 = (-1.0 / (fs * tau)).exp();
    let b0 = 1.0 - a1;
    for _ in 0..3 {
        sig = lfilter(&[b0], &[1.0, -a1], &sig);
    }
    sig
}
