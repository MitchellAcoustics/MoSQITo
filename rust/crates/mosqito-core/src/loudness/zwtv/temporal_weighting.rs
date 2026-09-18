//! Temporal weighting of total loudness: two first-order low-pass filters
//! (3.5 ms and 70 ms time constants) combined to simulate the
//! duration-dependent behaviour of loudness perception for short impulses.
//! Matches `_temporal_weighting.py`.

use super::lowpass_intp::lowpass_intp;

const SAMPLE_RATE: f64 = 2000.0;

pub fn temporal_weighting(loudness: &[f64]) -> Vec<f64> {
    let filt_1 = lowpass_intp(loudness, 3.5e-3, SAMPLE_RATE);
    let filt_2 = lowpass_intp(loudness, 70e-3, SAMPLE_RATE);
    filt_1
        .iter()
        .zip(&filt_2)
        .map(|(&a, &b)| 0.47 * a + 0.53 * b)
        .collect()
}
