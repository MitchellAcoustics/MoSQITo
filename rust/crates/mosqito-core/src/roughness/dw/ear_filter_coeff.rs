//! Zwicker's outer/inner ear transmission coefficient `a0(z)`, per E.
//! Zwicker, H. Fastl: *Psychoacoustics* (figure 8.18), matching
//! `_ear_filter_coeff.py`.

use crate::dsp::interp;

const XP: [f64; 22] = [
    0.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 16.5, 17.0, 18.0, 18.5, 19.0, 20.0, 21.0, 21.5, 22.0,
    22.5, 23.0, 23.5, 24.0, 25.0, 26.0,
];
const YP: [f64; 22] = [
    0.0, 0.0, 1.15, 2.31, 3.85, 5.62, 6.92, 7.38, 6.92, 4.23, 2.31, 0.0, -1.43, -2.59, -3.57,
    -5.19, -7.41, -11.3, -20.0, -40.0, -130.0, -999.0,
];

/// `a0(z)` over `bark_axis`.
pub fn ear_filter_coeff(bark_axis: &[f64]) -> Vec<f64> {
    interp(bark_axis, &XP, &YP)
}
