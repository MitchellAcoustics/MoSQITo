//! Aures's modulation-depth weighting function `g(z)`, matching
//! `_gzi_weighting.py`.

use crate::dsp::interp;

const GR_Y: [f64; 25] = [
    0.15, 0.26, 0.38, 0.47, 0.54, 0.65, 0.76, 0.83, 0.90, 0.98, 0.98, 0.90, 0.80, 0.70, 0.62, 0.54,
    0.49, 0.43, 0.39, 0.35, 0.30, 0.30, 0.30, 0.30, 0.30,
];

/// `g(z)` at each Bark value in `center_freq` (despite the name, this is a
/// Bark-axis lookup — `_gzi_weighting.py`'s own parameter name).
pub fn gzi_weighting(center_freq: &[f64]) -> Vec<f64> {
    let gr_x: Vec<f64> = (0..25).map(|k| k as f64).collect();
    interp(center_freq, &gr_x, &GR_Y)
}
