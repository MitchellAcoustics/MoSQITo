//! Low-frequency bandpass weighting functions `H_i`, per Daniel & Weber
//! ("Psychoacoustical roughness: implementation of an optimized model",
//! 1997), matching `_H_weighting.py`.
//!
//! `H2`, `H16`(sic — the code calls it `H5` internally) and `H42` are given
//! (figure 2); the other 44 of the 47 one-Bark-wide channels reuse one of
//! these three curves (or a fourth, `H21`, likewise given directly):
//! - channels 1-4 (`H[0..4]`) share `H2`
//! - channels 5-15 (`H[4..15]`) share `H5`
//! - channels 16-20 (`H[15..20]`) share `H16`
//! - channels 21-41 (`H[20..41]`) share `H21`
//! - channels 42-47 (`H[41..47]`) share `H42`
//!
//! # A reproduced quirk: three curves share one truncated frequency range
//! Python computes the highest bin index (`last`) to fill from each curve's
//! own top x-value for `H2` (358 Hz) and `H5` (502 Hz) — but then *reuses*
//! `H5`'s `last` (502 Hz) for `H16`, `H21` and `H42` too, instead of
//! recomputing it from their own top x-values (each 645 Hz). Every bin past
//! `502 Hz`'s index in those three curves is therefore left at zero, even
//! though their tables define values out to 645 Hz. There is no isolated
//! reference for `_H_weighting` alone to confirm whether this is
//! intentional or a transcription slip — only the end-to-end `roughness_dw`
//! output is validated (against Zwicker & Fastl's and Daniel & Weber's own
//! figures) — so it is reproduced exactly rather than "corrected" against a
//! guess. See `DEVIATIONS.md`.

use crate::dsp::interp;

const H2_X: [f64; 15] = [
    0.0, 17.0, 23.0, 25.0, 32.0, 37.0, 48.0, 67.0, 90.0, 114.0, 171.0, 206.0, 247.0, 294.0, 358.0,
];
const H2_Y: [f64; 15] = [
    0.0, 0.8, 0.95, 0.975, 1.0, 0.975, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0,
];

const H5_X: [f64; 14] = [
    0.0, 32.0, 43.0, 56.0, 69.0, 92.0, 120.0, 142.0, 165.0, 231.0, 277.0, 331.0, 397.0, 502.0,
];
const H5_Y: [f64; 14] = [
    0.0, 0.8, 0.95, 1.0, 0.975, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.0,
];

const H16_X: [f64; 20] = [
    0.0, 23.5, 34.0, 47.0, 56.0, 63.0, 79.0, 100.0, 115.0, 135.0, 159.0, 172.0, 194.0, 215.0,
    244.0, 290.0, 348.0, 415.0, 500.0, 645.0,
];
const H16_Y: [f64; 20] = [
    0.0, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 0.975, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2,
    0.1, 0.0,
];

const H21_X: [f64; 18] = [
    0.0, 19.0, 44.0, 52.5, 58.0, 75.0, 101.5, 114.5, 132.5, 143.5, 165.5, 197.5, 241.0, 290.0,
    348.0, 415.0, 500.0, 645.0,
];
const H21_Y: [f64; 18] = [
    0.0, 0.4, 0.8, 0.9, 0.95, 1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
];

const H42_X: [f64; 19] = [
    0.0, 15.0, 41.0, 49.0, 53.0, 64.0, 71.0, 88.0, 94.0, 106.0, 115.0, 137.0, 180.0, 238.0, 290.0,
    348.0, 415.0, 500.0, 645.0,
];
const H42_Y: [f64; 19] = [
    0.0, 0.4, 0.8, 0.9, 0.965, 0.99, 1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
    0.0,
];

fn fill_row(row: &mut [f64], cut: usize, last: usize, fs: f64, n: usize, x: &[f64], y: &[f64]) {
    for (j, v) in row.iter_mut().enumerate().take(last).skip(cut) {
        let freq = j as f64 * fs / n as f64;
        *v = interp(&[freq], x, y)[0];
    }
}

/// The 47 weighting functions `H_i`, each of length `n` (`n` is the full
/// symmetric spectrum size `_roughness_dw_main_calc` works with).
pub fn h_weighting(n: usize, fs: f64) -> Vec<Vec<f64>> {
    let cut = 2usize;
    let mut h = vec![vec![0.0f64; n]; 47];

    let last_358 = ((358.0 / fs) * n as f64).floor() as usize;
    fill_row(&mut h[1], cut, last_358, fs, n, &H2_X, &H2_Y);

    let last_502 = ((502.0 / fs) * n as f64).floor() as usize;
    fill_row(&mut h[4], cut, last_502, fs, n, &H5_X, &H5_Y);
    fill_row(&mut h[15], cut, last_502, fs, n, &H16_X, &H16_Y);
    fill_row(&mut h[20], cut, last_502, fs, n, &H21_X, &H21_Y);
    fill_row(&mut h[41], cut, last_502, fs, n, &H42_X, &H42_Y);

    h[0] = h[1].clone();
    h[2] = h[1].clone();
    h[3] = h[1].clone();
    for i in 5..15 {
        h[i] = h[4].clone();
    }
    for i in 16..20 {
        h[i] = h[15].clone();
    }
    for i in 21..41 {
        h[i] = h[20].clone();
    }
    for i in 42..47 {
        h[i] = h[41].clone();
    }

    h
}
