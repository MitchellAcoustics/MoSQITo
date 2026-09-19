//! ECMA-418-2 (2nd Ed, 2022) §7.1.7 non-linear transform, ~Eq. 105-108, with
//! the correction from Wanty, Glesser & Casagrande Hirono (D-ecma-4 in
//! `DEVIATIONS.md`): the standard's published Eq. 105 fitting curve is
//! wrong, so this uses the Sottek/Becker/Lobato (INTERNOISE 2020) form
//! instead, matching MoSQITo's own `_non_linear_transform.py`.
//!
//! The calibration factor `c_R` (D-ecma-5) is *not* MoSQITo's `0.045`: that
//! value was fitted against the buggy `_lowpass_filter.py` this port does
//! not reproduce (see `lowpass_filter.rs`'s module doc and `DEVIATIONS.md`).
//! With the corrected lowpass filter, `0.045` overshoots the ECMA-418-2
//! Annex C reference values by roughly 35% on average; `c_R` was re-fit
//! against that same reference corpus instead — see `lowpass_filter.rs`.

/// Applies the non-linear transform to a (`n_blocks_50`, 53) estimated
/// roughness array, returning the calibrated specific roughness before
/// temporal lowpass filtering.
pub fn non_linear_transform(r_est: &[Vec<f64>], c_r: f64) -> Vec<Vec<f64>> {
    let cbf = if r_est.is_empty() { 0 } else { r_est[0].len() };

    r_est
        .iter()
        .map(|row| {
            // Eq. 107/108: RMS-to-linear-mean ratio across bands.
            let sq_mean = (row.iter().map(|&v| v * v).sum::<f64>() / cbf as f64).sqrt();
            let lin_mean = row.iter().sum::<f64>() / cbf as f64;
            let b = if lin_mean != 0.0 {
                sq_mean / lin_mean
            } else {
                0.0
            };

            // Sottek/Becker/Lobato (INTERNOISE 2020) form, replacing the
            // standard's Eq. 105.
            let e = 0.25 * (1.75 * (b - 2.5)).tanh() + 0.7;

            row.iter().map(|&v| c_r * v.powf(e)).collect()
        })
        .collect()
}
