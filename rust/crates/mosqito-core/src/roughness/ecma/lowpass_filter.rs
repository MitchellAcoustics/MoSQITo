//! ECMA-418-2 (2nd Ed, 2022) §7.1.7 Eqs. 109-110: first-order lowpass
//! filtering of the specific roughness over time, with different time
//! constants for rising (0.0625 s) and falling (0.5 s) slopes.
//!
//! # An unsanctioned bug, fixed (not reproduced) — `c_R` re-derived
//!
//! MoSQITo's `_lowpass_filter.py` computes its rising/falling
//! classification, time constants and the whole filter recursion along
//! `axis=-1` — across **critical bands**, not time — then discards that
//! entire computation (`R_spec`, never returned) and instead returns a
//! second, differently broken pass: `R_time_spec[1:, :] = R_hat[1, :] * (1 -
//! exp(-1/(50*tau[1, :]))) + R_hat[:-1, :] * exp(-1/(50*tau[1, :]))`, which
//! uses the rising/falling classification and raw (never previously
//! filtered) value **from time index 1 only**, broadcast across every
//! subsequent time step, rather than a true recursive filter carrying its
//! own previous output forward. Neither of these matches ECMA-418-2 Eqs.
//! 109-110 (a first-order lowpass *over time*, with rising/falling decided
//! and `tau` selected fresh at each time step from that step's actual
//! `R_hat` vs. the filter's own previous output) — this is not something the
//! INTERNOISE 2024 paper mentions or sanctions, unlike the roughness
//! deviations that follow it (D-ecma-1 through D-ecma-4). See `DEVIATIONS.md`.
//!
//! This implements Eq. 109/110 as published: a genuine per-band recursive
//! filter over time. Because the fix changes every downstream value, the
//! calibration factor `c_R` in `non_linear_transform.rs` could not be kept
//! at MoSQITo's `0.045` (fitted against the buggy filter) — it was re-fit
//! against the ECMA-418-2 Annex C reference values instead. See
//! `DEVIATIONS.md` for the fit residuals.

/// Applies the corrected Eq. 109/110 lowpass, per critical band
/// independently, over the time axis (rows).
///
/// `r_hat` is (`n_blocks_50`, 53); returns the same shape.
pub fn lowpass_filter(r_hat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if r_hat.is_empty() {
        return Vec::new();
    }
    let cbf = r_hat[0].len();
    let mut out = Vec::with_capacity(r_hat.len());

    let mut prev = r_hat[0].clone();
    out.push(prev.clone());

    for row in &r_hat[1..] {
        let mut cur = vec![0.0f64; cbf];
        for z in 0..cbf {
            let rising = row[z] >= prev[z];
            let tau: f64 = if rising { 0.0625 } else { 0.5 };
            let e = (-1.0 / (50.0 * tau)).exp();
            cur[z] = row[z] * (1.0 - e) + prev[z] * e;
        }
        out.push(cur.clone());
        prev = cur;
    }

    out
}
