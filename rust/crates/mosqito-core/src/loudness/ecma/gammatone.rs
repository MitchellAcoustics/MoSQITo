//! ECMA-418-2 (2nd Ed, 2022) §5.1.4 gammatone filter design, Eqs. 8–17.
//!
//! Fixed at filter order `k = 5` (the only order ECMA-418-2 specifies, and
//! the only one MoSQITo's own `_band_pass_signals.py:54` ever passes), so
//! the `k`-dependent binomial coefficients are baked in as constants rather
//! than computed generically.

use num_complex::Complex64;

const AF_F0: f64 = 81.9289;
const C: f64 = 0.1618;

/// `comb(2*5-2, 5-1) = comb(8, 4)`, Eq. 8's binomial coefficient at `k = 5`.
const BINOM_2K_2_K_1: f64 = 70.0;
/// `comb(5, m)` for `m = 1..=5`, Eq. 14.
const COMB_5_M: [f64; 5] = [5.0, 10.0, 10.0, 5.0, 1.0];
/// The `e_m` constants in Eq. 15, for `m = 0..=4`.
const EM: [f64; 5] = [0.0, 1.0, 11.0, 11.0, 1.0];

/// Computes the complex gammatone filter coefficients `(bm_prim, am_prim)`
/// for a band centred at `freq` Hz, matching `_gammatone.py`.
///
/// `am_prim` has 6 taps (denominator, order 5); `bm_prim` has 5 taps
/// (numerator).
pub fn gammatone(freq: f64, fs: f64) -> ([Complex64; 5], [Complex64; 6]) {
    // Bandwidth (Eq. 10) and time constant (Eq. 8).
    let delta_f = (AF_F0 * AF_F0 + (C * freq).powi(2)).sqrt();
    let tau = (1.0 / 512.0) * BINOM_2K_2_K_1 / delta_f;
    let d = (-1.0 / (fs * tau)).exp();

    // am (Eq. 14): am[0] = 1, am[m] = (-d)^m * C(5, m) for m = 1..=5.
    let mut am = [0.0f64; 6];
    am[0] = 1.0;
    for m in 1..=5 {
        am[m] = (-d).powi(m as i32) * COMB_5_M[m - 1];
    }

    // bm (Eq. 15): bm[m] = (1-d)^5 / sum(e_i * d^i, i=1..4) * d^m * e_m.
    let denom: f64 = (1..=4).map(|i| EM[i] * d.powi(i as i32)).sum();
    let mut bm = [0.0f64; 5];
    for m in 0..5 {
        bm[m] = (1.0 - d).powi(5) / denom * d.powi(m as i32) * EM[m];
    }

    // Modulation onto the band centre frequency (Eqs. 16-17).
    let mut exponential = [Complex64::new(0.0, 0.0); 6];
    for (m, e) in exponential.iter_mut().enumerate() {
        let theta = 2.0 * std::f64::consts::PI * freq * m as f64 / fs;
        *e = Complex64::new(theta.cos(), theta.sin());
    }

    let mut am_prim = [Complex64::new(0.0, 0.0); 6];
    for m in 0..6 {
        am_prim[m] = Complex64::new(am[m], 0.0) * exponential[m];
    }
    let mut bm_prim = [Complex64::new(0.0, 0.0); 5];
    for m in 0..5 {
        bm_prim[m] = Complex64::new(bm[m], 0.0) * exponential[m];
    }

    (bm_prim, am_prim)
}
