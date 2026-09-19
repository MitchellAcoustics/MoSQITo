//! ECMA-418-2 (2nd Ed, 2022) §7.1.5.2 modulation-rate weighting: the
//! per-band constants (Eqs. 84, 86, 87, 96) and the high/low modulation-rate
//! weighting functions (Eqs. 83, 95). Matches `_weighting.py`.

/// Eq. 86: the modulation rate at which the high-rate weighting reaches 1,
/// per band.
pub fn f_max(centre_freq: f64) -> f64 {
    72.6937 * (1.0 - 1.1739 * (-5.4583 * centre_freq / 1000.0).exp())
}

/// Eq. 84: the per-band scaling factor.
pub fn r_max(centre_freq: f64) -> f64 {
    let (r1, r2) = if centre_freq < 1000.0 {
        (0.3560, 0.8049)
    } else {
        (0.8024, 0.9333)
    };
    1.0 / (1.0 + r1 * (centre_freq / 1000.0).log2().abs().powf(r2))
}

/// Eq. 87: the high-modulation-rate weighting shape parameter.
pub fn q2_high(centre_freq: f64) -> f64 {
    let ratio = centre_freq / 1000.0;
    if ratio < 2f64.powf(-3.4253) {
        0.2471
    } else {
        0.2471 + 0.0129 * (ratio.log2() + 3.4253).powi(2)
    }
}

/// Eq. 96: the low-modulation-rate weighting shape parameter.
pub fn q2_low(centre_freq: f64) -> f64 {
    1.0967 - 0.0640 * (centre_freq / 1000.0).log2()
}

/// Eq. 83: weights a single peak's amplitude for high modulation rates.
pub fn high_mod_rate_weighting(mod_rate: f64, amp: f64, fmax: f64, rmax: f64, q2_high: f64) -> f64 {
    if mod_rate < fmax {
        amp * rmax
    } else {
        let g = 1.0 / (1.0 + ((mod_rate / fmax - fmax / mod_rate) * 1.2822).powi(2)).powf(q2_high);
        g * amp * rmax
    }
}

/// Eq. 95: weights (and sums) the fundamental modulation rate's harmonic
/// complex amplitudes for low modulation rates.
pub fn low_mod_rate_weighting(mod_rate: f64, amp: &[f64], fmax: f64, q2_low: f64) -> f64 {
    if mod_rate < fmax {
        let g = 1.0 / (1.0 + ((mod_rate / fmax - fmax / mod_rate) * 0.7066).powi(2)).powf(q2_low);
        amp.iter().map(|&a| g * a).sum()
    } else {
        amp.iter().sum()
    }
}
