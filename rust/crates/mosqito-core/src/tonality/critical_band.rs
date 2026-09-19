//! ECMA-74 Annex D.8/D.10 critical-band-edge formulas, shared by TNR and PR.

/// The critical band centred on `f0`, per ECMA-74 Annex D.8.
pub fn critical_band(f0: f64) -> (f64, f64) {
    let delta_fc = 25.0 + 75.0 * (1.0 + 1.4 * (f0 / 1000.0).powi(2)).powf(0.69);
    if f0 < 500.0 {
        (f0 - delta_fc / 2.0, f0 + delta_fc / 2.0)
    } else {
        let f1 = -delta_fc / 2.0 + (delta_fc * delta_fc + 4.0 * f0 * f0).sqrt() / 2.0;
        (f1, f1 + delta_fc)
    }
}

/// The critical band immediately below and contiguous with the one centred
/// on `f0`, per ECMA-74 Annex D.10.
pub fn lower_critical_band(f0: f64) -> (f64, f64) {
    let (f2, _) = critical_band(f0);
    let (c0, c1, c2) = if f0 < 171.4 {
        (20.0, 0.0, 0.0)
    } else if f0 <= 1600.0 {
        (-149.5, 1.001, -6.9e-05)
    } else {
        (6.8, 0.806, -8.2e-06)
    };
    let f1 = c0 + c1 * f0 + c2 * f0 * f0;
    (f1, f2)
}

/// The critical band immediately above and contiguous with the one centred
/// on `f0`, per ECMA-74 Annex D.10.
pub fn upper_critical_band(f0: f64) -> (f64, f64) {
    let (_, f1) = critical_band(f0);
    let (c0, c1, c2) = if f0 <= 1600.0 {
        (149.5, 1.035, 7.7e-05)
    } else {
        (3.3, 1.215, 2.16e-05)
    };
    let f2 = c0 + c1 * f0 + c2 * f0 * f0;
    (f1, f2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_band_widens_with_frequency() {
        let (f1_lo, f2_lo) = critical_band(200.0);
        let (f1_hi, f2_hi) = critical_band(4000.0);
        assert!(f2_lo - f1_lo < f2_hi - f1_hi);
        assert!(f1_lo < 200.0 && f2_lo > 200.0);
        assert!(f1_hi < 4000.0 && f2_hi > 4000.0);
    }

    #[test]
    fn lower_and_upper_bands_are_contiguous_with_the_central_one() {
        let f0 = 1000.0;
        let (cf1, cf2) = critical_band(f0);
        let (_, lf2) = lower_critical_band(f0);
        let (uf1, _) = upper_critical_band(f0);
        assert!((lf2 - cf1).abs() < 1e-9);
        assert!((uf1 - cf2).abs() < 1e-9);
    }
}
