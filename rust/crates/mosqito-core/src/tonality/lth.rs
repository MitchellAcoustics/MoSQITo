//! ECMA-74 Annex D.7.1 threshold of hearing, used by the tonal-candidate
//! screening (distinct from `utils::LTQ`'s interpolated Zwicker/roughness
//! thresholds, which come from a different table).

/// Threshold of hearing at each frequency in `freqs`, per ECMA-74 Annex
/// D.7.1.
///
/// Valid for `f` in `[20, 22050)` Hz. Every real caller (`_tnr_main_calc`/
/// `_pr_main_calc`'s own "frequency of interest" filter) only ever supplies
/// frequencies already restricted to `(89.1, 11200)` Hz, so the `f < 20`
/// case Python leaves undefined (an `UnboundLocalError`, since none of its
/// `if`/`elif` branches match and the coefficients are never bound) is
/// unreachable here too — this panics instead of silently returning
/// garbage. See `DEVIATIONS.md`.
pub fn lth(freqs: &[f64]) -> Vec<f64> {
    freqs.iter().map(|&f| lth_one(f)).collect()
}

fn lth_one(f: f64) -> f64 {
    let (fmean, fstd, a1, a2, a3, a4, a5) = if (20.0..305.0).contains(&f) {
        (
            167.5, 87.3212, 1.415532, -2.451068, 1.498869, -6.983224, 8.621226,
        )
    } else if (305.0..2230.0).contains(&f) {
        (
            1157.5, 488.582, 0.397994, -0.891839, -0.815138, -1.221319, -7.600754,
        )
    } else if (2230.0..14000.0).contains(&f) {
        (
            7250.0, 3033.25, 1.584978, -2.766599, -6.9061912, 10.138553, -3.149339,
        )
    } else if (14000.0..22050.0).contains(&f) {
        (
            16990.0,
            4049.0,
            -5.775593,
            -9.200034,
            26.59115,
            52.16712,
            15.61552048,
        )
    } else {
        panic!("lth: frequency {f} Hz outside ECMA-74 Annex D.7.1's defined range [20, 22050)");
    };
    let ff = (f - fmean) / fstd;
    a1 * ff.powi(4) + a2 * ff.powi(3) + a3 * ff.powi(2) + a4 * ff + a5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_is_continuous_across_band_edges() {
        for edge in [305.0, 2230.0, 14000.0] {
            let below = lth_one(edge - 1e-6);
            let at = lth_one(edge);
            assert!((below - at).abs() < 0.02, "discontinuity at {edge} Hz");
        }
    }

    #[test]
    #[should_panic]
    fn panics_outside_the_defined_range() {
        lth_one(10.0);
    }
}
