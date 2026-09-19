//! Threshold-in-quiet, per E. Zwicker, H. Fastl: *Psychoacoustics*, Springer,
//! Berlin, Heidelberg, 1990 (figure 2.1), matching `mosqito.utils.LTQ`.
//!
//! Two independent digitised curves, both linearly interpolated (clamped, not
//! extrapolated) over a Bark axis: the standard Zwicker threshold (free-field
//! SPL re. 2e-5 Pa), and a second curve specific to `roughness_dw`'s
//! excitation-pattern stage, which the standard notes differs from the
//! ordinary hearing threshold.

/// Which digitised threshold-in-quiet curve to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtqReference {
    /// The standard Zwicker/Fastl absolute threshold of hearing.
    Zwicker,
    /// `roughness_dw`'s excitation-pattern threshold.
    Roughness,
}

const ZWICKER_X: [f64; 51] = [
    2.40445500e-04,
    2.97265560e-04,
    3.83444580e-04,
    4.87665300e-04,
    6.38024300e-04,
    8.26911900e-04,
    1.06166650e-03,
    1.54817010e-03,
    2.08369200e-03,
    2.92606650e-03,
    4.05138200e-03,
    6.02053940e-03,
    9.07457100e-03,
    1.38729700e-02,
    2.19208700e-02,
    3.27319300e-02,
    5.14789440e-02,
    7.68705100e-02,
    1.13174880e-01,
    1.81393830e-01,
    3.22530400e-01,
    4.97813570e-01,
    7.79302600e-01,
    1.26089480e00,
    1.83900010e00,
    2.49687100e00,
    3.30065520e00,
    4.01741500e00,
    4.82492625e00,
    5.38932800e00,
    6.06309500e00,
    6.92502964e00,
    8.12003875e00,
    9.61927600e00,
    1.15144812e01,
    1.26382940e01,
    1.41298833e01,
    1.53680458e01,
    1.68414347e01,
    1.86183590e01,
    1.99594350e01,
    2.10583273e01,
    2.17400030e01,
    2.22243315e01,
    2.25462820e01,
    2.27627940e01,
    2.29925427e01,
    2.31538743e01,
    2.32710993e01,
    2.33580350e01,
    2.34824357e01,
];

const ZWICKER_Y: [f64; 51] = [
    73.28456, 69.49444, 65.17124, 61.262524, 57.471996, 53.918324, 50.60151, 46.27743, 42.78269,
    39.050854, 36.08868, 32.060455, 28.624102, 25.48363, 22.461315, 20.090572, 18.015444, 16.47346,
    15.286756, 14.158645, 12.970594, 12.612313, 12.07634, 11.066555, 10.412693, 9.108175,
    7.8235803, 6.4902444, 5.551555, 5.00799, 4.513699, 5.2524257, 6.6815033, 8.505031, 11.117457,
    12.546872, 13.827873, 14.665177, 15.699439, 17.867777, 20.875132, 24.474821, 28.814922,
    33.59928, 38.82787, 45.191124, 51.653038, 58.213753, 65.02121, 75.87385, 89.63695,
];

const ROUGHNESS_X: [f64; 27] = [
    0.0, 0.01, 0.17, 0.8, 1.0, 1.5, 2.0, 3.3, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 13.3, 15.0, 16.0,
    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 24.5, 25.0,
];

const ROUGHNESS_Y: [f64; 27] = [
    130.0, 70.0, 60.0, 30.0, 25.0, 20.0, 15.0, 10.0, 8.1, 6.3, 5.0, 3.5, 2.5, 1.7, 0.0, -2.5, -4.0,
    -3.7, -1.5, 1.4, 3.8, 5.0, 7.5, 15.0, 48.0, 60.0, 130.0,
];

/// The threshold-in-quiet over `bark_axis`, matching `mosqito.utils.LTQ`.
pub fn ltq(bark_axis: &[f64], reference: LtqReference) -> Vec<f64> {
    match reference {
        LtqReference::Zwicker => crate::dsp::interp(bark_axis, &ZWICKER_X, &ZWICKER_Y),
        LtqReference::Roughness => crate::dsp::interp(bark_axis, &ROUGHNESS_X, &ROUGHNESS_Y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn zwicker_threshold_matches_mosqito_at_a_few_bark_points() {
        // Confirmed against the installed `mosqito` package directly
        // (`LTQ(array([1,6,15,24]), reference='zwicker')`); the digitised
        // curve's minimum sits near 6 Bark, not — as might be guessed from
        // the textbook 2-5 kHz most-sensitive region — near 15 Bark.
        let bark = [1.0, 6.0, 15.0, 24.0];
        let t = ltq(&bark, LtqReference::Zwicker);
        assert_relative_eq!(t[0], 11.613589722775826, max_relative = 1e-9);
        assert_relative_eq!(t[1], 4.559986946196534, max_relative = 1e-9);
        assert_relative_eq!(t[2], 14.41628703561875, max_relative = 1e-9);
        assert_relative_eq!(t[3], 89.63695, max_relative = 1e-9);
    }

    #[test]
    fn roughness_threshold_matches_a_table_point_exactly() {
        let t = ltq(&[13.3], LtqReference::Roughness);
        assert_relative_eq!(t[0], 0.0, epsilon = 1e-12);
    }
}
