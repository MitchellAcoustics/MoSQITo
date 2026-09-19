//! Loudness unit conversions: sone ↔ phon.

/// Converts loudness in sones to loudness level in phons, matching
/// `mosqito.sq_metrics.loudness.utils.sone_to_phon.sone_to_phon`.
///
/// Based on the BASIC program published in "Program for calculating loudness
/// according to DIN 45631 (ISO 532-1:2017)", E. Zwicker and H. Fastl, J.A.S.J
/// (E) 12, 1 (1991).
pub fn sone_to_phon(sone: f64) -> f64 {
    if sone < 1.0 {
        let phon = 40.0 * sone.powf(0.35);
        phon.max(3.0)
    } else {
        10.0 * sone.log2() + 40.0
    }
}

/// A 29-point ISO 226 equal-loudness contour at the given phon level (20 Hz
/// to 12.5 kHz), matching
/// `mosqito.sq_metrics.loudness.utils.equal_loudness_contours.equal_loudness_contours`.
///
/// Ported from the MATLAB project this Python itself is based on (Jeff
/// Tackett's "ISO 226 Equal-Loudness-Level Contour Signal").
///
/// Returns `(spl, freq_axis)`: `spl` in dB SPL, `freq_axis` in Hz.
pub fn equal_loudness_contours(phones: f64) -> ([f64; 29], [f64; 29]) {
    const FREQ: [f64; 29] = [
        20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
        500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
        6300.0, 8000.0, 10000.0, 12500.0,
    ];
    const AF: [f64; 29] = [
        0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, 0.301, 0.288,
        0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245,
        0.254, 0.271, 0.301,
    ];
    const LU: [f64; 29] = [
        -31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1, -2.0, -1.1, -0.4,
        0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1,
    ];
    const TF: [f64; 29] = [
        78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0,
        2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
    ];

    let mut spl = [0.0f64; 29];
    for i in 0..29 {
        let af_i = AF[i];
        let a_f = 4.47e-3 * (10f64.powf(0.025 * phones) - 1.15)
            + (0.4 * 10f64.powf((TF[i] + LU[i]) / 10.0 - 9.0)).powf(af_i);
        spl[i] = (10.0 / af_i) * a_f.log10() - LU[i] + 94.0;
    }

    (spl, FREQ)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sone_to_phon_matches_the_din_reference_point() {
        // 1 sone is defined as 40 phon.
        assert_relative_eq!(sone_to_phon(1.0), 40.0, epsilon = 1e-9);
    }

    #[test]
    fn sone_to_phon_doubles_loudness_by_10_phon() {
        assert_relative_eq!(sone_to_phon(2.0) - sone_to_phon(1.0), 10.0, epsilon = 1e-9);
    }

    #[test]
    fn sone_to_phon_is_floored_at_3_phon() {
        assert_eq!(sone_to_phon(1e-6), 3.0);
    }

    #[test]
    fn equal_loudness_contours_at_1khz_is_close_to_the_phon_level() {
        // At 1 kHz, phons are defined to equal dB SPL by construction.
        let (spl, freq) = equal_loudness_contours(60.0);
        let idx = freq.iter().position(|&f| f == 1000.0).unwrap();
        assert_relative_eq!(spl[idx], 60.0, max_relative = 0.05);
    }
}
