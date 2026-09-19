//! Amplitude/level and frequency/Bark conversions.

/// Converts a dB level to a linear amplitude, matching `mosqito.utils.db2amp`.
///
/// # Panics
/// Panics if `reference` is zero.
pub fn db2amp(db: f64, reference: f64) -> f64 {
    assert!(reference != 0.0, "reference must be non-zero");
    10f64.powf(0.05 * db) * reference
}

/// Zwicker & Fastl's Bark-to-Hertz table (`Psychoacoustics`, table 6.1),
/// linearly interpolated, shared by [`bark2freq`] and [`freq2bark`].
const BARK_TABLE_HZ: [f64; 50] = [
    0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 510.0, 570.0, 630.0, 700.0,
    770.0, 840.0, 920.0, 1000.0, 1080.0, 1170.0, 1270.0, 1370.0, 1480.0, 1600.0, 1720.0, 1850.0,
    2000.0, 2150.0, 2320.0, 2500.0, 2700.0, 2900.0, 3150.0, 3400.0, 3700.0, 4000.0, 4400.0, 4800.0,
    5300.0, 5800.0, 6400.0, 7000.0, 7700.0, 8500.0, 9500.0, 10500.0, 12000.0, 13500.0, 15500.0,
    20000.0,
];

/// Converts Bark frequencies to Hertz, matching `mosqito.utils.bark2freq`.
///
/// Linear interpolation from Zwicker & Fastl's table 6.1 (0 to 24.5 Bark in
/// 0.5 Bark steps); out-of-range input clamps to the table's first/last value,
/// matching `numpy.interp`'s default (no extrapolation).
pub fn bark2freq(bark_axis: &[f64]) -> Vec<f64> {
    let xp: Vec<f64> = (0..50).map(|k| k as f64 * 0.5).collect();
    crate::dsp::interp(bark_axis, &xp, &BARK_TABLE_HZ)
}

/// Converts Hertz frequencies to Bark, matching `mosqito.utils.freq2bark`.
///
/// The inverse table lookup of [`bark2freq`]: linear interpolation from the
/// same Zwicker & Fastl table 6.1, clamped (not extrapolated) out of range.
pub fn freq2bark(freq_axis: &[f64]) -> Vec<f64> {
    let yp: Vec<f64> = (0..50).map(|k| k as f64 * 0.5).collect();
    crate::dsp::interp(freq_axis, &BARK_TABLE_HZ, &yp)
}

/// CEI 61672:2014 A-weighting curve, applied to a dB spectrum, matching
/// `mosqito.utils.spectrum2dBA`.
///
/// `spectrum` is assumed to span `0..fs/2` Hz, linearly spaced (as
/// `comp_spectrum`'s one-sided output does); the standard's 1/3-octave
/// A-weighting table is interpolated onto that axis.
pub fn spectrum2dba(spectrum: &[f64], fs: f64) -> Vec<f64> {
    const FREQ_STANDARD: [f64; 34] = [
        10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0,
        250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0,
        4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
    ];
    const A_STANDARD: [f64; 34] = [
        -70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4,
        -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1,
        -1.1, -2.5, -4.3, -6.6, -9.3,
    ];

    let n = spectrum.len();
    // Python's `int(fs / 2)` truncates toward zero before building the
    // linspace; matched here rather than using the exact `fs / 2.0`.
    let fs_half = (fs / 2.0).trunc();
    let freq_axis: Vec<f64> = if n <= 1 {
        vec![0.0; n]
    } else {
        (0..n)
            .map(|i| i as f64 * fs_half / (n - 1) as f64)
            .collect()
    };
    let a_pond = crate::dsp::interp(&freq_axis, &FREQ_STANDARD, &A_STANDARD);
    spectrum.iter().zip(&a_pond).map(|(s, a)| s + a).collect()
}

/// Converts an amplitude signal to dB relative to `reference`, matching
/// `mosqito.utils.amp2db` — except that this does not mutate its input.
/// MoSQITo's Python replaces any exact zero *in place* with `2e-12` before
/// taking the log, to avoid a `log10(0)` warning; this returns a new `Vec`
/// with the same substitution applied instead, since a pure function
/// silently mutating a caller's array is a footgun this port does not
/// reproduce (see `DEVIATIONS.md`).
///
/// # Panics
/// Panics if `reference` is zero.
pub fn amp2db(amp: &[f64], reference: f64) -> Vec<f64> {
    assert!(reference != 0.0, "reference must be non-zero");
    amp.iter()
        .map(|&a| {
            let a = if a == 0.0 { 2e-12 } else { a };
            20.0 * (a / reference).log10()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unity_amplitude_at_unity_reference_is_zero_db() {
        assert_relative_eq!(amp2db(&[1.0], 1.0)[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn doubling_amplitude_adds_about_6_db() {
        let db = amp2db(&[1.0, 2.0], 1.0);
        assert_relative_eq!(db[1] - db[0], 20.0 * 2f64.log10(), epsilon = 1e-12);
    }

    #[test]
    fn zero_amplitude_does_not_panic_or_return_infinity() {
        let db = amp2db(&[0.0], 1.0);
        assert!(db[0].is_finite());
    }

    #[test]
    fn does_not_mutate_its_input() {
        let input = [0.0, 1.0];
        let _ = amp2db(&input, 2e-5);
        assert_eq!(
            input,
            [0.0, 1.0],
            "amp2db must not mutate the caller's array"
        );
    }
}
