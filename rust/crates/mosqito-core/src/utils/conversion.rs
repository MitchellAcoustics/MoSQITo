//! Amplitude/level conversions.

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
