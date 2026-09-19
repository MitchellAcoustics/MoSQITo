//! Order statistics matching numpy's defaults.

/// Percentile with linear interpolation between order statistics, matching
/// `numpy.percentile(x, q)` with its default `method="linear"`.
///
/// ECMA-418-2 §7.1.8 takes the 90th percentile of the time-dependent roughness
/// as the single representative value, so the interpolation convention is
/// directly visible in the published result.
///
/// # Panics
/// Panics if `x` is empty or `q` is outside `[0, 100]`.
pub fn percentile_linear(x: &[f64], q: f64) -> f64 {
    assert!(!x.is_empty(), "percentile of an empty slice is undefined");
    assert!(
        (0.0..=100.0).contains(&q),
        "q must lie in [0, 100], got {q}"
    );

    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);

    let pos = (q / 100.0) * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return v[lo];
    }
    let frac = pos - lo as f64;
    v[lo] + frac * (v[hi] - v[lo])
}

/// Median matching `numpy.median`: the mean of the two central values for an
/// even-length input.
///
/// # Panics
/// Panics if `x` is empty.
pub fn median(x: &[f64]) -> f64 {
    percentile_linear(x, 50.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn percentile_endpoints_are_the_extremes() {
        let x = [3.0, 1.0, 4.0, 1.0, 5.0];
        assert_relative_eq!(percentile_linear(&x, 0.0), 1.0, epsilon = 1e-15);
        assert_relative_eq!(percentile_linear(&x, 100.0), 5.0, epsilon = 1e-15);
    }

    #[test]
    fn percentile_interpolates_between_order_statistics() {
        // Sorted: [1, 2, 3, 4]. pos = 0.9 * 3 = 2.7, so 3 + 0.7*(4-3) = 3.7.
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_relative_eq!(percentile_linear(&x, 90.0), 3.7, epsilon = 1e-12);
    }

    #[test]
    fn percentile_lands_exactly_on_a_sample_when_it_should() {
        // Sorted: [0, 1, 2, 3, 4]. pos = 0.5 * 4 = 2 exactly.
        let x = [4.0, 0.0, 2.0, 1.0, 3.0];
        assert_relative_eq!(percentile_linear(&x, 50.0), 2.0, epsilon = 1e-15);
    }

    #[test]
    fn median_averages_the_two_central_values() {
        assert_relative_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5, epsilon = 1e-15);
        assert_relative_eq!(median(&[5.0, 1.0, 3.0]), 3.0, epsilon = 1e-15);
    }

    #[test]
    fn single_element_is_its_own_percentile() {
        assert_relative_eq!(percentile_linear(&[7.0], 90.0), 7.0, epsilon = 1e-15);
    }

    #[test]
    fn input_order_does_not_matter() {
        let a = [9.0, 2.0, 7.0, 4.0, 1.0];
        let b = [1.0, 2.0, 4.0, 7.0, 9.0];
        for q in [0.0, 12.5, 50.0, 90.0, 100.0] {
            assert_relative_eq!(
                percentile_linear(&a, q),
                percentile_linear(&b, q),
                epsilon = 1e-15
            );
        }
    }
}
