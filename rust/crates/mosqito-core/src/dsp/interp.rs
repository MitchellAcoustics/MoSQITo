//! Interpolation: piecewise linear and monotone piecewise cubic (PCHIP).

/// Piecewise linear interpolation matching `scipy.interpolate.interp1d(xp,
/// fp, bounds_error=False, fill_value=0)`.
///
/// Unlike [`interp`], queries outside `[xp[0], xp[-1]]` return `0.0` rather
/// than clamping to the nearest endpoint value. `loudness_zwst_freq` uses
/// this to zero-pad an input spectrum out to the full 24 kHz ISO 532-1
/// range, and getting the two functions' out-of-range behaviour mixed up
/// would silently extend the spectrum's edge level instead of padding with
/// silence.
///
/// # Panics
/// Panics if `xp` is empty or if `xp` and `fp` have different lengths.
pub fn interp_zero_fill(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    assert!(
        !xp.is_empty(),
        "interpolation needs at least one sample point"
    );
    assert_eq!(xp.len(), fp.len(), "xp and fp must have the same length");

    x.iter()
        .map(|&q| {
            if q < xp[0] || q > xp[xp.len() - 1] {
                return 0.0;
            }
            if q == xp[xp.len() - 1] {
                return fp[fp.len() - 1];
            }
            let i = match xp.binary_search_by(|v| v.total_cmp(&q)) {
                Ok(i) => return fp[i],
                Err(i) => i - 1,
            };
            let t = (q - xp[i]) / (xp[i + 1] - xp[i]);
            fp[i] + t * (fp[i + 1] - fp[i])
        })
        .collect()
}

/// Piecewise linear interpolation matching `numpy.interp`.
///
/// `xp` must be increasing. Queries outside the range clamp to the endpoint
/// values, which is numpy's default behaviour.
///
/// # Panics
/// Panics if `xp` is empty or if `xp` and `fp` have different lengths.
pub fn interp(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    assert!(
        !xp.is_empty(),
        "interpolation needs at least one sample point"
    );
    assert_eq!(xp.len(), fp.len(), "xp and fp must have the same length");

    x.iter()
        .map(|&q| {
            if q <= xp[0] {
                return fp[0];
            }
            if q >= xp[xp.len() - 1] {
                return fp[fp.len() - 1];
            }
            let i = match xp.binary_search_by(|v| v.total_cmp(&q)) {
                Ok(i) => return fp[i],
                Err(i) => i - 1,
            };
            let t = (q - xp[i]) / (xp[i + 1] - xp[i]);
            fp[i] + t * (fp[i + 1] - fp[i])
        })
        .collect()
}

/// Monotone piecewise cubic Hermite interpolation (Fritsch–Carlson), matching
/// `scipy.interpolate.pchip_interpolate`.
///
/// ECMA-418-2 §7.1.7 uses this to resample the specific-roughness estimate onto
/// a uniform 50 Hz grid. A plain cubic spline would overshoot and produce
/// negative roughness between samples; the monotone construction is what keeps
/// the result physical.
///
/// # Panics
/// Panics if fewer than two points are supplied, or if the lengths differ.
pub fn pchip(x: &[f64], y: &[f64], xq: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    assert!(x.len() >= 2, "PCHIP needs at least two points");

    let n = x.len();
    let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
    let delta: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / h[i]).collect();

    let mut d = vec![0.0; n];
    if n == 2 {
        d[0] = delta[0];
        d[1] = delta[0];
    } else {
        for i in 1..n - 1 {
            // A local extremum gets a zero derivative; that is what prevents
            // overshoot. Otherwise use the weighted harmonic mean of the
            // neighbouring secant slopes.
            if delta[i - 1] * delta[i] > 0.0 {
                let w1 = 2.0 * h[i] + h[i - 1];
                let w2 = h[i] + 2.0 * h[i - 1];
                d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }
        d[0] = edge_derivative(h[0], h[1], delta[0], delta[1]);
        d[n - 1] = edge_derivative(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
    }

    xq.iter()
        .map(|&q| {
            // Locate the interval, clamping so queries outside the data
            // extrapolate from the end cubics as SciPy's PCHIP does.
            let i = match x.binary_search_by(|v| v.total_cmp(&q)) {
                Ok(i) => i.min(n - 2),
                Err(i) => i.saturating_sub(1).min(n - 2),
            };
            let s = q - x[i];
            let hi = h[i];
            let c = (3.0 * delta[i] - 2.0 * d[i] - d[i + 1]) / hi;
            let b = (d[i] - 2.0 * delta[i] + d[i + 1]) / (hi * hi);
            y[i] + s * (d[i] + s * (c + s * b))
        })
        .collect()
}

/// One-sided three-point derivative estimate for a PCHIP endpoint, shape
/// preserving in the sense of Fritsch and Carlson.
fn edge_derivative(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if d * d0 <= 0.0 {
        0.0
    } else if d0 * d1 <= 0.0 && d.abs() > (3.0 * d0).abs() {
        3.0 * d0
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn interp_reproduces_the_sample_points() {
        let xp = [0.0, 1.0, 2.0, 4.0];
        let fp = [0.0, 10.0, 20.0, 40.0];
        let got = interp(&xp, &xp, &fp);
        for (g, w) in got.iter().zip(&fp) {
            assert_relative_eq!(g, w, epsilon = 1e-15);
        }
    }

    #[test]
    fn interp_zero_fill_zero_pads_outside_the_range() {
        let xp = [1.0, 2.0];
        let fp = [10.0, 20.0];
        assert_eq!(interp_zero_fill(&[-5.0, 99.0], &xp, &fp), vec![0.0, 0.0]);
    }

    #[test]
    fn interp_zero_fill_agrees_with_interp_inside_the_range() {
        let xp = [0.0, 1.0, 2.0, 4.0];
        let fp = [0.0, 10.0, 20.0, 40.0];
        let xq = [0.0, 0.5, 1.5, 4.0];
        assert_eq!(interp_zero_fill(&xq, &xp, &fp), interp(&xq, &xp, &fp));
    }

    #[test]
    fn interp_is_linear_between_samples() {
        let xp = [0.0, 2.0];
        let fp = [0.0, 10.0];
        assert_relative_eq!(interp(&[0.5], &xp, &fp)[0], 2.5, epsilon = 1e-15);
        assert_relative_eq!(interp(&[1.5], &xp, &fp)[0], 7.5, epsilon = 1e-15);
    }

    #[test]
    fn interp_clamps_outside_the_range() {
        let xp = [1.0, 2.0];
        let fp = [10.0, 20.0];
        assert_eq!(interp(&[-5.0, 99.0], &xp, &fp), vec![10.0, 20.0]);
    }

    #[test]
    fn pchip_passes_through_every_data_point() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.5];
        let y = [1.0, 3.0, 2.0, 5.0, 4.0];
        let got = pchip(&x, &y, &x);
        for (g, w) in got.iter().zip(&y) {
            assert_relative_eq!(g, w, epsilon = 1e-12);
        }
    }

    #[test]
    fn pchip_is_exact_for_a_straight_line() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
        for &q in &[0.25, 0.9, 1.5, 2.7] {
            assert_relative_eq!(pchip(&x, &y, &[q])[0], 2.0 * q + 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn pchip_does_not_overshoot_monotone_data() {
        // The defining property: no value may leave the bracketing data range.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 0.0, 0.0, 1.0, 1.0];
        for i in 0..=400 {
            let q = 4.0 * i as f64 / 400.0;
            let v = pchip(&x, &y, &[q])[0];
            assert!((-1e-12..=1.0 + 1e-12).contains(&v), "overshoot at {q}: {v}");
        }
    }

    #[test]
    fn pchip_flattens_at_local_extrema() {
        // A sign change in the secant slopes forces a zero derivative, so the
        // interpolant must not exceed the peak value.
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 1.0, 0.0];
        for i in 0..=200 {
            let q = 2.0 * i as f64 / 200.0;
            assert!(pchip(&x, &y, &[q])[0] <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn pchip_handles_two_points_as_a_straight_line() {
        let x = [0.0, 2.0];
        let y = [1.0, 5.0];
        assert_relative_eq!(pchip(&x, &y, &[1.0])[0], 3.0, epsilon = 1e-12);
    }
}
