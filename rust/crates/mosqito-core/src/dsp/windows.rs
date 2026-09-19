//! `numpy.hanning`/`numpy.blackman`-matching symmetric windows.
//!
//! Reproduces NumPy's own formula (built from `n = arange(1-M, M, 2)`, not the
//! more familiar `k = 0..M-1` indexing) rather than the mathematically
//! equivalent textbook form, so the two agree to machine precision rather
//! than merely up to floating-point summation-order noise. Distinct from
//! [`crate::roughness::ecma::von_hann_window`], which ECMA-418-2 normalises
//! differently.

use std::f64::consts::PI;

/// `numpy.hanning(m)`.
pub fn hanning(m: usize) -> Vec<f64> {
    symmetric_window(m, |x| 0.5 + 0.5 * (PI * x).cos())
}

/// `numpy.blackman(m)`.
pub fn blackman(m: usize) -> Vec<f64> {
    symmetric_window(m, |x| {
        0.42 + 0.5 * (PI * x).cos() + 0.08 * (2.0 * PI * x).cos()
    })
}

fn symmetric_window(m: usize, f: impl Fn(f64) -> f64) -> Vec<f64> {
    if m < 1 {
        return Vec::new();
    }
    if m == 1 {
        return vec![1.0];
    }
    let m_f = m as f64;
    (0..m)
        .map(|k| {
            let n = (1.0 - m_f) + 2.0 * k as f64;
            f(n / (m_f - 1.0))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hanning_is_zero_at_the_edges_and_one_in_the_middle() {
        let w = hanning(5);
        assert_relative_eq!(w[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(w[4], 0.0, epsilon = 1e-12);
        assert_relative_eq!(w[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn blackman_matches_numpy_at_length_5() {
        // numpy.blackman(5) == [-1.38777878e-17, 0.34, 1.0, 0.34, -1.38777878e-17]
        // — the edges are a tiny negative float, not exactly zero.
        let w = blackman(5);
        assert_relative_eq!(w[0], -1.38777878e-17, epsilon = 1e-22);
        assert_relative_eq!(w[1], 0.34, epsilon = 1e-12);
        assert_relative_eq!(w[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn length_one_and_zero_are_special_cased() {
        assert_eq!(hanning(1), vec![1.0]);
        assert_eq!(hanning(0), Vec::<f64>::new());
        assert_eq!(blackman(1), vec![1.0]);
    }
}
