//! IIR filtering: `lfilter`, `sosfilt` and `filtfilt`.
//!
//! All three reproduce SciPy's behaviour exactly, including the details that
//! are easy to get subtly wrong:
//!
//! * `lfilter` and `sosfilt` use the direct-form II transposed structure, so
//!   the state is the `z` accumulator array rather than delayed input/output
//!   samples. This matters for numerical agreement, not just for tidiness.
//! * [`lfilter_complex`] exists because ECMA-418-2's gammatone filter bank has
//!   complex coefficients (§5.1.4, eqs. 16–17); the band-pass signal is twice
//!   the real part of the complex-filtered signal.
//! * [`filtfilt`] pads with an *odd* extension of length
//!   `3 * max(len(a), len(b))` before the forward-backward pass, and
//!   initialises the filter state from the signal edge, matching
//!   `scipy.signal.filtfilt(..., padtype="odd")`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use num_complex::Complex64;

/// Applies an IIR filter along a signal using the transposed direct-form II
/// structure, as `scipy.signal.lfilter` does.
///
/// `b` and `a` are the numerator and denominator polynomials. `a[0]` must be
/// non-zero; the coefficients are normalised by it.
///
/// # Panics
/// Panics if `b` or `a` is empty, or if `a[0]` is zero.
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    assert!(
        !b.is_empty() && !a.is_empty(),
        "filter coefficients must be non-empty"
    );
    assert!(a[0] != 0.0, "a[0] must be non-zero");

    let n = b.len().max(a.len());
    let a0 = a[0];
    let bn: Vec<f64> = (0..n)
        .map(|i| b.get(i).copied().unwrap_or(0.0) / a0)
        .collect();
    let an: Vec<f64> = (0..n)
        .map(|i| a.get(i).copied().unwrap_or(0.0) / a0)
        .collect();

    let mut z = vec![0.0; n.saturating_sub(1)];
    let mut y = Vec::with_capacity(x.len());

    for &xi in x {
        let yi = bn[0] * xi + z.first().copied().unwrap_or(0.0);
        for k in 0..z.len() {
            let next = z.get(k + 1).copied().unwrap_or(0.0);
            z[k] = bn[k + 1] * xi + next - an[k + 1] * yi;
        }
        y.push(yi);
    }
    y
}

/// Applies an IIR filter with complex coefficients to a real signal.
///
/// ECMA-418-2 §5.1.4 defines the auditory filter bank this way: the band-pass
/// coefficients are the low-pass gammatone coefficients modulated by
/// `exp(j·2π·f·m/fs)`, and the band-pass signal is `2·Re{y}`. Returning the
/// full complex output keeps that final step explicit at the call site.
///
/// # Panics
/// Panics if `b` or `a` is empty, or if `a[0]` is zero.
pub fn lfilter_complex(b: &[Complex64], a: &[Complex64], x: &[f64]) -> Vec<Complex64> {
    assert!(
        !b.is_empty() && !a.is_empty(),
        "filter coefficients must be non-empty"
    );
    assert!(a[0] != Complex64::new(0.0, 0.0), "a[0] must be non-zero");

    let n = b.len().max(a.len());
    let zero = Complex64::new(0.0, 0.0);
    let a0 = a[0];
    let bn: Vec<Complex64> = (0..n)
        .map(|i| b.get(i).copied().unwrap_or(zero) / a0)
        .collect();
    let an: Vec<Complex64> = (0..n)
        .map(|i| a.get(i).copied().unwrap_or(zero) / a0)
        .collect();

    let mut z = vec![zero; n.saturating_sub(1)];
    let mut y = Vec::with_capacity(x.len());

    for &xi in x {
        let yi = bn[0] * xi + z.first().copied().unwrap_or(zero);
        for k in 0..z.len() {
            let next = z.get(k + 1).copied().unwrap_or(zero);
            z[k] = bn[k + 1] * xi + next - an[k + 1] * yi;
        }
        y.push(yi);
    }
    y
}

/// Applies a cascade of second-order sections, as `scipy.signal.sosfilt` does.
///
/// `sos` has one row per section, each `[b0, b1, b2, a0, a1, a2]`. Sections are
/// applied in order, each consuming the previous section's output.
///
/// # Panics
/// Panics if any row does not have exactly 6 coefficients.
pub fn sosfilt(sos: &[[f64; 6]], x: &[f64]) -> Vec<f64> {
    let mut y = x.to_vec();
    for section in sos {
        let b = [section[0], section[1], section[2]];
        let a = [section[3], section[4], section[5]];
        y = lfilter(&b, &a, &y);
    }
    y
}

/// Computes the initial filter state for a step response of unit amplitude,
/// equivalent to `scipy.signal.lfilter_zi`.
///
/// `filtfilt` uses this to start each pass from the signal's edge value rather
/// than from zero, which is what keeps the output free of startup transients.
fn lfilter_zi(b: &[f64], a: &[f64]) -> Vec<f64> {
    let n = b.len().max(a.len());
    let a0 = a[0];
    let bn: Vec<f64> = (0..n)
        .map(|i| b.get(i).copied().unwrap_or(0.0) / a0)
        .collect();
    let an: Vec<f64> = (0..n)
        .map(|i| a.get(i).copied().unwrap_or(0.0) / a0)
        .collect();

    let m = n - 1;
    if m == 0 {
        return Vec::new();
    }

    // zi solves (I - Aᵀ) zi = B, where A is the companion matrix of `an` and
    // B[i] = bn[i+1] - an[i+1] * bn[0]. Built and solved densely: these filters
    // are order 8 at most.
    //
    // The companion matrix has -an[1..] along its first *row* and ones on the
    // subdiagonal, so its transpose has -an[1..] down the first *column* and
    // ones on the superdiagonal. Transposing is not optional: for a first-order
    // section the two coincide, but from second order upwards they differ.
    let mut mat = vec![vec![0.0; m]; m];
    let mut rhs = vec![0.0; m];
    for i in 0..m {
        rhs[i] = bn[i + 1] - an[i + 1] * bn[0];
    }
    for (i, row) in mat.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut at_ij = 0.0;
            if j == 0 {
                at_ij -= an[i + 1];
            }
            if j == i + 1 {
                at_ij += 1.0;
            }
            *cell = if i == j { 1.0 - at_ij } else { -at_ij };
        }
    }
    gaussian_solve(&mut mat, &mut rhs);
    rhs
}

/// Solves a small dense linear system in place by Gaussian elimination with
/// partial pivoting.
fn gaussian_solve(mat: &mut [Vec<f64>], rhs: &mut [f64]) {
    let n = rhs.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| mat[i][col].abs().total_cmp(&mat[j][col].abs()))
            .unwrap();
        mat.swap(col, pivot);
        rhs.swap(col, pivot);
        let d = mat[col][col];
        if d == 0.0 {
            continue;
        }
        // Split so the pivot row can be read while the rows below are written.
        let (upper, lower) = mat.split_at_mut(col + 1);
        let pivot_row = &upper[col];
        let pivot_rhs = rhs[col];
        for (offset, row) in lower.iter_mut().enumerate() {
            let factor = row[col] / d;
            if factor == 0.0 {
                continue;
            }
            for (cell, p) in row.iter_mut().zip(pivot_row.iter()).skip(col) {
                *cell -= factor * p;
            }
            rhs[col + 1 + offset] -= factor * pivot_rhs;
        }
    }
    for col in (0..n).rev() {
        let acc = mat[col]
            .iter()
            .zip(rhs.iter())
            .skip(col + 1)
            .fold(rhs[col], |acc, (m, r)| acc - m * r);
        rhs[col] = if mat[col][col] == 0.0 {
            0.0
        } else {
            acc / mat[col][col]
        };
    }
}

/// Applies an IIR filter forwards and backwards, giving zero phase distortion.
///
/// Matches `scipy.signal.filtfilt` with its default `padtype="odd"` and
/// `padlen = 3 * max(len(a), len(b))`. The odd extension reflects the signal
/// through its endpoint — `2*x[0] - x[k]` at the start — which suppresses the
/// edge transients a plain reflection would leave behind.
///
/// Returns `None` if the signal is too short for the required padding, which is
/// the condition SciPy raises a `ValueError` for.
pub fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Option<Vec<f64>> {
    let ntaps = a.len().max(b.len());
    let padlen = 3 * ntaps;
    if x.len() <= padlen {
        return None;
    }

    let ext = odd_extension(x, padlen);
    let zi = lfilter_zi(b, a);

    // Forward pass, state seeded from the leading edge.
    let fwd = lfilter_with_zi(b, a, &ext, &zi, ext[0]);
    // Backward pass over the reversed forward output.
    let mut rev: Vec<f64> = fwd.iter().rev().copied().collect();
    let edge = rev[0];
    rev = lfilter_with_zi(b, a, &rev, &zi, edge);
    rev.reverse();

    Some(rev[padlen..rev.len() - padlen].to_vec())
}

/// Runs `lfilter` with the state initialised to `zi * x0`, as the two passes of
/// `filtfilt` do.
fn lfilter_with_zi(b: &[f64], a: &[f64], x: &[f64], zi: &[f64], x0: f64) -> Vec<f64> {
    let n = b.len().max(a.len());
    let a0 = a[0];
    let bn: Vec<f64> = (0..n)
        .map(|i| b.get(i).copied().unwrap_or(0.0) / a0)
        .collect();
    let an: Vec<f64> = (0..n)
        .map(|i| a.get(i).copied().unwrap_or(0.0) / a0)
        .collect();

    let mut z: Vec<f64> = zi.iter().map(|v| v * x0).collect();
    z.resize(n.saturating_sub(1), 0.0);

    let mut y = Vec::with_capacity(x.len());
    for &xi in x {
        let yi = bn[0] * xi + z.first().copied().unwrap_or(0.0);
        for k in 0..z.len() {
            let next = z.get(k + 1).copied().unwrap_or(0.0);
            z[k] = bn[k + 1] * xi + next - an[k + 1] * yi;
        }
        y.push(yi);
    }
    y
}

/// Initial state for an SOS cascade's step response, matching
/// `scipy.signal.sosfilt_zi`.
///
/// Each section's state is scaled by the cumulative DC gain of the sections
/// before it, so the whole cascade settles immediately on a constant input.
fn sosfilt_zi(sos: &[[f64; 6]]) -> Vec<[f64; 2]> {
    let mut zi = Vec::with_capacity(sos.len());
    let mut scale = 1.0;
    for s in sos {
        let b = [s[0], s[1], s[2]];
        let a = [s[3], s[4], s[5]];
        let z = lfilter_zi(&b, &a);
        zi.push([
            scale * z.first().copied().unwrap_or(0.0),
            scale * z.get(1).copied().unwrap_or(0.0),
        ]);
        // b.sum()/a.sum() is this section's gain at DC.
        scale *= b.iter().sum::<f64>() / a.iter().sum::<f64>();
    }
    zi
}

/// Runs an SOS cascade with a given per-section initial state, scaled by `x0`.
fn sosfilt_with_zi(sos: &[[f64; 6]], x: &[f64], zi: &[[f64; 2]], x0: f64) -> Vec<f64> {
    let mut y = x.to_vec();
    for (s, z) in sos.iter().zip(zi) {
        let b = [s[0], s[1], s[2]];
        let a = [s[3], s[4], s[5]];
        y = lfilter_with_zi(&b, &a, &y, z, x0);
    }
    y
}

/// Zero-phase forward-backward filtering through an SOS cascade, matching
/// `scipy.signal.sosfiltfilt`.
///
/// The padding length differs from [`filtfilt`]'s: SciPy uses
/// `3 * (2·n_sections + 1 - min(#{b2 == 0}, #{a2 == 0}))`, discounting sections
/// that are effectively first order. `decimate` goes through this path, so the
/// discount is visible in every decimated result.
///
/// Returns `None` when the signal is shorter than the required padding.
pub fn sosfiltfilt(sos: &[[f64; 6]], x: &[f64]) -> Option<Vec<f64>> {
    if sos.is_empty() {
        return Some(x.to_vec());
    }
    let b2_zeros = sos.iter().filter(|s| s[2] == 0.0).count();
    let a2_zeros = sos.iter().filter(|s| s[5] == 0.0).count();
    let ntaps = 2 * sos.len() + 1 - b2_zeros.min(a2_zeros);
    let padlen = 3 * ntaps;
    if x.len() <= padlen {
        return None;
    }

    let ext = odd_extension(x, padlen);
    let zi = sosfilt_zi(sos);

    let fwd = sosfilt_with_zi(sos, &ext, &zi, ext[0]);
    let mut rev: Vec<f64> = fwd.iter().rev().copied().collect();
    let edge = rev[0];
    rev = sosfilt_with_zi(sos, &rev, &zi, edge);
    rev.reverse();

    Some(rev[padlen..rev.len() - padlen].to_vec())
}

/// Downsamples by an integer factor with a zero-phase anti-alias filter,
/// matching `scipy.signal.decimate(x, q)` with its defaults.
///
/// SciPy's default is an order-8 Chebyshev type I lowpass at `0.8/q` with
/// 0.05 dB ripple, applied through [`sosfiltfilt`]. ECMA-418-2 does not specify
/// a downsampling method for the envelopes (§7.1.2); MoSQITo chooses decimation
/// over Fourier resampling to avoid assuming the signal is periodic, so this is
/// the behaviour the published roughness values were produced with.
///
/// Returns `None` when the signal is too short for the filter's padding.
pub fn decimate(x: &[f64], q: usize) -> Option<Vec<f64>> {
    assert!(q >= 1, "decimation factor must be at least 1");
    if q == 1 {
        return Some(x.to_vec());
    }
    let sos = decimate_filter_design(q);
    let filtered = sosfiltfilt(&sos, x)?;
    Some(filtered.iter().step_by(q).copied().collect())
}

/// `decimate`'s anti-alias filter, memoised by `q`.
///
/// Only `q` varies call to call (the order and ripple are `decimate`'s own
/// fixed constants); every caller that decimates by a given factor
/// repeatedly — e.g. once per block, per critical band, in ECMA-418-2
/// roughness's envelope pipeline — would otherwise redesign the identical
/// Chebyshev-I cascade (pole/zero computation, bilinear transform, SOS
/// pairing) from scratch on every call.
fn decimate_filter_design(q: usize) -> Vec<crate::dsp::design::Sos> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Vec<crate::dsp::design::Sos>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    cache
        .lock()
        .expect("decimate filter cache poisoned")
        .entry(q)
        .or_insert_with(|| crate::dsp::design::cheby1_lowpass(8, 0.05, 0.8 / q as f64))
        .clone()
}

/// Extends a signal at both ends by an odd reflection about its endpoints,
/// matching `scipy.signal._arraytools.odd_ext`.
fn odd_extension(x: &[f64], padlen: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * padlen);
    // Leading: 2*x[0] - x[padlen], ..., 2*x[0] - x[1]
    for k in (1..=padlen).rev() {
        out.push(2.0 * x[0] - x[k]);
    }
    out.extend_from_slice(x);
    // Trailing: 2*x[-1] - x[-2], ..., 2*x[-1] - x[-padlen-1]
    let last = x[n - 1];
    for k in 1..=padlen {
        out.push(2.0 * last - x[n - 1 - k]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn lfilter_matches_manual_difference_equation() {
        // y[n] = 0.5 x[n] + 0.2 x[n-1] - 0.3 y[n-1]
        let b = [0.5, 0.2];
        let a = [1.0, 0.3];
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = lfilter(&b, &a, &x);

        let mut expected = Vec::new();
        let mut y_prev = 0.0;
        let mut x_prev = 0.0;
        for &xi in &x {
            let yi = 0.5 * xi + 0.2 * x_prev - 0.3 * y_prev;
            expected.push(yi);
            y_prev = yi;
            x_prev = xi;
        }
        for (got, want) in y.iter().zip(&expected) {
            assert_relative_eq!(got, want, max_relative = 1e-14);
        }
    }

    #[test]
    fn lfilter_normalises_by_a0() {
        let x = [1.0, -2.0, 3.0];
        let plain = lfilter(&[0.5, 0.2], &[1.0, 0.3], &x);
        let scaled = lfilter(&[1.0, 0.4], &[2.0, 0.6], &x);
        for (p, s) in plain.iter().zip(&scaled) {
            assert_relative_eq!(p, s, max_relative = 1e-14);
        }
    }

    #[test]
    fn sosfilt_equals_chained_lfilter() {
        let sos = [
            [0.5, 0.2, 0.1, 1.0, 0.3, 0.05],
            [1.0, -0.5, 0.25, 1.0, 0.1, 0.02],
        ];
        let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.37).sin()).collect();

        let via_sos = sosfilt(&sos, &x);
        let stage1 = lfilter(&sos[0][..3], &sos[0][3..], &x);
        let stage2 = lfilter(&sos[1][..3], &sos[1][3..], &stage1);
        for (got, want) in via_sos.iter().zip(&stage2) {
            assert_relative_eq!(got, want, max_relative = 1e-14);
        }
    }

    #[test]
    fn lfilter_complex_on_real_coefficients_matches_real_lfilter() {
        let b = [0.5, 0.2];
        let a = [1.0, 0.3];
        let x: Vec<f64> = (0..16).map(|i| (i as f64 * 0.21).cos()).collect();

        let real = lfilter(&b, &a, &x);
        let bc: Vec<Complex64> = b.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let ac: Vec<Complex64> = a.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let cplx = lfilter_complex(&bc, &ac, &x);

        for (got, want) in cplx.iter().zip(&real) {
            assert_relative_eq!(got.re, want, max_relative = 1e-14);
            assert_relative_eq!(got.im, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn odd_extension_reflects_through_endpoints() {
        let x = [1.0, 2.0, 4.0, 8.0];
        let ext = odd_extension(&x, 2);
        // Leading: 2*1 - 4 = -2, 2*1 - 2 = 0. Trailing: 2*8 - 4 = 12, 2*8 - 2 = 14.
        assert_eq!(ext, vec![-2.0, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn filtfilt_is_zero_phase_away_from_the_edges() {
        // Zero-phase filtering leaves a symmetric signal symmetric in the
        // interior. The edges are not exactly symmetric — the odd extension
        // plus the seeded initial state leaves a small transient, and SciPy
        // has the same asymmetry — so the padded region is excluded here and
        // the edge behaviour is pinned by the golden-vector test instead.
        let b = [0.2, 0.2];
        let a = [1.0, -0.6];
        let n = 201;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 - (n as f64 - 1.0) / 2.0;
                (-t * t / 800.0).exp()
            })
            .collect();
        let y = filtfilt(&b, &a, &x).expect("signal long enough");
        assert_eq!(y.len(), x.len());
        for i in 40..n / 2 {
            assert_relative_eq!(y[i], y[n - 1 - i], max_relative = 1e-6);
        }
    }

    #[test]
    fn sosfiltfilt_agrees_with_filtfilt_for_a_single_section() {
        let sos = [[0.2, 0.2, 0.0, 1.0, -0.6, 0.0]];
        let x: Vec<f64> = (0..128).map(|i| (i as f64 * 0.17).sin()).collect();
        let via_sos = sosfiltfilt(&sos, &x).expect("long enough");
        let via_tf = filtfilt(&[0.2, 0.2], &[1.0, -0.6], &x).expect("long enough");
        // The padding lengths differ (sosfiltfilt discounts the zero
        // second-order terms), so the two transients are not identical and
        // agreement is only approximate, even well inside the signal.
        for i in 30..x.len() - 30 {
            assert_relative_eq!(via_sos[i], via_tf[i], max_relative = 1e-6, epsilon = 1e-12);
        }
    }

    #[test]
    fn decimate_keeps_every_qth_sample() {
        let x: Vec<f64> = (0..512).map(|i| (i as f64 * 0.01).sin()).collect();
        let y = decimate(&x, 4).expect("long enough");
        assert_eq!(y.len(), x.len().div_ceil(4));
    }

    #[test]
    fn decimate_rejects_signals_shorter_than_the_padding() {
        let x = vec![1.0; 8];
        assert!(decimate(&x, 4).is_none());
    }

    #[test]
    fn filtfilt_rejects_signals_shorter_than_the_padding() {
        let b = [0.2, 0.2];
        let a = [1.0, -0.6];
        let x = [1.0, 2.0, 3.0];
        assert!(filtfilt(&b, &a, &x).is_none());
    }

    #[test]
    fn lfilter_zi_settles_a_constant_signal_immediately() {
        // With the state seeded from the edge value, a constant input must come
        // straight out unchanged — that is the property filtfilt relies on.
        let b = [0.2, 0.2];
        let a = [1.0, -0.6];
        let zi = lfilter_zi(&b, &a);
        let x = vec![3.0; 10];
        let y = lfilter_with_zi(&b, &a, &x, &zi, x[0]);
        for v in &y {
            assert_relative_eq!(v, &3.0, max_relative = 1e-12);
        }
    }
}
