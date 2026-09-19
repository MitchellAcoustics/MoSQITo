//! IIR filter design: Butterworth and Chebyshev type I, via the bilinear
//! transform, producing second-order sections.
//!
//! This reproduces SciPy's `butter(..., output="sos")` and
//! `cheby1(..., output="sos")` including the pole/zero pairing and section
//! ordering of `zpk2sos(..., pairing="nearest")`. The ordering is not cosmetic:
//! MoSQITo designs band-pass filters down to a 25 Hz centre frequency at 48 kHz
//! (`_n_oct_time_filter.py:62`), where the poles sit extremely close to `z = 1`
//! and the cascade's conditioning depends on which poles share a section.

use num_complex::Complex64;
use std::f64::consts::PI;

/// A second-order section: `[b0, b1, b2, a0, a1, a2]`.
pub type Sos = [f64; 6];

/// Zero/pole/gain representation of a filter.
struct Zpk {
    zeros: Vec<Complex64>,
    poles: Vec<Complex64>,
    gain: f64,
}

/// Analog Butterworth lowpass prototype with unit cutoff.
///
/// Poles are `-exp(i·π·m / 2N)` for `m = -N+1, -N+3, …, N-1`; there are no
/// finite zeros and the gain is 1. Matches `scipy.signal.buttap`.
fn buttap(n: usize) -> Zpk {
    let nn = n as f64;
    let poles = (0..n)
        .map(|i| {
            let m = -(nn) + 1.0 + 2.0 * i as f64;
            -Complex64::new(0.0, PI * m / (2.0 * nn)).exp()
        })
        .collect();
    Zpk {
        zeros: Vec::new(),
        poles,
        gain: 1.0,
    }
}

/// Analog Chebyshev type I lowpass prototype with unit cutoff and `rp` dB of
/// passband ripple. Matches `scipy.signal.cheby1ap`.
fn cheby1ap(n: usize, rp: f64) -> Zpk {
    let nn = n as f64;
    let eps = (10f64.powf(0.1 * rp) - 1.0).sqrt();
    let mu = (1.0 / nn) * (1.0 / eps).asinh();

    let poles: Vec<Complex64> = (0..n)
        .map(|i| {
            let m = -(nn) + 1.0 + 2.0 * i as f64;
            let theta = PI * m / (2.0 * nn);
            -Complex64::new(mu, theta).sinh()
        })
        .collect();

    let mut gain = poles
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, p| acc * (-p))
        .re;
    if n % 2 == 0 {
        gain /= (1.0 + eps * eps).sqrt();
    }
    Zpk {
        zeros: Vec::new(),
        poles,
        gain,
    }
}

/// Scales a lowpass prototype to a new cutoff `wo`. Matches `lp2lp_zpk`.
fn lp2lp(zpk: &Zpk, wo: f64) -> Zpk {
    let degree = zpk.poles.len() - zpk.zeros.len();
    Zpk {
        zeros: zpk.zeros.iter().map(|z| z * wo).collect(),
        poles: zpk.poles.iter().map(|p| p * wo).collect(),
        gain: zpk.gain * wo.powi(degree as i32),
    }
}

/// Transforms a lowpass prototype to a bandpass with centre `wo` and width
/// `bw`. Matches `lp2bp_zpk`.
fn lp2bp(zpk: &Zpk, wo: f64, bw: f64) -> Zpk {
    let degree = zpk.poles.len() - zpk.zeros.len();
    let wo2 = Complex64::new(wo * wo, 0.0);

    let expand = |v: &[Complex64]| -> Vec<Complex64> {
        let scaled: Vec<Complex64> = v.iter().map(|x| x * (bw / 2.0)).collect();
        let mut out = Vec::with_capacity(scaled.len() * 2);
        for s in &scaled {
            out.push(s + (s * s - wo2).sqrt());
        }
        for s in &scaled {
            out.push(s - (s * s - wo2).sqrt());
        }
        out
    };

    let mut zeros = expand(&zpk.zeros);
    zeros.extend(std::iter::repeat(Complex64::new(0.0, 0.0)).take(degree));

    Zpk {
        zeros,
        poles: expand(&zpk.poles),
        gain: zpk.gain * bw.powi(degree as i32),
    }
}

/// Maps an analog filter to the digital domain. Matches `bilinear_zpk`.
fn bilinear(zpk: &Zpk, fs: f64) -> Zpk {
    let degree = zpk.poles.len() - zpk.zeros.len();
    let fs2 = Complex64::new(2.0 * fs, 0.0);

    let mut zeros: Vec<Complex64> = zpk.zeros.iter().map(|z| (fs2 + z) / (fs2 - z)).collect();
    let poles: Vec<Complex64> = zpk.poles.iter().map(|p| (fs2 + p) / (fs2 - p)).collect();

    let num = zpk
        .zeros
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, z| acc * (fs2 - z));
    let den = zpk
        .poles
        .iter()
        .fold(Complex64::new(1.0, 0.0), |acc, p| acc * (fs2 - p));
    let gain = zpk.gain * (num / den).re;

    zeros.extend(std::iter::repeat(Complex64::new(-1.0, 0.0)).take(degree));
    Zpk { zeros, poles, gain }
}

/// Pre-warps normalised digital frequencies for the bilinear transform.
///
/// SciPy's `iirfilter` works internally at `fs = 2`, giving
/// `warped = 4·tan(π·Wn/2)` for `Wn` normalised to Nyquist.
fn prewarp(wn: f64) -> f64 {
    4.0 * (PI * wn / 2.0).tan()
}

const CPLX_TOL: f64 = 100.0 * f64::EPSILON;

fn is_real(z: Complex64) -> bool {
    z.im.abs() <= CPLX_TOL * z.norm()
}

/// Splits values into one representative per complex-conjugate pair (with
/// positive imaginary part) followed by the purely real values, matching
/// `np.concatenate(_cplxreal(v))`.
fn cplxreal(v: &[Complex64]) -> Vec<Complex64> {
    if v.is_empty() {
        return Vec::new();
    }
    let mut sorted = v.to_vec();
    // lexsort by (real, |imag|)
    sorted.sort_by(|a, b| {
        a.re.total_cmp(&b.re)
            .then(a.im.abs().total_cmp(&b.im.abs()))
    });

    let (reals, complexes): (Vec<Complex64>, Vec<Complex64>) =
        sorted.iter().partition(|z| is_real(**z));

    let mut pos: Vec<Complex64> = complexes.iter().filter(|z| z.im > 0.0).copied().collect();
    let mut neg: Vec<Complex64> = complexes.iter().filter(|z| z.im < 0.0).copied().collect();
    assert_eq!(
        pos.len(),
        neg.len(),
        "complex value without a matching conjugate"
    );

    let key = |a: &Complex64, b: &Complex64| {
        a.re.total_cmp(&b.re)
            .then(a.im.abs().total_cmp(&b.im.abs()))
    };
    pos.sort_by(key);
    neg.sort_by(key);

    // Average out roundoff between each pair's two members.
    let mut out: Vec<Complex64> = pos
        .iter()
        .zip(&neg)
        .map(|(p, n)| (p + n.conj()) / 2.0)
        .collect();
    out.extend(reals.iter().map(|z| Complex64::new(z.re, 0.0)));
    out
}

/// Index of the pole closest to the unit circle — the "worst" one, which
/// `zpk2sos` places in the last section.
fn idx_worst(p: &[Complex64]) -> usize {
    (0..p.len())
        .min_by(|&i, &j| {
            (1.0 - p[i].norm())
                .abs()
                .total_cmp(&(1.0 - p[j].norm()).abs())
        })
        .expect("non-empty pole list")
}

/// Kind of root to select in [`nearest_idx`].
#[derive(Clone, Copy, PartialEq)]
enum Which {
    Real,
    Complex,
    Any,
}

/// Index of the value in `from` nearest to `to`, optionally restricted to real
/// or complex values. Matches `_nearest_real_complex_idx`.
fn nearest_idx(from: &[Complex64], to: Complex64, which: Which) -> usize {
    let mut order: Vec<usize> = (0..from.len()).collect();
    order.sort_by(|&i, &j| (from[i] - to).norm().total_cmp(&(from[j] - to).norm()));
    match which {
        Which::Any => order[0],
        Which::Real => *order
            .iter()
            .find(|&&i| is_real(from[i]))
            .expect("a real root"),
        Which::Complex => *order
            .iter()
            .find(|&&i| !is_real(from[i]))
            .expect("a complex root"),
    }
}

/// Builds one second-order section from up to two zeros and two poles,
/// right-aligning the shorter polynomials. Matches `_single_zpksos`.
fn single_sos(z: &[Complex64], p: &[Complex64], k: f64) -> Sos {
    let poly = |roots: &[Complex64]| -> Vec<f64> {
        let mut coeffs = vec![Complex64::new(1.0, 0.0)];
        for r in roots {
            let mut next = vec![Complex64::new(0.0, 0.0); coeffs.len() + 1];
            for (i, c) in coeffs.iter().enumerate() {
                next[i] += c;
                next[i + 1] -= c * r;
            }
            coeffs = next;
        }
        coeffs.iter().map(|c| c.re).collect()
    };

    let b: Vec<f64> = poly(z).iter().map(|v| v * k).collect();
    let a = poly(p);

    let mut sos = [0.0f64; 6];
    sos[3 - b.len()..3].copy_from_slice(&b);
    sos[6 - a.len()..6].copy_from_slice(&a);
    sos
}

/// Converts zeros, poles and gain to second-order sections, reproducing
/// `scipy.signal.zpk2sos` with `pairing="nearest"` for digital filters.
fn zpk2sos(zpk: &Zpk) -> Vec<Sos> {
    let mut z = zpk.zeros.clone();
    let mut p = zpk.poles.clone();

    if z.is_empty() && p.is_empty() {
        return vec![[zpk.gain, 0.0, 0.0, 1.0, 0.0, 0.0]];
    }

    // Equalise the counts, padding with zeros at the origin.
    let zero = Complex64::new(0.0, 0.0);
    while p.len() < z.len() {
        p.push(zero);
    }
    while z.len() < p.len() {
        z.push(zero);
    }
    let n_sections = p.len().max(z.len()).div_ceil(2);
    if p.len() % 2 == 1 {
        p.push(zero);
        z.push(zero);
    }

    z = cplxreal(&z);
    p = cplxreal(&p);

    let mut sos = vec![[0.0f64; 6]; n_sections];

    // Fill from the last section backwards, so the worst poles land last.
    for si in (0..n_sections).rev() {
        let p1_idx = idx_worst(&p);
        let p1 = p.remove(p1_idx);

        let remaining_real_poles = p.iter().filter(|x| is_real(**x)).count();
        let remaining_real_zeros = z.iter().filter(|x| is_real(**x)).count();

        if is_real(p1) && remaining_real_poles == 0 {
            // Last remaining real pole: pair it with a real zero.
            let z1 = z.remove(nearest_idx(&z, p1, Which::Real));
            sos[si] = single_sos(&[z1, zero], &[p1, zero], 1.0);
        } else if p.len() + 1 == z.len()
            && !is_real(p1)
            && remaining_real_poles == 1
            && remaining_real_zeros == 1
        {
            // One real pole and one real zero left over: p1 must take a
            // complex zero, or the leftovers cannot be paired.
            let z1 = z.remove(nearest_idx(&z, p1, Which::Complex));
            sos[si] = single_sos(&[z1, z1.conj()], &[p1, p1.conj()], 1.0);
        } else {
            let p2 = if is_real(p1) {
                let real_idx: Vec<usize> = (0..p.len()).filter(|&i| is_real(p[i])).collect();
                let sub: Vec<Complex64> = real_idx.iter().map(|&i| p[i]).collect();
                p.remove(real_idx[idx_worst(&sub)])
            } else {
                p1.conj()
            };

            if z.is_empty() {
                sos[si] = single_sos(&[], &[p1, p2], 1.0);
            } else {
                let z1 = z.remove(nearest_idx(&z, p1, Which::Any));
                if !is_real(z1) {
                    sos[si] = single_sos(&[z1, z1.conj()], &[p1, p2], 1.0);
                } else if !z.is_empty() {
                    let z2 = z.remove(nearest_idx(&z, p1, Which::Real));
                    sos[si] = single_sos(&[z1, z2], &[p1, p2], 1.0);
                } else {
                    sos[si] = single_sos(&[z1], &[p1, p2], 1.0);
                }
            }
        }
    }
    debug_assert!(p.is_empty() && z.is_empty(), "all poles and zeros consumed");

    // The overall gain goes entirely into the first section.
    for c in sos[0].iter_mut().take(3) {
        *c *= zpk.gain;
    }
    sos
}

/// Designs a Butterworth bandpass filter as second-order sections.
///
/// `low` and `high` are normalised to the Nyquist frequency, i.e. in `(0, 1)`.
/// Equivalent to `scipy.signal.butter(order, [low, high], "bandpass",
/// output="sos")`.
///
/// # Panics
/// Panics unless `0 < low < high < 1`.
pub fn butter_bandpass_sos(order: usize, low: f64, high: f64) -> Vec<Sos> {
    assert!(
        low > 0.0 && high < 1.0 && low < high,
        "band edges must satisfy 0 < low < high < 1 (got {low}, {high})"
    );
    let wl = prewarp(low);
    let wh = prewarp(high);
    let proto = buttap(order);
    let analog = lp2bp(&proto, (wl * wh).sqrt(), wh - wl);
    zpk2sos(&bilinear(&analog, 2.0))
}

/// Designs a Butterworth lowpass filter as second-order sections, with `wn`
/// normalised to the Nyquist frequency.
///
/// # Panics
/// Panics unless `0 < wn < 1`.
pub fn butter_lowpass_sos(order: usize, wn: f64) -> Vec<Sos> {
    assert!(
        wn > 0.0 && wn < 1.0,
        "cutoff must satisfy 0 < wn < 1 (got {wn})"
    );
    let proto = buttap(order);
    let analog = lp2lp(&proto, prewarp(wn));
    zpk2sos(&bilinear(&analog, 2.0))
}

/// Designs a Chebyshev type I lowpass filter as second-order sections.
///
/// `rp` is the passband ripple in dB and `wn` is normalised to the Nyquist
/// frequency. Equivalent to `scipy.signal.cheby1(order, rp, wn, output="sos")`.
/// `decimate` uses `cheby1_lowpass(8, 0.05, 0.8 / q)`.
///
/// # Panics
/// Panics unless `0 < wn < 1`.
pub fn cheby1_lowpass(order: usize, rp: f64, wn: f64) -> Vec<Sos> {
    assert!(
        wn > 0.0 && wn < 1.0,
        "cutoff must satisfy 0 < wn < 1 (got {wn})"
    );
    let proto = cheby1ap(order, rp);
    let analog = lp2lp(&proto, prewarp(wn));
    zpk2sos(&bilinear(&analog, 2.0))
}

/// Evaluates the frequency response of an SOS cascade at `n` points uniformly
/// spaced over `[0, π)`, matching `scipy.signal.sosfreqz(sos, worN=n)`.
pub fn sosfreqz(sos: &[Sos], n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let w = PI * i as f64 / n as f64;
            let z = Complex64::new(0.0, w).exp();
            let zi = 1.0 / z;
            let zi2 = zi * zi;
            sos.iter().fold(Complex64::new(1.0, 0.0), |acc, s| {
                let num = s[0] + s[1] * zi + s[2] * zi2;
                let den = s[3] + s[4] * zi + s[5] * zi2;
                acc * (num / den)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::filter::sosfilt;
    use approx::assert_relative_eq;

    /// Evaluates |H| of an SOS cascade at a normalised frequency (1 = Nyquist).
    fn gain_at(sos: &[Sos], wn: f64) -> f64 {
        let z = Complex64::new(0.0, PI * wn).exp();
        let zi = 1.0 / z;
        let zi2 = zi * zi;
        sos.iter()
            .fold(Complex64::new(1.0, 0.0), |acc, s| {
                acc * ((s[0] + s[1] * zi + s[2] * zi2) / (s[3] + s[4] * zi + s[5] * zi2))
            })
            .norm()
    }

    #[test]
    fn butterworth_lowpass_is_minus_three_db_at_cutoff() {
        for &order in &[2usize, 4, 8] {
            let sos = butter_lowpass_sos(order, 0.25);
            let g = gain_at(&sos, 0.25);
            assert_relative_eq!(20.0 * g.log10(), -3.0102999566, epsilon = 1e-8);
        }
    }

    #[test]
    fn butterworth_lowpass_is_unity_at_dc_and_rolls_off() {
        let sos = butter_lowpass_sos(8, 0.25);
        assert_relative_eq!(gain_at(&sos, 0.0), 1.0, epsilon = 1e-12);
        assert!(
            gain_at(&sos, 0.5) < 1e-2,
            "stopband should be well attenuated"
        );
        assert!(gain_at(&sos, 0.1) > 0.999, "passband should be flat");
    }

    #[test]
    fn butterworth_bandpass_is_minus_three_db_at_both_edges() {
        let sos = butter_bandpass_sos(3, 0.2, 0.4);
        for &edge in &[0.2, 0.4] {
            let db = 20.0 * gain_at(&sos, edge).log10();
            assert_relative_eq!(db, -3.0102999566, epsilon = 1e-7);
        }
        assert!(gain_at(&sos, 0.0) < 1e-9, "no DC response in a bandpass");
    }

    #[test]
    fn chebyshev1_ripple_stays_within_the_passband_bound() {
        let rp = 0.05;
        let sos = cheby1_lowpass(8, rp, 0.1);
        let floor = 10f64.powf(-rp / 20.0);
        for i in 0..=100 {
            let wn = 0.1 * i as f64 / 100.0;
            let g = gain_at(&sos, wn);
            assert!(g <= 1.0 + 1e-9, "gain {g} exceeded unity at wn={wn}");
            assert!(
                g >= floor - 1e-9,
                "gain {g} dipped below ripple floor at wn={wn}"
            );
        }
    }

    #[test]
    fn chebyshev1_hits_the_ripple_floor_exactly_at_cutoff() {
        // For Chebyshev I the cutoff is where the response leaves the ripple
        // band, not the -3 dB point.
        let rp = 0.05;
        let sos = cheby1_lowpass(8, rp, 0.2);
        assert_relative_eq!(gain_at(&sos, 0.2), 10f64.powf(-rp / 20.0), epsilon = 1e-9);
    }

    #[test]
    fn section_count_matches_filter_order() {
        assert_eq!(butter_lowpass_sos(8, 0.3).len(), 4);
        assert_eq!(cheby1_lowpass(8, 0.05, 0.1).len(), 4);
        // A bandpass doubles the order.
        assert_eq!(butter_bandpass_sos(3, 0.2, 0.4).len(), 3);
    }

    #[test]
    fn worst_pole_lands_in_the_last_section() {
        // zpk2sos orders sections so the poles nearest the unit circle are
        // filtered last, which is what keeps low-frequency bands conditioned.
        let sos = butter_lowpass_sos(8, 0.01);
        let pole_radius = |s: &Sos| {
            let (a1, a2) = (s[4] / s[3], s[5] / s[3]);
            let disc = Complex64::new(a1 * a1 - 4.0 * a2, 0.0).sqrt();
            let r1 = (-a1 + disc) / 2.0;
            let r2 = (-a1 - disc) / 2.0;
            r1.norm().max(r2.norm())
        };
        let first = pole_radius(&sos[0]);
        let last = pole_radius(&sos[sos.len() - 1]);
        assert!(
            last > first,
            "last section {last} should hold poles nearer |z|=1 than {first}"
        );
    }

    #[test]
    fn narrow_low_frequency_bandpass_stays_stable() {
        // The 25 Hz third-octave band at 48 kHz: the case that motivates
        // faithful section ordering.
        let (fs, fc) = (48000.0, 25.0);
        let (lo, hi) = (fc / 2f64.powf(1.0 / 6.0), fc * 2f64.powf(1.0 / 6.0));
        let sos = butter_bandpass_sos(3, 2.0 * lo / fs, 2.0 * hi / fs);

        let n = 48000;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * fc * i as f64 / fs).sin())
            .collect();
        let y = sosfilt(&sos, &x);
        assert!(y.iter().all(|v| v.is_finite()), "filter must not blow up");

        // After the transient, a tone at band centre passes at roughly unity.
        let tail = &y[n / 2..];
        let rms = (tail.iter().map(|v| v * v).sum::<f64>() / tail.len() as f64).sqrt();
        assert!(
            (0.5..=1.5).contains(&rms),
            "unexpected passband gain, rms = {rms}"
        );
    }
}
