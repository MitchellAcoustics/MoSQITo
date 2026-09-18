//! FFT-based operations: the analytic-signal envelope and Fourier resampling.

use realfft::RealFftPlanner;
use rustfft::{num_complex::Complex, FftPlanner};

/// Magnitude of the analytic signal, i.e. `abs(scipy.signal.hilbert(x))`.
///
/// The analytic signal is formed by zeroing the negative-frequency half of the
/// spectrum and doubling the positive half, leaving DC — and, for even-length
/// inputs, Nyquist — unscaled. ECMA-418-2 §7.1.2 uses this to extract the
/// band-pass signals' envelopes.
pub fn hilbert_envelope(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![x[0].abs()];
    }

    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);

    let mut buf: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fwd.process(&mut buf);

    // h[0] = 1; h[n/2] = 1 when n is even; h[1..n/2] = 2; the rest are 0.
    if n % 2 == 0 {
        for v in buf.iter_mut().take(n / 2).skip(1) {
            *v *= 2.0;
        }
        for v in buf.iter_mut().skip(n / 2 + 1) {
            *v = Complex::new(0.0, 0.0);
        }
    } else {
        for v in buf.iter_mut().take(n.div_ceil(2)).skip(1) {
            *v *= 2.0;
        }
        for v in buf.iter_mut().skip(n.div_ceil(2)) {
            *v = Complex::new(0.0, 0.0);
        }
    }

    inv.process(&mut buf);
    let scale = 1.0 / n as f64;
    buf.iter().map(|c| c.norm() * scale).collect()
}

/// Resamples a signal to `num` samples using the Fourier method, matching
/// `scipy.signal.resample`.
///
/// This treats the signal as periodic. MoSQITo applies it to every input that
/// is not already at 48 kHz (`mosqito/utils/load.py:85-88`), so the ISO 532-1
/// 44.1 kHz test signal reaches its published loudness only through exactly
/// this resampler.
///
/// The Nyquist bin needs care: when downsampling from an even length, SciPy
/// halves the retained Nyquist component, and when upsampling to an even length
/// it halves the original Nyquist bin and mirrors it. Skipping either step
/// leaves a small but systematic error.
pub fn resample(x: &[f64], num: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || num == 0 {
        return vec![0.0; num];
    }
    if num == n {
        return x.to_vec();
    }

    let mut planner = RealFftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(n);
    let mut spectrum = fwd.make_output_vec();
    let mut input = x.to_vec();
    fwd.process(&mut input, &mut spectrum)
        .expect("real FFT of the input");

    // Only the first min(num, n)/2 + 1 bins are meaningful in both the old and
    // the new sampling; everything above that is discarded or left zero.
    let m = num.min(n);
    let m2 = m / 2 + 1;
    let mut out_spec = vec![Complex::new(0.0, 0.0); num / 2 + 1];
    let keep = m2.min(spectrum.len()).min(out_spec.len());
    out_spec[..keep].copy_from_slice(&spectrum[..keep]);

    if m % 2 == 0 && num != n {
        // Bin m/2 is unpaired on exactly one side of the conversion. Going
        // down it becomes the new Nyquist bin, which carries a whole cosine
        // rather than half of a conjugate pair, so it doubles; going up it
        // stops being Nyquist and splits back into a pair, so it halves.
        let idx = m / 2;
        if idx < out_spec.len() {
            out_spec[idx] *= if num < n { 2.0 } else { 0.5 };
        }
    }
    if num % 2 == 0 {
        // A real signal carries no imaginary part at Nyquist; the inverse real
        // transform requires that explicitly.
        let last = out_spec.len() - 1;
        out_spec[last] = Complex::new(out_spec[last].re, 0.0);
    }
    out_spec[0] = Complex::new(out_spec[0].re, 0.0);

    let inv = planner.plan_fft_inverse(num);
    let mut out = inv.make_output_vec();
    inv.process(&mut out_spec, &mut out)
        .expect("inverse real FFT");

    let scale = 1.0 / n as f64;
    out.iter().map(|v| v * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn hilbert_envelope_of_a_pure_tone_is_its_amplitude() {
        let n = 4096;
        let amp = 0.7;
        let x: Vec<f64> = (0..n)
            .map(|i| amp * (2.0 * PI * 64.0 * i as f64 / n as f64).sin())
            .collect();
        let env = hilbert_envelope(&x);
        // Endpoints ring; check the interior.
        for e in &env[64..n - 64] {
            assert_relative_eq!(e, &amp, max_relative = 1e-9);
        }
    }

    #[test]
    fn hilbert_envelope_tracks_amplitude_modulation() {
        let n = 8192;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (1.0 + 0.5 * (2.0 * PI * 4.0 * t).sin()) * (2.0 * PI * 512.0 * t).sin()
            })
            .collect();
        let env = hilbert_envelope(&x);
        for (i, e) in env.iter().enumerate().take(n - 256).skip(256) {
            let t = i as f64 / n as f64;
            let want = 1.0 + 0.5 * (2.0 * PI * 4.0 * t).sin();
            assert_relative_eq!(e, &want, max_relative = 5e-3);
        }
    }

    #[test]
    fn resample_is_identity_for_an_unchanged_length() {
        let x: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
        assert_eq!(resample(&x, 64), x);
    }

    #[test]
    fn resample_preserves_a_band_limited_tone_when_upsampling() {
        let n = 256;
        let f = 8.0;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / n as f64).sin())
            .collect();
        let up = resample(&x, 2 * n);
        for (i, v) in up.iter().enumerate() {
            let want = (2.0 * PI * f * i as f64 / (2 * n) as f64).sin();
            assert_relative_eq!(v, &want, epsilon = 1e-9);
        }
    }

    #[test]
    fn resample_preserves_a_band_limited_tone_when_downsampling() {
        let n = 512;
        let f = 5.0;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / n as f64).cos())
            .collect();
        let down = resample(&x, n / 2);
        for (i, v) in down.iter().enumerate() {
            let want = (2.0 * PI * f * i as f64 / (n / 2) as f64).cos();
            assert_relative_eq!(v, &want, epsilon = 1e-9);
        }
    }

    #[test]
    fn resample_preserves_the_mean() {
        let n = 300;
        let x: Vec<f64> = (0..n).map(|i| 2.0 + (i as f64 * 0.11).sin()).collect();
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert_relative_eq!(mean(&resample(&x, 450)), mean(&x), epsilon = 1e-10);
        assert_relative_eq!(mean(&resample(&x, 150)), mean(&x), epsilon = 1e-10);
    }

    #[test]
    fn resample_handles_the_48k_to_44k1_ratio_used_by_the_iso_corpus() {
        let n = 44100;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / n as f64).sin())
            .collect();
        let y = resample(&x, 48000);
        assert_eq!(y.len(), 48000);
        assert!(y.iter().all(|v| v.is_finite()));
        // RMS is preserved for a tone well inside the band.
        let rms = |v: &[f64]| (v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64).sqrt();
        assert_relative_eq!(rms(&y), rms(&x), max_relative = 1e-6);
    }
}
