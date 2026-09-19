//! Test-signal generators, matching `mosqito.utils.{sine_wave,am_sine,
//! am_noise,fm_sine}_generator`.
//!
//! `am_noise_generator`'s Python draws its noise carrier from
//! `numpy.random.default_rng()` — freshly seeded from OS entropy on every
//! call, so even MoSQITo's own output is not reproducible run to run. This
//! port takes an explicit `seed` instead of reaching for system entropy, so a
//! caller gets a reproducible signal; matching NumPy's PCG64 bit-for-bit is
//! not attempted (no standard requires it), so the two implementations'
//! Gaussian carriers diverge past their shared statistical properties. See
//! `DEVIATIONS.md`.

use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

const P_REF: f64 = 20e-6;

/// Population standard deviation (`numpy.std`'s default, `ddof=0`).
fn population_std(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    (x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n).sqrt()
}

/// Scales `y` in place so its population standard deviation matches the RMS
/// pressure of `spl_level` dB SPL, matching each generator's shared
/// `A_rms = P_REF * 10**(spl_level/20); y *= A_rms / y.std()` tail.
fn normalize_to_spl(y: &mut [f64], spl_level: f64) {
    let a_rms = P_REF * 10f64.powf(spl_level / 20.0);
    let scale = a_rms / population_std(y);
    for v in y.iter_mut() {
        *v *= scale;
    }
}

/// Generates a sine wave at `spl_level` dB SPL, matching
/// `sine_wave_generator(fs, d, freq, spl_level)`.
///
/// Returns `(signal, time)` in Pa / seconds.
pub fn sine_wave_generator(fs: f64, d: f64, freq: f64, spl_level: f64) -> (Vec<f64>, Vec<f64>) {
    let p_ref = 2e-5;
    let pressure_rms = p_ref * 10f64.powf(spl_level / 20.0);
    let amplitude = 2f64.sqrt() * pressure_rms;

    let n = (d / (1.0 / fs)).ceil() as usize;
    let time: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let signal: Vec<f64> = time
        .iter()
        .map(|&t| amplitude * (2.0 * std::f64::consts::PI * freq * t).sin())
        .collect();
    (signal, time)
}

/// Amplitude-modulates a unit-amplitude sine carrier at `fc` Hz by `xmod`,
/// normalised to `spl_level` dB SPL, matching
/// `am_sine_generator(xmod, fs, fc, spl_level)`.
///
/// Returns `(signal, modulation_index)`.
///
/// # Panics
/// Panics if `fc >= fs / 2`.
pub fn am_sine_generator(xmod: &[f64], fs: f64, fc: f64, spl_level: f64) -> (Vec<f64>, f64) {
    assert!(fc < fs / 2.0, "carrier frequency must be less than fs/2");

    let mut y_am: Vec<f64> = xmod
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let t = i as f64 / fs;
            (1.0 + x) * (2.0 * std::f64::consts::PI * fc * t).sin()
        })
        .collect();

    let m = xmod.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));

    normalize_to_spl(&mut y_am, spl_level);

    (y_am, m)
}

/// Amplitude-modulates a Gaussian broadband noise carrier by `xmod`,
/// normalised to `spl_level` dB SPL, matching
/// `am_noise_generator(xmod, spl_level)` — with an explicit `seed` in place
/// of Python's OS-entropy-seeded RNG; see this module's doc comment.
///
/// Returns `(signal, modulation_index)`.
pub fn am_noise_generator(xmod: &[f64], spl_level: f64, seed: u64) -> (Vec<f64>, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut y_am: Vec<f64> = xmod
        .iter()
        .map(|&x| {
            let carrier: f64 = StandardNormal.sample(&mut rng);
            (1.0 + x) * carrier
        })
        .collect();

    let m = xmod.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));

    normalize_to_spl(&mut y_am, spl_level);

    (y_am, m)
}

/// Frequency-modulates a unit-amplitude sine carrier at `fc` Hz by `xmod`
/// with sensitivity `k`, normalised to `spl_level` dB SPL, matching
/// `fm_sine_generator(xmod, fs, fc, k, spl_level)`.
///
/// Returns `(signal, instantaneous_frequency, max_freq_deviation,
/// modulation_index)`.
///
/// # Panics
/// Panics if `fc >= fs / 2`.
pub fn fm_sine_generator(
    xmod: &[f64],
    fs: f64,
    fc: f64,
    k: f64,
    spl_level: f64,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    assert!(fc < fs / 2.0, "carrier frequency must be less than fs/2");

    let dt = 1.0 / fs;
    let inst_freq: Vec<f64> = xmod.iter().map(|&x| fc + k * x).collect();

    let mut cum_inst_freq = 0.0f64;
    let mut y_fm = Vec::with_capacity(xmod.len());
    let mut cum_xmod = 0.0f64;
    let mut m = 0.0f64;
    for (&f, &x) in inst_freq.iter().zip(xmod) {
        cum_inst_freq += f;
        y_fm.push((2.0 * std::f64::consts::PI * cum_inst_freq * dt).sin());
        cum_xmod += x;
        m = m.max((2.0 * std::f64::consts::PI * k * cum_xmod * dt).abs());
    }

    let f_delta = k * xmod.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));

    normalize_to_spl(&mut y_fm, spl_level);

    (y_fm, inst_freq, f_delta, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sine_wave_generator_reaches_the_requested_rms_level() {
        let (sig, time) = sine_wave_generator(48000.0, 1.0, 100.0, 60.0);
        assert_eq!(sig.len(), time.len());
        let rms = (sig.iter().map(|v| v * v).sum::<f64>() / sig.len() as f64).sqrt();
        let want_rms = 2e-5 * 10f64.powf(60.0 / 20.0);
        assert_relative_eq!(rms, want_rms, max_relative = 1e-9);
    }

    #[test]
    fn am_sine_generator_rejects_a_carrier_at_or_above_nyquist() {
        let xmod = vec![0.0; 10];
        let result = std::panic::catch_unwind(|| am_sine_generator(&xmod, 100.0, 50.0, 60.0));
        assert!(result.is_err());
    }

    #[test]
    fn am_sine_generator_modulation_index_is_the_peak_of_xmod() {
        let xmod = vec![0.1, -0.5, 0.3, 0.2];
        let (_, m) = am_sine_generator(&xmod, 48000.0, 100.0, 60.0);
        assert_relative_eq!(m, 0.5, max_relative = 1e-12);
    }

    #[test]
    fn am_noise_generator_is_reproducible_for_a_fixed_seed() {
        let xmod = vec![0.0; 100];
        let (a, _) = am_noise_generator(&xmod, 60.0, 42);
        let (b, _) = am_noise_generator(&xmod, 60.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn am_noise_generator_reaches_the_requested_std_level() {
        // The generator scales by `A_rms / population_std(y)`, which sets
        // the population *standard deviation* exactly — not the plain RMS,
        // which is only equal to it when the mean happens to be zero (not
        // the case for a finite noise sample).
        let xmod = vec![0.0; 20000];
        let (sig, _) = am_noise_generator(&xmod, 60.0, 7);
        let want_std = 20e-6 * 10f64.powf(60.0 / 20.0);
        assert_relative_eq!(population_std(&sig), want_std, max_relative = 1e-9);
    }

    #[test]
    fn fm_sine_generator_instantaneous_frequency_tracks_xmod() {
        let xmod = vec![1.0, -1.0, 0.5];
        let (_, inst_freq, f_delta, _) = fm_sine_generator(&xmod, 48000.0, 100.0, 20.0, 60.0);
        assert_relative_eq!(inst_freq[0], 120.0, max_relative = 1e-12);
        assert_relative_eq!(inst_freq[1], 80.0, max_relative = 1e-12);
        assert_relative_eq!(f_delta, 20.0, max_relative = 1e-12);
    }
}
