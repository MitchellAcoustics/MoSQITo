//! Windowed one-sided FFT spectrum, matching `mosqito.sound_level_meter.comp_spectrum`.
//!
//! Every caller in the codebase uses `nfft='default'` (the full signal length)
//! and `one_sided=True`, so only that combination is implemented — the same
//! scope-narrowing this port applies elsewhere to parameter combinations
//! nothing exercises.
//!
//! Python's `comp_spectrum` returns the *complex* spectrum unless `db=True`,
//! in which case it returns `20*log10(abs(spectrum)/2e-5)`. Rather than one
//! function polymorphic on a `db` flag, this is two: [`comp_spectrum_complex`]
//! (the `db=False` case — the only real caller, `roughness_dw`, needs the
//! complex value itself, phase included, to reconstruct a time-domain
//! excitation later) and [`comp_spectrum_db`] (the `db=True` case every other
//! caller uses).

use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Window applied before the FFT, matching `comp_spectrum`'s `window` string
/// argument (`'hanning'` is the Python default; `roughness_dw` and `sii_ansi`
/// pass `'blackman'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumWindow {
    Hanning,
    Blackman,
}

/// Computes the one-sided windowed complex FFT spectrum of `signal`, matching
/// `comp_spectrum(signal, fs, nfft='default', window=.., one_sided=True,
/// db=False)`.
///
/// `signal` is (nperseg, nseg): axis 0 samples, axis 1 one column per segment
/// (a single-column view for a 1-D signal). Returns `(spectrum, freq_axis)`:
/// `spectrum` is (`nperseg/2`, nseg) in Pa, `freq_axis` has `nperseg/2` entries.
///
/// # A reproduced indexing quirk
/// Python's `freq_axis` labels the returned bins `df, 2·df, ..., (nfft/2)·df`
/// while `spectrum` itself is `fft(..)[0 .. nfft/2]` — bins `0..nfft/2-1`,
/// which *includes* the DC bin (index 0). The DC bin therefore comes back
/// mislabelled as `df` rather than `0`, and the bin at exactly `nfft/2`
/// (Nyquist, for even `nfft`) is silently dropped. `comp_spectrum` is a
/// generic FFT utility rather than a clause of a specific standard, and every
/// downstream metric (`roughness_dw`, SII, TNR/PR) was validated against real
/// MoSQITo output produced through this exact indexing — reproduced as-is,
/// not "fixed", per `DEVIATIONS.md`.
pub fn comp_spectrum_complex(
    signal: ArrayView2<f64>,
    fs: f64,
    window: SpectrumWindow,
) -> (Array2<Complex64>, Vec<f64>) {
    let nfft = signal.nrows();
    let nseg = signal.ncols();
    let half = nfft / 2;

    let win = normalized_window(window, nfft);
    let freq_axis: Vec<f64> = (1..=half).map(|k| k as f64 * (fs / nfft as f64)).collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);

    let mut spectrum = Array2::<Complex64>::from_elem((half, nseg), Complex64::new(0.0, 0.0));
    for col in 0..nseg {
        let mut buf: Vec<Complex64> = (0..nfft)
            .map(|i| Complex64::new(signal[[i, col]] * win[i], 0.0))
            .collect();
        fft.process(&mut buf);
        for k in 0..half {
            spectrum[[k, col]] = buf[k] * 1.42;
        }
    }

    (spectrum, freq_axis)
}

/// Computes the one-sided windowed dB spectrum of `signal` (re. 2e-5 Pa),
/// matching `comp_spectrum(signal, fs, nfft='default', window=.., one_sided=True,
/// db=True)`. See [`comp_spectrum_complex`] for the shape/indexing details
/// this shares.
pub fn comp_spectrum_db(
    signal: ArrayView2<f64>,
    fs: f64,
    window: SpectrumWindow,
) -> (Array2<f64>, Vec<f64>) {
    let (complex, freq_axis) = comp_spectrum_complex(signal, fs, window);
    let half = complex.nrows();
    let nseg = complex.ncols();

    let mut spectrum = Array2::<f64>::zeros((half, nseg));
    for col in 0..nseg {
        let amps: Vec<f64> = (0..half).map(|k| complex[[k, col]].norm()).collect();
        let db = crate::utils::amp2db(&amps, 2e-5);
        for (k, v) in db.into_iter().enumerate() {
            spectrum[[k, col]] = v;
        }
    }

    (spectrum, freq_axis)
}

fn normalized_window(window: SpectrumWindow, nfft: usize) -> Vec<f64> {
    let win = match window {
        SpectrumWindow::Hanning => crate::dsp::hanning(nfft),
        SpectrumWindow::Blackman => crate::dsp::blackman(nfft),
    };
    let win_sum: f64 = win.iter().sum();
    win.iter().map(|&w| w / win_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;
    use std::f64::consts::PI;

    #[test]
    fn a_pure_tone_peaks_near_its_own_frequency_bin() {
        let fs = 48000.0;
        let n = 4800;
        let f = 1000.0;
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f * i as f64 / fs).sin())
            .collect();
        let sig2d = Array2::from_shape_vec((n, 1), sig).unwrap();
        let (spec, freqs) = comp_spectrum_complex(sig2d.view(), fs, SpectrumWindow::Hanning);

        let peak_bin = (0..spec.nrows())
            .max_by(|&a, &b| spec[[a, 0]].norm().total_cmp(&spec[[b, 0]].norm()))
            .unwrap();
        assert_relative_eq!(freqs[peak_bin], f, max_relative = 0.01);
    }

    #[test]
    fn db_output_is_finite() {
        let n = 1024;
        let sig: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 1e-3).collect();
        let sig2d = Array2::from_shape_vec((n, 1), sig).unwrap();
        let (spec, _) = comp_spectrum_db(sig2d.view(), 48000.0, SpectrumWindow::Blackman);
        assert!(spec.iter().all(|v| v.is_finite()));
    }
}
