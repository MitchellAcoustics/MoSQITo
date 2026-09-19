//! Third-octave band level spectrogram at 0.5 ms (2 kHz) temporal
//! resolution, per ISO 532-1 section 6.3. Matches `_third_octave_levels.py`.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use super::square_and_smooth::square_and_smooth;
use super::tables::{
    CENTER_FREQ_NOMINAL, FILTER_GAIN, THIRD_OCTAVE_FILTER, THIRD_OCTAVE_FILTER_REF,
};
use crate::dsp::sosfilt;

const N_BANDS: usize = 28;
/// `fs / 2000`, fixed since `fs` is required to be exactly 48 kHz.
const DEC_FACTOR: usize = 24;

/// Error from [`third_octave_levels`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThirdOctaveLevelsError {
    /// ISO 532-1 section 6.3 requires exactly 48 kHz (`_third_octave_levels.py:32-33`).
    WrongSampleRate { got: f64 },
}

impl std::fmt::Display for ThirdOctaveLevelsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThirdOctaveLevelsError::WrongSampleRate { got } => {
                write!(f, "sampling frequency shall be equal to 48 kHz, got {got}")
            }
        }
    }
}

impl std::error::Error for ThirdOctaveLevelsError {}

fn linspace_inclusive(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num <= 1 {
        return vec![start; num];
    }
    let step = (stop - start) / (num - 1) as f64;
    (0..num).map(|i| start + i as f64 * step).collect()
}

/// `(levels, time_axis, nominal_center_freqs)` — [`third_octave_levels`]'s
/// result.
type ThirdOctaveLevelsResult = (Array2<f64>, Vec<f64>, [f64; N_BANDS]);

/// Computes the 28-band third-octave level spectrogram of `sig`, sampled at
/// `fs` (must be exactly 48000 Hz).
///
/// Returns `(levels, time_axis, nominal_center_freqs)`: `levels` is (28,
/// n_time) in dB SPL, `time_axis` each frame's time in seconds
/// (`linspace(0, len(sig)/fs, n_time)`, matching
/// `_third_octave_levels.py:259` — note this is an *inclusive* endpoint,
/// unlike a plain arange), `nominal_center_freqs` the ANSI-preferred center
/// frequency of each band (unused by `loudness_zwtv` itself, which discards
/// it too — kept for API parity with MoSQITo's Python, which returns it).
pub fn third_octave_levels(
    sig: &[f64],
    fs: f64,
) -> Result<ThirdOctaveLevelsResult, ThirdOctaveLevelsError> {
    if fs != 48000.0 {
        return Err(ThirdOctaveLevelsError::WrongSampleRate { got: fs });
    }

    let n_time = sig.len().div_ceil(DEC_FACTOR);
    let time_axis = linspace_inclusive(0.0, sig.len() as f64 / fs, n_time);

    let tiny_value = 1e-12;
    let i_ref = 4e-10;

    let bands: Vec<Vec<f64>> = (0..N_BANDS)
        .into_par_iter()
        .map(|i| {
            let mut coeff = [[0.0f64; 6]; 3];
            for s in 0..3 {
                for k in 0..6 {
                    coeff[s][k] = THIRD_OCTAVE_FILTER_REF[s][k] - THIRD_OCTAVE_FILTER[i][s][k];
                }
            }
            let mut sig_filt = sosfilt(&coeff, sig);
            for v in sig_filt.iter_mut() {
                *v *= FILTER_GAIN[i];
            }

            let center_freq = 10f64.powf((i as f64 - 16.0) / 10.0) * 1000.0;
            let smoothed = square_and_smooth(&sig_filt, center_freq, fs);

            (0..n_time)
                .map(|t| 10.0 * ((smoothed[t * DEC_FACTOR] + tiny_value) / i_ref).log10())
                .collect()
        })
        .collect();

    let mut out = Array2::<f64>::zeros((N_BANDS, n_time));
    for (i, row) in bands.into_iter().enumerate() {
        out.row_mut(i).assign(&Array1::from(row));
    }

    Ok((out, time_axis, CENTER_FREQ_NOMINAL))
}
