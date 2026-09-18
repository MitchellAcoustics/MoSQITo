//! Top-level ISO 532-1 time-varying loudness entry point, matching
//! `loudness_zwtv.py`.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use super::nonlinear_decay::nl_loudness;
use super::temporal_weighting::temporal_weighting;
use super::third_octave_levels::{third_octave_levels, ThirdOctaveLevelsError};
use crate::dsp::resample_up_to;
use crate::loudness::zwst::{bark_axis, calc_slopes, main_loudness, FieldType};

/// Errors from [`loudness_zwtv`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoudnessZwtvError {
    ThirdOctaveLevels(ThirdOctaveLevelsError),
}

impl std::fmt::Display for LoudnessZwtvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoudnessZwtvError::ThirdOctaveLevels(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LoudnessZwtvError {}

/// `(N, N_specific, bark_axis, time_axis)` — [`loudness_zwtv`]'s result.
type LoudnessZwtvResult = (Vec<f64>, Array2<f64>, [f64; 240], Vec<f64>);

/// Computes time-varying loudness from a time signal, per ISO 532-1:2017
/// (Zwicker method). Matches `loudness_zwtv(signal, fs, field_type)`.
///
/// Resamples to 48 kHz first if `fs < 48000` (`loudness_zwtv.py:102-109`).
///
/// Returns `(N, N_specific, bark_axis, time_axis)` at the final 2 ms
/// temporal resolution (decimated by 4 from the 0.5 ms working resolution):
/// `N` in sone, `N_specific` in sone/Bark, one column per output time frame.
pub fn loudness_zwtv(
    signal: &[f64],
    fs: f64,
    field_type: FieldType,
) -> Result<LoudnessZwtvResult, LoudnessZwtvError> {
    let (signal, fs) = resample_up_to(signal, fs, 48000.0);

    let (spec_third, time_axis, _nominal_center_freqs) =
        third_octave_levels(&signal, fs).map_err(LoudnessZwtvError::ThirdOctaveLevels)?;
    let ntime = spec_third.ncols();

    // Core loudness per frame: `_main_loudness` has no cross-column
    // coupling (confirmed by inspection of its vectorised Python), so this
    // reuses the single-spectrum core validated by `loudness_zwst`'s golden
    // vectors, run in parallel across frames, rather than reimplementing its
    // vectorised form.
    let core_loudness_cols: Vec<[f64; 21]> = (0..ntime)
        .into_par_iter()
        .map(|t| main_loudness(&spec_third.column(t).to_vec(), field_type))
        .collect();
    let mut core_loudness = Array2::<f64>::zeros((21, ntime));
    for (t, col) in core_loudness_cols.into_iter().enumerate() {
        core_loudness
            .column_mut(t)
            .assign(&Array1::from(col.to_vec()));
    }

    let core_loudness = nl_loudness(&core_loudness);

    // Specific loudness per frame: likewise no cross-column coupling in
    // `_calc_slopes`, so this reuses the same per-frame core `loudness_zwst`
    // already validates, run in parallel.
    let results: Vec<(f64, [f64; 240])> = (0..ntime)
        .into_par_iter()
        .map(|t| {
            let mut nm = [0.0f64; 21];
            for row in 0..21 {
                nm[row] = core_loudness[[row, t]];
            }
            calc_slopes(&nm)
        })
        .collect();

    let loudness: Vec<f64> = results.iter().map(|&(n, _)| n).collect();
    let mut spec_loudness = Array2::<f64>::zeros((240, ntime));
    for (t, (_, spec)) in results.into_iter().enumerate() {
        spec_loudness
            .column_mut(t)
            .assign(&Array1::from(spec.to_vec()));
    }

    let filt_loudness = temporal_weighting(&loudness);

    // Decimation from 0.5 ms temporal resolution to 2 ms: `[::4]`.
    const DEC_FACTOR: usize = 4;
    let n: Vec<f64> = filt_loudness.iter().step_by(DEC_FACTOR).copied().collect();
    let n_out = n.len();
    let mut n_specific = Array2::<f64>::zeros((240, n_out));
    for (j, t) in (0..ntime).step_by(DEC_FACTOR).enumerate() {
        n_specific.column_mut(j).assign(&spec_loudness.column(t));
    }
    let time_axis: Vec<f64> = time_axis.iter().step_by(DEC_FACTOR).copied().collect();

    Ok((n, n_specific, bark_axis(), time_axis))
}
