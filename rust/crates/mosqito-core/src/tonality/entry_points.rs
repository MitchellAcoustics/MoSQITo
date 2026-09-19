//! `tnr_ecma_*`/`pr_ecma_*` entry points, built on [`super::tnr::tnr_main_calc`]
//! / [`super::pr::pr_main_calc`].
//!
//! `tnr_ecma_perseg`/`pr_ecma_perseg` only port the 1-D-signal branch of
//! their Python counterparts. The 2-D-signal branch
//! (`tnr_ecma_perseg.py:127`/`pr_ecma_perseg.py:129`) references an
//! undefined `sig` (should be `signal`) — a real `NameError` in MoSQITo,
//! confirmed by direct read, never exercised by any test in that repo. Not
//! ported; see `DEVIATIONS.md`.

use ndarray::Array2;

use super::pr::{pr_main_calc, PrResult};
use super::tnr::{tnr_main_calc, TnrResult};
use crate::dsp::nearest_index;
use crate::slm::{comp_spectrum_db, SpectrumWindow};
use crate::utils::{amp2db, time_segmentation};

/// Matches `tnr_ecma_st(signal, fs)` (before the `prominence` filter, which
/// the Python-wrapper layer applies — see `DEVIATIONS.md`'s note on where
/// this port resolves MoSQITo's boolean-flag dispatches).
pub fn tnr_ecma_st(signal: &[f64], fs: f64) -> TnrResult {
    let sig2d = Array2::from_shape_vec((signal.len(), 1), signal.to_vec()).unwrap();
    let (spec_db, freq_axis) = comp_spectrum_db(sig2d.view(), fs, SpectrumWindow::Hanning);
    let spectrum_db_by_seg = vec![spec_db.column(0).to_vec()];
    tnr_main_calc(&spectrum_db_by_seg, &freq_axis)
}

/// Matches `pr_ecma_st(signal, fs)`.
pub fn pr_ecma_st(signal: &[f64], fs: f64) -> PrResult {
    let sig2d = Array2::from_shape_vec((signal.len(), 1), signal.to_vec()).unwrap();
    let (spec_db, freq_axis) = comp_spectrum_db(sig2d.view(), fs, SpectrumWindow::Hanning);
    let spectrum_db_by_seg = vec![spec_db.column(0).to_vec()];
    pr_main_calc(&spectrum_db_by_seg, &freq_axis)
}

/// Matches `tnr_ecma_freq(spectrum, freqs)` for a 1-D spectrum. `spectrum`
/// must already be a non-negative amplitude spectrum (Python's
/// `abs(spectrum)` happens at the wrapper layer, same as
/// `roughness_dw_freq`).
pub fn tnr_ecma_freq(spectrum_amp: &[f64], freqs: &[f64]) -> TnrResult {
    let spectrum_db = amp2db(spectrum_amp, 2e-5);
    tnr_main_calc(&[spectrum_db], freqs)
}

/// Matches `pr_ecma_freq(spectrum, freqs)` for a 1-D spectrum.
pub fn pr_ecma_freq(spectrum_amp: &[f64], freqs: &[f64]) -> PrResult {
    let spectrum_db = amp2db(spectrum_amp, 2e-5);
    pr_main_calc(&[spectrum_db], freqs)
}

/// A time/frequency grid of TNR (or PR) values and their prominence flags,
/// matching `tnr_ecma_perseg`/`pr_ecma_perseg`'s `(tnr, promi)` regridding
/// of each segment's ragged tone list onto a shared 1000-point log
/// frequency axis.
pub struct PersegGrid {
    pub t: Vec<f64>,
    pub values: Array2<f64>,
    pub prominence: Array2<bool>,
    pub freqs: Vec<f64>,
    pub time: Vec<f64>,
}

const PERSEG_GRID_POINTS: usize = 1000;

fn regrid(
    tones_freqs: &[Vec<f64>],
    values_in: &[Vec<f64>],
    prom_in: &[Vec<bool>],
    t: Vec<f64>,
    time: Vec<f64>,
    prominence_only: bool,
) -> PersegGrid {
    let nseg = tones_freqs.len();
    let log_lo = 90f64.log10();
    let log_hi = 11200f64.log10();
    let freqs: Vec<f64> = (0..PERSEG_GRID_POINTS)
        .map(|i| {
            let frac = i as f64 / (PERSEG_GRID_POINTS - 1) as f64;
            10f64.powf(log_lo + frac * (log_hi - log_lo))
        })
        .collect();

    let mut values = Array2::<f64>::from_elem((PERSEG_GRID_POINTS, nseg), f64::NAN);
    let mut prominence = Array2::<bool>::from_elem((PERSEG_GRID_POINTS, nseg), false);

    for s in 0..nseg {
        for f in 0..tones_freqs[s].len() {
            let target = tones_freqs[s][f];
            let best_idx = nearest_index(&freqs, target);
            let is_prom = prom_in[s][f];
            if !prominence_only || is_prom {
                values[[best_idx, s]] = values_in[s][f];
                prominence[[best_idx, s]] = is_prom;
            }
        }
    }

    PersegGrid {
        t,
        values,
        prominence,
        freqs,
        time,
    }
}

/// Matches `tnr_ecma_perseg(signal, fs, prominence, overlap)`'s 1-D-signal
/// branch.
pub fn tnr_ecma_perseg(signal: &[f64], fs: f64, overlap: f64, prominence_only: bool) -> PersegGrid {
    let nperseg = (0.5 * fs) as usize;
    let noverlap = (overlap * nperseg as f64) as usize;
    let (segments, time) = time_segmentation(signal, fs, nperseg, Some(noverlap));
    let (spec_db, freq_axis) = comp_spectrum_db(segments.view(), fs, SpectrumWindow::Hanning);
    let nseg = spec_db.ncols();
    let spectrum_db_by_seg: Vec<Vec<f64>> = (0..nseg).map(|c| spec_db.column(c).to_vec()).collect();

    let result = tnr_main_calc(&spectrum_db_by_seg, &freq_axis);
    regrid(
        &result.tones_freqs,
        &result.tnr,
        &result.prominence,
        result.t_tnr,
        time,
        prominence_only,
    )
}

/// Matches `pr_ecma_perseg(signal, fs, prominence, overlap)`'s 1-D-signal
/// branch.
pub fn pr_ecma_perseg(signal: &[f64], fs: f64, overlap: f64, prominence_only: bool) -> PersegGrid {
    let nperseg = (0.5 * fs) as usize;
    let noverlap = (overlap * nperseg as f64) as usize;
    let (segments, time) = time_segmentation(signal, fs, nperseg, Some(noverlap));
    let (spec_db, freq_axis) = comp_spectrum_db(segments.view(), fs, SpectrumWindow::Hanning);
    let nseg = spec_db.ncols();
    let spectrum_db_by_seg: Vec<Vec<f64>> = (0..nseg).map(|c| spec_db.column(c).to_vec()).collect();

    let result = pr_main_calc(&spectrum_db_by_seg, &freq_axis);
    regrid(
        &result.tones_freqs,
        &result.pr,
        &result.prominence,
        result.t_pr,
        time,
        prominence_only,
    )
}
