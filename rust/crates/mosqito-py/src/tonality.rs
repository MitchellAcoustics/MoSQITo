//! Tonality (TNR/PR) bindings.
//!
//! Each binding returns the *unfiltered* result (every detected tone, not
//! just the prominent ones) for `_st`/`_freq`; the `prominence` boolean
//! mask MoSQITo's Python applies is resolved at the `python/mosqito_rs`
//! wrapper layer, the same pattern used for `sharpness_din_from_loudness`'s
//! scalar/array dispatch and SII's threshold dispatch. `_perseg`'s
//! `prominence` flag instead changes what gets written into the regridded
//! output (not a post-hoc mask), so it is threaded straight through to the
//! core function that needs it to build the grid correctly.

use mosqito_core::tonality::{
    pr_ecma_freq as core_pr_ecma_freq, pr_ecma_perseg as core_pr_ecma_perseg,
    pr_ecma_st as core_pr_ecma_st, tnr_ecma_freq as core_tnr_ecma_freq,
    tnr_ecma_perseg as core_tnr_ecma_perseg, tnr_ecma_st as core_tnr_ecma_st, PersegGrid, PrResult,
    TnrResult,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

type StResult<'py> = (
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray1<f64>>,
);

fn tnr_to_py(py: Python<'_>, r: TnrResult) -> StResult<'_> {
    let TnrResult {
        mut tones_freqs,
        mut tnr,
        mut prominence,
        t_tnr,
    } = r;
    (
        t_tnr[0],
        tnr.remove(0).into_pyarray(py),
        prominence.remove(0).into_pyarray(py),
        tones_freqs.remove(0).into_pyarray(py),
    )
}

fn pr_to_py(py: Python<'_>, r: PrResult) -> StResult<'_> {
    let PrResult {
        mut tones_freqs,
        mut pr,
        mut prominence,
        t_pr,
    } = r;
    (
        t_pr[0],
        pr.remove(0).into_pyarray(py),
        prominence.remove(0).into_pyarray(py),
        tones_freqs.remove(0).into_pyarray(py),
    )
}

/// Matches `tnr_ecma_st(signal, fs)` before the `prominence` filter.
#[pyfunction]
#[pyo3(name = "tnr_ecma_st")]
#[pyo3(signature = (signal, fs))]
pub fn tnr_ecma_st<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
) -> PyResult<StResult<'py>> {
    let result = core_tnr_ecma_st(signal.as_slice()?, fs);
    Ok(tnr_to_py(py, result))
}

/// Matches `tnr_ecma_freq(spectrum, freqs)` for a 1-D spectrum, before the
/// `prominence` filter. `spectrum` must already be a non-negative amplitude
/// spectrum (`abs()` applied at the wrapper layer).
#[pyfunction]
#[pyo3(name = "tnr_ecma_freq")]
#[pyo3(signature = (spectrum, freqs))]
pub fn tnr_ecma_freq<'py>(
    py: Python<'py>,
    spectrum: PyReadonlyArray1<'py, f64>,
    freqs: PyReadonlyArray1<'py, f64>,
) -> PyResult<StResult<'py>> {
    let result = core_tnr_ecma_freq(spectrum.as_slice()?, freqs.as_slice()?);
    Ok(tnr_to_py(py, result))
}

/// Matches `pr_ecma_st(signal, fs)` before the `prominence` filter.
#[pyfunction]
#[pyo3(name = "pr_ecma_st")]
#[pyo3(signature = (signal, fs))]
pub fn pr_ecma_st<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
) -> PyResult<StResult<'py>> {
    let result = core_pr_ecma_st(signal.as_slice()?, fs);
    Ok(pr_to_py(py, result))
}

/// Matches `pr_ecma_freq(spectrum, freqs)` for a 1-D spectrum, before the
/// `prominence` filter.
#[pyfunction]
#[pyo3(name = "pr_ecma_freq")]
#[pyo3(signature = (spectrum, freqs))]
pub fn pr_ecma_freq<'py>(
    py: Python<'py>,
    spectrum: PyReadonlyArray1<'py, f64>,
    freqs: PyReadonlyArray1<'py, f64>,
) -> PyResult<StResult<'py>> {
    let result = core_pr_ecma_freq(spectrum.as_slice()?, freqs.as_slice()?);
    Ok(pr_to_py(py, result))
}

type PersegResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<bool>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

fn grid_to_py(py: Python<'_>, g: PersegGrid) -> PersegResult<'_> {
    (
        g.t.into_pyarray(py),
        g.values.into_pyarray(py),
        g.prominence.into_pyarray(py),
        g.freqs.into_pyarray(py),
        g.time.into_pyarray(py),
    )
}

/// Matches `tnr_ecma_perseg(signal, fs, prominence, overlap)`'s 1-D-signal
/// branch.
#[pyfunction]
#[pyo3(name = "tnr_ecma_perseg")]
#[pyo3(signature = (signal, fs, overlap, prominence))]
pub fn tnr_ecma_perseg<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
    overlap: f64,
    prominence: bool,
) -> PyResult<PersegResult<'py>> {
    let grid = core_tnr_ecma_perseg(signal.as_slice()?, fs, overlap, prominence);
    Ok(grid_to_py(py, grid))
}

/// Matches `pr_ecma_perseg(signal, fs, prominence, overlap)`'s 1-D-signal
/// branch.
#[pyfunction]
#[pyo3(name = "pr_ecma_perseg")]
#[pyo3(signature = (signal, fs, overlap, prominence))]
pub fn pr_ecma_perseg<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
    overlap: f64,
    prominence: bool,
) -> PyResult<PersegResult<'py>> {
    let grid = core_pr_ecma_perseg(signal.as_slice()?, fs, overlap, prominence);
    Ok(grid_to_py(py, grid))
}
