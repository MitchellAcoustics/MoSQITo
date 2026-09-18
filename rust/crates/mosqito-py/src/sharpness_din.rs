//! DIN 45692:2009 sharpness bindings.
//!
//! `sharpness_din_from_loudness`'s Python signature accepts `N` as either a
//! scalar or an array (dispatching on that shape to two different masking
//! behaviours — see `mosqito-core`'s doc comments); that dispatch is handled
//! in `python/mosqito_rs/sharpness_din.py`, which calls one of the two
//! functions below depending on `N`'s shape. Keeping that logic in Python
//! avoids a PyO3-side `Either`-style union type for no benefit, matching how
//! `loudness_zwst`'s bindings keep MoSQITo's own signature fidelity one
//! layer up.

use mosqito_core::loudness::zwst::FieldType;
use mosqito_core::sharpness::din::{self as core_din, Weighting};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_field_type(s: &str) -> PyResult<FieldType> {
    match s {
        "free" => Ok(FieldType::Free),
        "diffuse" => Ok(FieldType::Diffuse),
        other => Err(PyValueError::new_err(format!(
            "field_type must be 'free' or 'diffuse', got {other:?}"
        ))),
    }
}

fn parse_weighting(s: &str) -> PyResult<Weighting> {
    Weighting::parse(s).map_err(PyValueError::new_err)
}

/// The `S.size == 1` branch of `sharpness_din_from_loudness`.
#[pyfunction]
#[pyo3(name = "sharpness_din_from_loudness_scalar")]
#[pyo3(signature = (n, n_specific, weighting))]
pub fn sharpness_din_from_loudness_scalar(
    n: f64,
    n_specific: PyReadonlyArray1<'_, f64>,
    weighting: &str,
) -> PyResult<f64> {
    let w = parse_weighting(weighting)?;
    let slice = n_specific.as_slice()?;
    if slice.len() != 240 {
        return Err(PyValueError::new_err(format!(
            "N_specific must have 240 bark bands, got {}",
            slice.len()
        )));
    }
    let mut arr = [0.0f64; 240];
    arr.copy_from_slice(slice);
    Ok(core_din::sharpness_din_from_loudness(n, &arr, w))
}

/// The segmented (`N < 0.1` masking) branch of `sharpness_din_from_loudness`.
#[pyfunction]
#[pyo3(name = "sharpness_din_from_loudness_segmented")]
#[pyo3(signature = (n, n_specific, weighting))]
pub fn sharpness_din_from_loudness_segmented<'py>(
    py: Python<'py>,
    n: PyReadonlyArray1<'py, f64>,
    n_specific: PyReadonlyArray2<'py, f64>,
    weighting: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let w = parse_weighting(weighting)?;
    let n_slice = n.as_slice()?;
    let n_specific = n_specific.as_array().to_owned();
    let s = core_din::sharpness_din_from_loudness_segmented(n_slice, &n_specific, w);
    Ok(Array1::from(s).into_pyarray(py))
}

/// Matches `sharpness_din_st(signal, fs, weighting, field_type)`.
#[pyfunction]
#[pyo3(name = "sharpness_din_st")]
#[pyo3(signature = (signal, fs, weighting, field_type))]
pub fn sharpness_din_st(
    signal: PyReadonlyArray1<'_, f64>,
    fs: f64,
    weighting: &str,
    field_type: &str,
) -> PyResult<f64> {
    let w = parse_weighting(weighting)?;
    let ft = parse_field_type(field_type)?;
    Ok(core_din::sharpness_din_st(signal.as_slice()?, fs, w, ft))
}

/// Matches `sharpness_din_freq(spectrum, freqs, weighting, field_type)` for a
/// 1-D spectrum.
#[pyfunction]
#[pyo3(name = "sharpness_din_freq")]
#[pyo3(signature = (spectrum, freqs, weighting, field_type))]
pub fn sharpness_din_freq(
    spectrum: PyReadonlyArray1<'_, f64>,
    freqs: PyReadonlyArray1<'_, f64>,
    weighting: &str,
    field_type: &str,
) -> PyResult<f64> {
    let w = parse_weighting(weighting)?;
    let ft = parse_field_type(field_type)?;
    Ok(core_din::sharpness_din_freq(
        spectrum.as_slice()?,
        freqs.as_slice()?,
        w,
        ft,
    ))
}

type PersegResult<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Matches `sharpness_din_perseg(signal, fs, weighting, nperseg, noverlap,
/// field_type)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "sharpness_din_perseg")]
#[pyo3(signature = (signal, fs, nperseg, noverlap, weighting, field_type))]
pub fn sharpness_din_perseg<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
    weighting: &str,
    field_type: &str,
) -> PyResult<PersegResult<'py>> {
    let w = parse_weighting(weighting)?;
    let ft = parse_field_type(field_type)?;
    let (s, time) =
        core_din::sharpness_din_perseg(signal.as_slice()?, fs, nperseg, noverlap, w, ft)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        Array1::from(s).into_pyarray(py),
        Array1::from(time).into_pyarray(py),
    ))
}
