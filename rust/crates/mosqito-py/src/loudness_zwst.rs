//! ISO 532-1 stationary loudness bindings.
//!
//! As with `slm::noct_spectrum`/`noct_synthesis`, these are an internal
//! implementation detail behind `mosqito_rs.loudness_zwst*`, which is what
//! needs to match MoSQITo's exact public signatures (default arguments,
//! `field_type` validation messages, etc.) — so these keep ordinary Rust
//! naming and positional calling.

use mosqito_core::loudness::zwst::{self as core_zwst, FieldType};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
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

type Loudness = f64;
type Specific<'py> = Bound<'py, PyArray1<f64>>;
type Bark<'py> = Bound<'py, PyArray1<f64>>;
type Time<'py> = Bound<'py, PyArray1<f64>>;
type PersegResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bark<'py>,
    Time<'py>,
);

/// Matches `loudness_zwst(signal, fs, field_type)` for a 1-D signal.
#[pyfunction]
#[pyo3(name = "loudness_zwst")]
#[pyo3(signature = (signal, fs, field_type))]
pub fn loudness_zwst<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
    field_type: &str,
) -> PyResult<(Loudness, Specific<'py>, Bark<'py>)> {
    let ft = parse_field_type(field_type)?;
    let (n, n_spec, bark) = core_zwst::loudness_zwst(signal.as_slice()?, fs, ft);
    Ok((
        n,
        Array1::from(n_spec.to_vec()).into_pyarray(py),
        Array1::from(bark.to_vec()).into_pyarray(py),
    ))
}

/// Matches `loudness_zwst_freq(spectrum, freqs, field_type)` for a 1-D
/// spectrum.
#[pyfunction]
#[pyo3(name = "loudness_zwst_freq")]
#[pyo3(signature = (spectrum, freqs, field_type))]
pub fn loudness_zwst_freq<'py>(
    py: Python<'py>,
    spectrum: PyReadonlyArray1<'py, f64>,
    freqs: PyReadonlyArray1<'py, f64>,
    field_type: &str,
) -> PyResult<(Loudness, Specific<'py>, Bark<'py>)> {
    let ft = parse_field_type(field_type)?;
    let (n, n_spec, bark) =
        core_zwst::loudness_zwst_freq(spectrum.as_slice()?, freqs.as_slice()?, ft);
    Ok((
        n,
        Array1::from(n_spec.to_vec()).into_pyarray(py),
        Array1::from(bark.to_vec()).into_pyarray(py),
    ))
}

/// Matches `loudness_zwst_perseg(signal, fs, nperseg, noverlap, field_type)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "loudness_zwst_perseg")]
#[pyo3(signature = (signal, fs, nperseg, noverlap, field_type))]
pub fn loudness_zwst_perseg<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
    field_type: &str,
) -> PyResult<PersegResult<'py>> {
    let ft = parse_field_type(field_type)?;
    let (n, n_spec, bark, time) =
        core_zwst::loudness_zwst_perseg(signal.as_slice()?, fs, nperseg, noverlap, ft)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        Array1::from(n).into_pyarray(py),
        n_spec.into_pyarray(py),
        Array1::from(bark.to_vec()).into_pyarray(py),
        Array1::from(time).into_pyarray(py),
    ))
}
