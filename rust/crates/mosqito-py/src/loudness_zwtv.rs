//! ISO 532-1 time-varying loudness bindings.

use mosqito_core::loudness::zwst::FieldType;
use mosqito_core::loudness::zwtv::loudness_zwtv as core_loudness_zwtv;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2};
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

type LoudnessZwtvResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Matches `loudness_zwtv(signal, fs, field_type)`.
#[pyfunction]
#[pyo3(name = "loudness_zwtv")]
#[pyo3(signature = (signal, fs, field_type))]
pub fn loudness_zwtv<'py>(
    py: Python<'py>,
    signal: numpy::PyReadonlyArray1<'py, f64>,
    fs: f64,
    field_type: &str,
) -> PyResult<LoudnessZwtvResult<'py>> {
    let ft = parse_field_type(field_type)?;
    let (n, n_specific, bark, time) = core_loudness_zwtv(signal.as_slice()?, fs, ft)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        Array1::from(n).into_pyarray(py),
        n_specific.into_pyarray(py),
        Array1::from(bark.to_vec()).into_pyarray(py),
        Array1::from(time).into_pyarray(py),
    ))
}
