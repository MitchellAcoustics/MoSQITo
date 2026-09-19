//! ECMA-418-2 stationary roughness bindings.

use mosqito_core::roughness::ecma::roughness_ecma as core_roughness_ecma;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

type RoughnessEcmaResult<'py> = (
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Matches `roughness_ecma(signal, fs)`.
#[pyfunction]
#[pyo3(name = "roughness_ecma")]
#[pyo3(signature = (signal, fs))]
pub fn roughness_ecma<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
) -> PyResult<RoughnessEcmaResult<'py>> {
    let (r, r_time, r_spec, bark, time_axis) = core_roughness_ecma(signal.as_slice()?, fs);
    Ok((
        r,
        Array1::from(r_time).into_pyarray(py),
        Array1::from(r_spec.to_vec()).into_pyarray(py),
        Array1::from(bark.to_vec()).into_pyarray(py),
        Array1::from(time_axis).into_pyarray(py),
    ))
}
