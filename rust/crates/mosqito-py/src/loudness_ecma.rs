//! ECMA-418-2 stationary loudness bindings.

use mosqito_core::loudness::ecma::loudness_ecma as core_loudness_ecma;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

type LoudnessEcmaResult<'py> = (
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Matches `loudness_ecma(signal, fs, sb, sh)` for scalar `sb`/`sh`.
#[pyfunction]
#[pyo3(name = "loudness_ecma")]
#[pyo3(signature = (signal, fs, sb, sh))]
pub fn loudness_ecma<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
    sb: usize,
    sh: usize,
) -> PyResult<LoudnessEcmaResult<'py>> {
    let (n, n_time, n_specific, bark, time_axis) =
        core_loudness_ecma(signal.as_slice()?, fs, sb, sh);
    Ok((
        n,
        Array1::from(n_time).into_pyarray(py),
        n_specific.into_pyarray(py),
        Array1::from(bark.to_vec()).into_pyarray(py),
        Array1::from(time_axis).into_pyarray(py),
    ))
}
