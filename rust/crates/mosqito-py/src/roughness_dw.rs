//! Daniel & Weber roughness bindings.

use mosqito_core::roughness::dw::{
    roughness_dw as core_roughness_dw, roughness_dw_freq as core_roughness_dw_freq,
};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

type RoughnessDwResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Matches `roughness_dw(signal, fs, overlap)`.
#[pyfunction]
#[pyo3(name = "roughness_dw")]
#[pyo3(signature = (signal, fs, overlap))]
pub fn roughness_dw<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<'py, f64>,
    fs: f64,
    overlap: f64,
) -> PyResult<RoughnessDwResult<'py>> {
    let (r, r_spec, bark, time) = core_roughness_dw(signal.as_slice()?, fs, overlap);
    Ok((
        Array1::from(r).into_pyarray(py),
        r_spec.into_pyarray(py),
        Array1::from(bark.to_vec()).into_pyarray(py),
        Array1::from(time).into_pyarray(py),
    ))
}

type RoughnessDwFreqResult<'py> = (f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Matches `roughness_dw_freq(spectrum, freqs)` for a 1-D amplitude
/// spectrum.
#[pyfunction]
#[pyo3(name = "roughness_dw_freq")]
#[pyo3(signature = (spectrum, freqs))]
pub fn roughness_dw_freq<'py>(
    py: Python<'py>,
    spectrum: PyReadonlyArray1<'py, f64>,
    freqs: PyReadonlyArray1<'py, f64>,
) -> PyResult<RoughnessDwFreqResult<'py>> {
    let (r, r_spec, bark) = core_roughness_dw_freq(spectrum.as_slice()?, freqs.as_slice()?);
    Ok((
        r,
        Array1::from(r_spec.to_vec()).into_pyarray(py),
        Array1::from(bark.to_vec()).into_pyarray(py),
    ))
}
