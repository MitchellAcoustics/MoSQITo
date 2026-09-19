//! Test-signal generator bindings.

use mosqito_core::generators as core_gen;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

type Signal<'py> = Bound<'py, PyArray1<f64>>;
type Time<'py> = Bound<'py, PyArray1<f64>>;

/// Matches `sine_wave_generator(fs, d, freq, spl_level)`.
#[pyfunction]
#[pyo3(name = "sine_wave_generator")]
#[pyo3(signature = (fs, d, freq, spl_level))]
pub fn sine_wave_generator(
    py: Python<'_>,
    fs: f64,
    d: f64,
    freq: f64,
    spl_level: f64,
) -> PyResult<(Signal<'_>, Time<'_>)> {
    let (signal, time) = core_gen::sine_wave_generator(fs, d, freq, spl_level);
    Ok((
        Array1::from(signal).into_pyarray(py),
        Array1::from(time).into_pyarray(py),
    ))
}

/// Matches `am_sine_generator(xmod, fs, fc, spl_level)`.
#[pyfunction]
#[pyo3(name = "am_sine_generator")]
#[pyo3(signature = (xmod, fs, fc, spl_level))]
pub fn am_sine_generator<'py>(
    py: Python<'py>,
    xmod: PyReadonlyArray1<'py, f64>,
    fs: f64,
    fc: f64,
    spl_level: f64,
) -> PyResult<(Signal<'py>, f64)> {
    let (y, m) = core_gen::am_sine_generator(xmod.as_slice()?, fs, fc, spl_level);
    Ok((Array1::from(y).into_pyarray(py), m))
}

/// Matches `am_noise_generator(xmod, spl_level)`, with an explicit `seed` in
/// place of Python's OS-entropy-seeded RNG (see `mosqito_rs.am_noise_generator`'s
/// docstring and `DEVIATIONS.md`).
#[pyfunction]
#[pyo3(name = "am_noise_generator")]
#[pyo3(signature = (xmod, spl_level, seed))]
pub fn am_noise_generator<'py>(
    py: Python<'py>,
    xmod: PyReadonlyArray1<'py, f64>,
    spl_level: f64,
    seed: u64,
) -> PyResult<(Signal<'py>, f64)> {
    let (y, m) = core_gen::am_noise_generator(xmod.as_slice()?, spl_level, seed);
    Ok((Array1::from(y).into_pyarray(py), m))
}

type FmResult<'py> = (Signal<'py>, Signal<'py>, f64, f64);

/// Matches `fm_sine_generator(xmod, fs, fc, k, spl_level)`.
#[pyfunction]
#[pyo3(name = "fm_sine_generator")]
#[pyo3(signature = (xmod, fs, fc, k, spl_level))]
pub fn fm_sine_generator<'py>(
    py: Python<'py>,
    xmod: PyReadonlyArray1<'py, f64>,
    fs: f64,
    fc: f64,
    k: f64,
    spl_level: f64,
) -> PyResult<FmResult<'py>> {
    let (y, inst_freq, f_delta, m) =
        core_gen::fm_sine_generator(xmod.as_slice()?, fs, fc, k, spl_level);
    Ok((
        Array1::from(y).into_pyarray(py),
        Array1::from(inst_freq).into_pyarray(py),
        f_delta,
        m,
    ))
}
