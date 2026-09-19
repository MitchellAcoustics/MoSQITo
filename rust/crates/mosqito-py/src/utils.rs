//! Small shared-utility bindings: time segmentation.

use mosqito_core::utils::time_segmentation as core_time_segmentation;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

type Blocks<'py> = Bound<'py, PyArray2<f64>>;
type Time<'py> = Bound<'py, PyArray1<f64>>;

/// Matches `time_segmentation(sig, fs, nperseg, noverlap)`'s `is_ecma=False`
/// case — the only one any real caller (`roughness_dw`, `tnr_ecma_perseg`,
/// `pr_ecma_perseg`) exercises; `mosqito_rs.time_segmentation` handles
/// rejecting `is_ecma=True`, matching this project's precedent of narrowing
/// to what's actually reachable (see `DEVIATIONS.md`).
#[pyfunction]
#[pyo3(name = "time_segmentation")]
#[pyo3(signature = (sig, fs, nperseg, noverlap))]
pub fn time_segmentation<'py>(
    py: Python<'py>,
    sig: PyReadonlyArray1<'py, f64>,
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
) -> PyResult<(Blocks<'py>, Time<'py>)> {
    let (blocks, time) = core_time_segmentation(sig.as_slice()?, fs, nperseg, noverlap);
    Ok((blocks.into_pyarray(py), Array1::from(time).into_pyarray(py)))
}
