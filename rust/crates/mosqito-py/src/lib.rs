//! Python bindings for `mosqito-core`.
//!
//! This layer is deliberately thin: it converts numpy arrays to slices, calls
//! into `mosqito-core`, and converts the results back. All algorithmic work
//! lives in the core crate so it stays usable from Rust without Python.

use pyo3::prelude::*;

/// The version of the underlying `mosqito-core` crate.
#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    Ok(())
}
