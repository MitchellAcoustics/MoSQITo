//! Python bindings for `mosqito-core`.
//!
//! This layer is deliberately thin: it converts numpy arrays to slices, calls
//! into `mosqito-core`, and converts the results back. All algorithmic work
//! lives in the core crate so it stays usable from Rust without Python.
//!
//! Fidelity to MoSQITo's exact public signatures (accepting a 1-D signal
//! where MoSQITo's Python does, `squeeze`-ing results the same way, matching
//! keyword-argument names and casing) is handled one layer up, in
//! `python/mosqito_rs/`, not here.

use pyo3::prelude::*;

mod loudness_zwst;
mod sharpness_din;
mod slm;

/// The version of the underlying `mosqito-core` crate.
#[pyfunction]
fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(slm::noct_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(slm::noct_synthesis, m)?)?;
    m.add_function(wrap_pyfunction!(loudness_zwst::loudness_zwst, m)?)?;
    m.add_function(wrap_pyfunction!(loudness_zwst::loudness_zwst_freq, m)?)?;
    m.add_function(wrap_pyfunction!(loudness_zwst::loudness_zwst_perseg, m)?)?;
    m.add_function(wrap_pyfunction!(
        sharpness_din::sharpness_din_from_loudness_scalar,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sharpness_din::sharpness_din_from_loudness_segmented,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sharpness_din::sharpness_din_st, m)?)?;
    m.add_function(wrap_pyfunction!(sharpness_din::sharpness_din_freq, m)?)?;
    m.add_function(wrap_pyfunction!(sharpness_din::sharpness_din_perseg, m)?)?;
    Ok(())
}
