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

use mosqito_core::loudness::zwst::FieldType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod generators;
mod loudness_ecma;
mod loudness_zwst;
mod loudness_zwtv;
mod roughness_dw;
mod roughness_ecma;
mod sharpness_din;
mod slm;
mod speech_intelligibility;

/// Parses MoSQITo's `field_type` string argument, shared by every binding
/// that takes it (`loudness_zwst*`, `loudness_zwtv`, `sharpness_din*`).
fn parse_field_type(s: &str) -> PyResult<FieldType> {
    match s {
        "free" => Ok(FieldType::Free),
        "diffuse" => Ok(FieldType::Diffuse),
        other => Err(PyValueError::new_err(format!(
            "field_type must be 'free' or 'diffuse', got {other:?}"
        ))),
    }
}

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
    m.add_function(wrap_pyfunction!(loudness_zwtv::loudness_zwtv, m)?)?;
    m.add_function(wrap_pyfunction!(loudness_ecma::loudness_ecma, m)?)?;
    m.add_function(wrap_pyfunction!(roughness_ecma::roughness_ecma, m)?)?;
    m.add_function(wrap_pyfunction!(roughness_dw::roughness_dw, m)?)?;
    m.add_function(wrap_pyfunction!(roughness_dw::roughness_dw_freq, m)?)?;
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
    m.add_function(wrap_pyfunction!(sharpness_din::sharpness_din_tv, m)?)?;
    m.add_function(wrap_pyfunction!(speech_intelligibility::sii_ansi, m)?)?;
    m.add_function(wrap_pyfunction!(speech_intelligibility::sii_ansi_freq, m)?)?;
    m.add_function(wrap_pyfunction!(speech_intelligibility::sii_ansi_level, m)?)?;
    m.add_function(wrap_pyfunction!(generators::sine_wave_generator, m)?)?;
    m.add_function(wrap_pyfunction!(generators::am_sine_generator, m)?)?;
    m.add_function(wrap_pyfunction!(generators::am_noise_generator, m)?)?;
    m.add_function(wrap_pyfunction!(generators::fm_sine_generator, m)?)?;
    Ok(())
}
