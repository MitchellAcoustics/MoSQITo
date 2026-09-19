//! ANSI S3.5 speech intelligibility index (SII) bindings.
//!
//! `threshold` is `None` / `'zwicker'` / an explicit array in MoSQITo's
//! Python. Rather than a PyO3-side `Either`-style union type, the dispatch
//! is resolved in `python/mosqito_rs/`, which calls these bindings with a
//! `use_zwicker_threshold` flag and an optional explicit array — the same
//! pattern `sharpness_din_from_loudness`'s scalar/array dispatch already
//! uses one layer up.

use mosqito_core::speech_intelligibility::{
    self as core_sii, SiiMethod, SiiThreshold, SpeechLevel,
};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_method(s: &str) -> PyResult<SiiMethod> {
    match s {
        "critical" => Ok(SiiMethod::Critical),
        "equally_critical" => Ok(SiiMethod::EquallyCritical),
        "third_octave" => Ok(SiiMethod::ThirdOctave),
        "octave" => Ok(SiiMethod::Octave),
        other => Err(PyValueError::new_err(format!(
            "Method should be within {{\"critical\", \"equally_critical\", \"third_octave\", \"octave\"}}, got {other:?}."
        ))),
    }
}

fn parse_speech_level(s: &str) -> PyResult<SpeechLevel> {
    match s {
        "normal" => Ok(SpeechLevel::Normal),
        "raised" => Ok(SpeechLevel::Raised),
        "loud" => Ok(SpeechLevel::Loud),
        "shout" => Ok(SpeechLevel::Shout),
        other => Err(PyValueError::new_err(format!(
            "Speech level should be within {{\"normal\", \"raised\", \"loud\", \"shout\"}}, got {other:?}."
        ))),
    }
}

fn resolve_threshold(use_zwicker: bool, custom: &Option<Vec<f64>>) -> SiiThreshold<'_> {
    if let Some(arr) = custom {
        SiiThreshold::Custom(arr)
    } else if use_zwicker {
        SiiThreshold::Zwicker
    } else {
        SiiThreshold::Zero
    }
}

type SiiPyResult<'py> = (f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Matches `sii_ansi(noise, fs, method, speech_level, threshold)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "sii_ansi")]
#[pyo3(signature = (noise, fs, method, speech_level, use_zwicker_threshold, custom_threshold))]
pub fn sii_ansi<'py>(
    py: Python<'py>,
    noise: PyReadonlyArray1<'py, f64>,
    fs: f64,
    method: &str,
    speech_level: &str,
    use_zwicker_threshold: bool,
    custom_threshold: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<SiiPyResult<'py>> {
    let method = parse_method(method)?;
    let speech_level = parse_speech_level(speech_level)?;
    let custom: Option<Vec<f64>> = custom_threshold.map(|a| a.as_slice().unwrap().to_vec());
    let threshold = resolve_threshold(use_zwicker_threshold, &custom);

    let (sii, sii_spec, freq_axis) =
        core_sii::sii_ansi(noise.as_slice()?, fs, method, speech_level, threshold);
    Ok((
        sii,
        Array1::from(sii_spec).into_pyarray(py),
        Array1::from(freq_axis).into_pyarray(py),
    ))
}

/// Matches `sii_ansi_freq(spectrum, freqs, method, speech_level, threshold)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "sii_ansi_freq")]
#[pyo3(signature = (spectrum, freqs, method, speech_level, use_zwicker_threshold, custom_threshold))]
pub fn sii_ansi_freq<'py>(
    py: Python<'py>,
    spectrum: PyReadonlyArray1<'py, f64>,
    freqs: PyReadonlyArray1<'py, f64>,
    method: &str,
    speech_level: &str,
    use_zwicker_threshold: bool,
    custom_threshold: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<SiiPyResult<'py>> {
    let method = parse_method(method)?;
    let speech_level = parse_speech_level(speech_level)?;
    let custom: Option<Vec<f64>> = custom_threshold.map(|a| a.as_slice().unwrap().to_vec());
    let threshold = resolve_threshold(use_zwicker_threshold, &custom);

    let (sii, sii_spec, freq_axis) = core_sii::sii_ansi_freq(
        spectrum.as_slice()?,
        freqs.as_slice()?,
        method,
        speech_level,
        threshold,
    );
    Ok((
        sii,
        Array1::from(sii_spec).into_pyarray(py),
        Array1::from(freq_axis).into_pyarray(py),
    ))
}

/// Matches `sii_ansi_level(noise_level, method, speech_level, threshold)`.
#[pyfunction]
#[pyo3(name = "sii_ansi_level")]
#[pyo3(signature = (noise_level, method, speech_level, use_zwicker_threshold, custom_threshold))]
pub fn sii_ansi_level<'py>(
    py: Python<'py>,
    noise_level: f64,
    method: &str,
    speech_level: &str,
    use_zwicker_threshold: bool,
    custom_threshold: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<SiiPyResult<'py>> {
    let method = parse_method(method)?;
    let speech_level = parse_speech_level(speech_level)?;
    let custom: Option<Vec<f64>> = custom_threshold.map(|a| a.as_slice().unwrap().to_vec());
    let threshold = resolve_threshold(use_zwicker_threshold, &custom);

    let (sii, sii_spec, freq_axis) =
        core_sii::sii_ansi_level(noise_level, method, speech_level, threshold);
    Ok((
        sii,
        Array1::from(sii_spec).into_pyarray(py),
        Array1::from(freq_axis).into_pyarray(py),
    ))
}
