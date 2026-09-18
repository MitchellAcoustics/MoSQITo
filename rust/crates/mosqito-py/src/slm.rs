//! Sound level meter bindings: n-th octave band analysis.

use mosqito_core::slm::noct as core_noct;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn map_noct_error(err: core_noct::NoctError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

// Every binding here returns `(band levels, band frequencies)`; these aliases
// name that pattern once instead of repeating the nested `Bound<'py, ...>`
// tuple at each function signature.
type Levels1D<'py> = Bound<'py, PyArray1<f64>>;
type Levels2D<'py> = Bound<'py, PyArray2<f64>>;
type Freqs<'py> = Bound<'py, PyArray1<f64>>;

/// Measures the RMS level of a signal in each n-th octave band. `sig` is
/// always 2-D here (samples, segments); `mosqito_rs.noct_spectrum` handles
/// accepting a 1-D signal and squeezing the result, matching
/// `mosqito.sound_level_meter.noct_spectrum`'s exact behaviour.
///
/// This binding is an internal implementation detail behind
/// `mosqito_rs.noct_spectrum`, which handles the 1-D-signal and `squeeze`
/// cases and is what needs to match MoSQITo's exact public signature
/// (including its capitalised `G` keyword) — so this one keeps ordinary
/// Rust naming and is always called positionally from Python.
// `sig, fs, fmin, fmax, n, g, fr` is MoSQITo's own noct_spectrum signature;
// splitting these into a params struct would just move the complexity rather
// than reduce it, for a signature that isn't ours to shorten.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "noct_spectrum")]
#[pyo3(signature = (sig, fs, fmin, fmax, n=3, g=10, fr=1000.0))]
pub fn noct_spectrum<'py>(
    py: Python<'py>,
    sig: PyReadonlyArray2<'py, f64>,
    fs: f64,
    fmin: f64,
    fmax: f64,
    n: u32,
    g: u32,
    fr: f64,
) -> PyResult<(Levels2D<'py>, Freqs<'py>)> {
    let (spec, freq) = core_noct::noct_spectrum(sig.as_array(), fs, fmin, fmax, n, g, fr)
        .map_err(map_noct_error)?;
    Ok((spec.into_pyarray(py), Array1::from(freq).into_pyarray(py)))
}

/// Converts a 1-D frequency spectrum to n-th octave band levels. Matches
/// `mosqito.sound_level_meter.noct_synthesis` for a 1-D spectrum; the 2-D
/// case (per-segment spectra) is not yet ported — see `mosqito-core`'s
/// `slm::noct` module docs. See [`noct_spectrum`] on this binding's naming.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "noct_synthesis")]
#[pyo3(signature = (spectrum, freqs, fmin, fmax, n=3, g=10, fr=1000.0))]
pub fn noct_synthesis<'py>(
    py: Python<'py>,
    spectrum: PyReadonlyArray1<'py, f64>,
    freqs: PyReadonlyArray1<'py, f64>,
    fmin: f64,
    fmax: f64,
    n: u32,
    g: u32,
    fr: f64,
) -> PyResult<(Levels1D<'py>, Freqs<'py>)> {
    let (spec, freq) = core_noct::noct_synthesis(
        spectrum.as_slice()?,
        freqs.as_slice()?,
        fmin,
        fmax,
        n,
        g,
        fr,
    )
    .map_err(map_noct_error)?;
    Ok((
        Array1::from(spec).into_pyarray(py),
        Array1::from(freq).into_pyarray(py),
    ))
}
