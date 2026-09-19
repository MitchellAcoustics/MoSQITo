//! Top-level ISO 532-1 stationary loudness entry points, matching
//! `mosqito.sq_metrics.loudness.loudness_zwst`'s three public functions.

use ndarray::Array2;
use rayon::prelude::*;

use super::calc_slopes::calc_slopes;
use super::main_loudness::{main_loudness, FieldType};
use crate::dsp::{interp_zero_fill, resample_up_to};
use crate::slm::noct::{noct_spectrum, NoctError};
use crate::utils::{amp2db, time_segmentation};

/// The Bark axis every specific-loudness output shares: 0.1 to 24.0 Bark in
/// 0.1 Bark steps (240 points).
pub fn bark_axis() -> [f64; 240] {
    std::array::from_fn(|k| (k + 1) as f64 * 0.1)
}

const REF_PRESSURE: f64 = 2e-5;

/// Floating-point floor division matching Python's `a // b` (equivalently
/// `numpy.floor_divide`), which is *not* the same as `(a / b).floor()`.
///
/// IEEE-754 division rounds its result to the nearest representable double,
/// which can round a quotient that is mathematically a hair below an
/// integer up to that integer exactly — `(a / b).floor()` then keeps the
/// wrong integer, and checking the rounded quotient against the rounded
/// product of `floor(a/b) * b` isn't reliable either: for this exact case,
/// that product rounds right back to `a`, hiding the same sub-ULP error a
/// second time. CPython's `float.__floordiv__` avoids this by working from
/// `fmod(a, b)` (an *exact* IEEE-754 operation — no final rounding step,
/// since the true remainder is always representable) rather than from `a /
/// b`, which is what this reproduces (mirroring `float_divmod` in
/// `floatobject.c`).
fn python_float_floordiv(a: f64, b: f64) -> f64 {
    let m = a % b;
    let mut d = (a - m) / b;
    // CPython's `float_divmod` also corrects `mod` itself here
    // (`mod += wx`), since it returns both halves of `divmod`; this only
    // needs the quotient, so just the matching `div -= 1.0` is kept.
    if m != 0.0 && (b < 0.0) != (m < 0.0) {
        d -= 1.0;
    }
    let mut floordiv = d.floor();
    if d - floordiv > 0.5 {
        floordiv += 1.0;
    }
    floordiv
}

/// Computes stationary loudness from a time signal, per ISO 532-1:2017
/// (Zwicker method). Matches `loudness_zwst(signal, fs, field_type)` for a
/// 1-D signal.
///
/// Resamples to 48 kHz first if `fs < 48000`, as MoSQITo's Python does
/// (`loudness_zwst.py:96-103`) — ISO 532-1 requires 48 kHz.
///
/// Returns `(N, N_specific, bark_axis)`: `N` in sone, `N_specific` in
/// sone/Bark over the 240-point Bark axis.
pub fn loudness_zwst(
    signal: &[f64],
    fs: f64,
    field_type: FieldType,
) -> (f64, [f64; 240], [f64; 240]) {
    let (signal, fs) = resample_up_to(signal, fs, 48000.0);

    let sig2d = Array2::from_shape_vec((signal.len(), 1), signal).expect("column signal");
    let (spec_third, _freq) =
        noct_spectrum(sig2d.view(), fs, 24.0, 12600.0, 3, 10, 1000.0).expect("valid design");
    let spec_third_db = amp2db(&spec_third.column(0).to_vec(), REF_PRESSURE);

    let nm = main_loudness(&spec_third_db, field_type);
    let (n, n_specific) = calc_slopes(&nm);
    (n, n_specific, bark_axis())
}

/// Computes stationary loudness from a fine-band spectrum, per ISO 532-1:2017.
/// Matches `loudness_zwst_freq(spectrum, freqs, field_type)` for a 1-D
/// spectrum.
///
/// `spectrum` is an RMS amplitude spectrum (not dB) and `freqs` its
/// frequency axis in Hz. When `freqs` does not already span the full
/// 24 Hz–24 kHz range ISO 532-1 requires, the spectrum is zero-padded out to
/// it (`loudness_zwst_freq.py:109-117`), rather than left to `noct_synthesis`
/// to silently drop out-of-range content.
///
/// # Panics
/// Panics if `spectrum` and `freqs` differ in length.
pub fn loudness_zwst_freq(
    spectrum: &[f64],
    freqs: &[f64],
    field_type: FieldType,
) -> (f64, [f64; 240], [f64; 240]) {
    assert_eq!(
        spectrum.len(),
        freqs.len(),
        "spectrum and freqs must have the same length"
    );

    let freq_max = freqs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let freq_min = freqs.iter().cloned().fold(f64::INFINITY, f64::min);

    let (spectrum, freqs) = if freq_max < 24000.0 || freq_min > 24.0 {
        let df = freqs[1] - freqs[0];
        // Python computes `int(24000 // df)`, floating-point *floor*
        // division — not `int(24000 / df)`. The two disagree here: with
        // `df` the float64 nearest 0.1, `24000.0 / df` rounds to exactly
        // 240000.0, but the true quotient is a hair below that integer, so
        // floor division correctly gives 239999. Naively truncating the
        // rounded quotient instead gives 240000 — a one-sample-too-long
        // padded axis, and a measurably wrong loudness for the reference
        // signal (confirmed against the installed `mosqito` package).
        let n_new = python_float_floordiv(24000.0, df) as usize;
        let new_freqs: Vec<f64> = (0..n_new)
            .map(|k| 24000.0 * k as f64 / (n_new - 1) as f64)
            .collect();
        let padded = interp_zero_fill(&new_freqs, freqs, spectrum);
        (padded, new_freqs)
    } else {
        (spectrum.to_vec(), freqs.to_vec())
    };

    let (spec_third, _freq) =
        crate::slm::noct::noct_synthesis(&spectrum, &freqs, 24.0, 12600.0, 3, 10, 1000.0)
            .expect("valid design");
    let spec_third_db = amp2db(&spec_third, REF_PRESSURE);

    let nm = main_loudness(&spec_third_db, field_type);
    let (n, n_specific) = calc_slopes(&nm);
    (n, n_specific, bark_axis())
}

/// Errors from [`loudness_zwst_perseg`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoudnessZwstError {
    /// A per-segment `noct_spectrum` call failed (see [`NoctError`]).
    Noct(NoctError),
}

impl std::fmt::Display for LoudnessZwstError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoudnessZwstError::Noct(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LoudnessZwstError {}

/// `(N per segment, N_specific, bark_axis, time_axis)` — [`loudness_zwst_perseg`]'s
/// result.
type PersegResult = (Vec<f64>, Array2<f64>, [f64; 240], Vec<f64>);

/// Computes stationary loudness per time segment, per ISO 532-1:2017.
/// Matches `loudness_zwst_perseg(signal, fs, nperseg, noverlap, field_type)`.
///
/// Segments the signal with [`time_segmentation`], then runs the full
/// `loudness_zwst` pipeline — including `noct_spectrum`, run in parallel
/// across segments — on each block.
///
/// Returns `(N, N_specific, bark_axis, time_axis)`: `N` and each column of
/// `N_specific` are one value/column per segment; `time_axis` is each
/// segment's mean sample time, in seconds.
pub fn loudness_zwst_perseg(
    signal: &[f64],
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
    field_type: FieldType,
) -> Result<PersegResult, LoudnessZwstError> {
    let (signal, fs) = resample_up_to(signal, fs, 48000.0);

    let (blocks, time_axis) = time_segmentation(&signal, fs, nperseg, noverlap);
    let nseg = blocks.ncols();

    let (spec_third, _freq) = noct_spectrum(blocks.view(), fs, 24.0, 12600.0, 3, 10, 1000.0)
        .map_err(LoudnessZwstError::Noct)?;

    let results: Vec<(f64, [f64; 240])> = (0..nseg)
        .into_par_iter()
        .map(|col| {
            let spec_db = amp2db(&spec_third.column(col).to_vec(), REF_PRESSURE);
            let nm = main_loudness(&spec_db, field_type);
            calc_slopes(&nm)
        })
        .collect();

    let n: Vec<f64> = results.iter().map(|&(n, _)| n).collect();
    let mut n_specific = Array2::<f64>::zeros((240, nseg));
    for (col, (_, spec)) in results.into_iter().enumerate() {
        n_specific
            .column_mut(col)
            .assign(&ndarray::Array1::from(spec.to_vec()));
    }

    Ok((n, n_specific, bark_axis(), time_axis))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_float_floordiv_matches_pythons_edge_case() {
        // 24000.0 / 0.1 rounds to exactly 240000.0 in IEEE-754, but the true
        // quotient sits fractionally below that integer (0.1 is not exactly
        // representable), so Python's `24000 // df` is 239999.0, not
        // 240000.0. Confirmed against numpy directly.
        let df = 48000.0 / 480000.0; // exactly how loudness_zwst_freq derives it
        assert_eq!(df, 0.1);
        assert_eq!(python_float_floordiv(24000.0, df), 239999.0);
    }

    #[test]
    fn python_float_floordiv_matches_plain_floor_away_from_edge_cases() {
        assert_eq!(python_float_floordiv(10.0, 3.0), 3.0);
        assert_eq!(python_float_floordiv(9.0, 3.0), 3.0);
        assert_eq!(python_float_floordiv(1.0, 4.0), 0.0);
    }
}
