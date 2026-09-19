//! Frequency-band energy synthesis, matching
//! `mosqito.sound_level_meter.freq_band_synthesis`.
//!
//! When the requested band range reaches outside the input spectrum's own
//! frequency axis, Python rebuilds that axis as `arange(fmin.min(),
//! fmax.max() + df, df)` and resamples the spectrum onto it with
//! `numpy.interp`. Despite the warning it prints ("Empty values will be
//! filled with 0"), `numpy.interp` *clamps* to the edge value rather than
//! zero-filling, so the extension repeats the spectrum's first/last value —
//! reproduced here, message notwithstanding.
//!
//! This is reachable in practice: `sii_ansi` synthesises bands topping out
//! at 11360 Hz (octave procedure) from a `comp_spectrum` axis that only
//! reaches `fs/2`, so any `fs` below ~22.7 kHz — 16 kHz speech audio being
//! the obvious case — takes the `fmax` branch. Skipping it understated the
//! top band by ~4 dB at `fs = 16000`, which moved the reported SII by up to
//! 7e-3 once that band was not clamped out of the result.

use std::borrow::Cow;

use crate::dsp::{interp, nearest_index};

/// Sums `spectrum_db` (a dB spectrum on `freqs`) into the frequency bands
/// `[fmin[i], fmax[i])`, on an energy basis.
///
/// `fmin`/`fmax` must be contiguous and ascending (each band's upper edge at
/// or near the next band's lower edge), as every ANSI S3.5 band table is —
/// band `i`'s samples are `freqs[idx_low[i] .. idx_low[i+1])` for all but the
/// last band, which instead runs to `idx_up[last]`, matching
/// `numpy.split(spectrum, concatenate((idx_low, [idx_up[-1]])))[1:-1]`.
///
/// Returns `(band_levels_db, band_centers)`, `band_centers[i] = (fmin[i] +
/// fmax[i]) / 2`.
///
/// # Panics
/// Panics if `spectrum_db` and `freqs` differ in length, or if `fmin` and
/// `fmax` differ in length.
pub fn freq_band_synthesis(
    spectrum_db: &[f64],
    freqs: &[f64],
    fmin: &[f64],
    fmax: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(spectrum_db.len(), freqs.len());
    assert_eq!(fmin.len(), fmax.len());
    let n = fmin.len();

    let fmin_min = fmin.iter().copied().fold(f64::INFINITY, f64::min);
    let fmax_max = fmax.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Extend the frequency axis when the requested bands reach past either
    // end of it, matching Python's two `numpy.interp` branches (the second
    // sees whatever the first left behind).
    let mut spec: Cow<[f64]> = Cow::Borrowed(spectrum_db);
    let mut fr: Cow<[f64]> = Cow::Borrowed(freqs);
    if fmin_min < fr.iter().copied().fold(f64::INFINITY, f64::min) {
        let (grid, resampled) = regrid(&fr, &spec, fmin_min, fmax_max);
        fr = Cow::Owned(grid);
        spec = Cow::Owned(resampled);
    }
    if fmax_max > fr.iter().copied().fold(f64::NEG_INFINITY, f64::max) {
        let (grid, resampled) = regrid(&fr, &spec, fmin_min, fmax_max);
        fr = Cow::Owned(grid);
        spec = Cow::Owned(resampled);
    }

    let idx_low: Vec<usize> = fmin.iter().map(|&f| nearest_index(&fr, f)).collect();
    let idx_up: Vec<usize> = fmax.iter().map(|&f| nearest_index(&fr, f)).collect();

    let mut band_spectrum = vec![0.0; n];
    for i in 0..n {
        let start = idx_low[i];
        let end = if i + 1 < n {
            idx_low[i + 1]
        } else {
            idx_up[n - 1]
        };
        let sum: f64 = spec[start..end].iter().map(|&s| 10f64.powf(s / 10.0)).sum();
        band_spectrum[i] = 10.0 * sum.log10();
    }

    let centers: Vec<f64> = fmin
        .iter()
        .zip(fmax)
        .map(|(&a, &b)| (a + b) / 2.0)
        .collect();
    (band_spectrum, centers)
}

/// Rebuilds the frequency axis as `arange(fmin_min, fmax_max + df, df)` and
/// resamples `spec` onto it, matching Python's
/// `interp(arange(...), freqs, spectrum)` (edge-clamping, not zero-filling).
///
/// # Panics
/// Panics if `fr` has fewer than 2 points, as Python's `freqs[1] - freqs[0]`
/// would.
fn regrid(fr: &[f64], spec: &[f64], fmin_min: f64, fmax_max: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(
        fr.len() >= 2,
        "extending the frequency axis needs at least 2 input points to infer df"
    );
    let df = fr[1] - fr[0];
    let grid = arange(fmin_min, fmax_max + df, df);
    let resampled = interp(&grid, fr, spec);
    (grid, resampled)
}

/// `numpy.arange(start, stop, step)`: `start + k*step` for
/// `k < ceil((stop - start) / step)`, matching numpy's length computation
/// (and so its floating-point edge behaviour).
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let count = ((stop - start) / step).ceil();
    let count = if count.is_finite() && count > 0.0 {
        count as usize
    } else {
        0
    };
    (0..count).map(|k| start + k as f64 * step).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn two_equal_bands_split_a_flat_spectrum_evenly() {
        // 100 bins at 0 dB each, split into two 50-bin bands: each band's
        // energy sum is 10*log10(50).
        let freqs: Vec<f64> = (0..100).map(|i| i as f64 * 10.0).collect();
        let spectrum = vec![0.0; 100];
        let fmin = [0.0, 500.0];
        let fmax = [500.0, 990.0];
        let (levels, centers) = freq_band_synthesis(&spectrum, &freqs, &fmin, &fmax);
        assert_relative_eq!(levels[0], 10.0 * 50f64.log10(), epsilon = 1e-9);
        assert_relative_eq!(centers[0], 250.0, epsilon = 1e-9);
        assert_relative_eq!(centers[1], 745.0, epsilon = 1e-9);
    }
}
