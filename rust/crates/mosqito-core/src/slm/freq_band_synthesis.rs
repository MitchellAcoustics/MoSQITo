//! Frequency-band energy synthesis, matching
//! `mosqito.sound_level_meter.freq_band_synthesis`.
//!
//! Only the in-range case is implemented: every caller (SII's band
//! procedures) synthesises bands that top out under 12 kHz from a
//! `comp_spectrum` output spanning up to `fs/2` (24 kHz at 48 kHz), so the
//! Python function's below-`fmin`/above-`fmax` zero-padding branches — which
//! only fire when the input spectrum doesn't already cover the requested
//! band range — are never exercised and are not ported, the same
//! scope-narrowing this port applies to other never-hit parameter
//! combinations.

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

    let idx_low: Vec<usize> = fmin.iter().map(|&f| nearest_index(freqs, f)).collect();
    let idx_up: Vec<usize> = fmax.iter().map(|&f| nearest_index(freqs, f)).collect();

    let mut band_spectrum = vec![0.0; n];
    for i in 0..n {
        let start = idx_low[i];
        let end = if i + 1 < n {
            idx_low[i + 1]
        } else {
            idx_up[n - 1]
        };
        let sum: f64 = spectrum_db[start..end]
            .iter()
            .map(|&s| 10f64.powf(s / 10.0))
            .sum();
        band_spectrum[i] = 10.0 * sum.log10();
    }

    let centers: Vec<f64> = fmin
        .iter()
        .zip(fmax)
        .map(|(&a, &b)| (a + b) / 2.0)
        .collect();
    (band_spectrum, centers)
}

fn nearest_index(freqs: &[f64], target: f64) -> usize {
    freqs
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| (a - target).abs().total_cmp(&(b - target).abs()))
        .map(|(i, _)| i)
        .expect("freqs must be non-empty")
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
