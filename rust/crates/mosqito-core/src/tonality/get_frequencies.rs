//! Port of MoSQITo's `_getFrequencies` (n-th octave band edges), used by
//! [`super::spectrum_smoothing`]'s 1/24-octave energy averaging.
//!
//! This is a fresh, direct translation rather than a reuse of
//! [`crate::slm::center_freq`]/[`crate::slm::filter_bandwidth`]: those
//! compute center frequencies by rounding `log(f/fr)/log(u)` to the nearest
//! band number and derive bandwidth from a Butterworth quality factor —
//! genuinely different formulas from `_getFrequencies`'s direct
//! increment-and-test loop and its `G^(±1/(2b))` band-edge ratio. Reusing
//! the Phase 1 functions would have been an unverified assumption of
//! equivalence; this instead matches Python's algorithm line for line.

/// One n-th octave band: (lower edge, exact midband, upper edge).
#[derive(Debug, Clone, Copy)]
pub struct BandEdges {
    pub f1: f64,
    pub fm: f64,
    pub f2: f64,
}

/// Matches `_getFrequencies(fstart, fend, b, G, fr)["f"]`. `b` is the number
/// of bands per octave (`spectrum_smoothing` always calls this with `b=24`,
/// `g_base=10`, `fr=1000`).
///
/// # Panics
/// Panics if `g_base` is neither 10 nor 2.
pub fn get_frequencies(fstart: f64, fend: f64, b: u32, g_base: u32, fr: f64) -> Vec<BandEdges> {
    let g = match g_base {
        10 => 10f64.powf(3.0 / 10.0),
        2 => 2.0,
        other => panic!("get_frequencies: G must be 10 or 2, got {other}"),
    };
    let mut freqs = Vec::new();
    let mut x: i64 = -1000;
    let mut f2 = 0.0;
    while f2 <= fend {
        let fm = if b % 2 == 0 {
            g.powf((2.0 * x as f64 - 59.0) / (2.0 * b as f64)) * fr
        } else {
            g.powf((x as f64 - 30.0) / b as f64) * fr
        };
        let f1 = g.powf(-1.0 / (2.0 * b as f64)) * fm;
        f2 = g.powf(1.0 / (2.0 * b as f64)) * fm;
        if f2 >= fstart {
            freqs.push(BandEdges { f1, fm, f2 });
        }
        x += 1;
    }
    freqs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_cover_the_requested_range_and_increase_monotonically() {
        let bands = get_frequencies(90.0, 11200.0, 24, 10, 1000.0);
        assert!(!bands.is_empty());
        assert!(bands.first().unwrap().f2 >= 90.0);
        assert!(bands.last().unwrap().f2 >= 11200.0);
        for w in bands.windows(2) {
            assert!(w[1].fm > w[0].fm);
        }
    }
}
