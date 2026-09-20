//! Port of `_spectrum_smoothing`: a 1/24-octave energy-averaged smoothed
//! spectrum, used by [`super::screening_for_tones`] as the "6 dB above the
//! smoothed spectrum" screening criterion.
//!
//! # A reproduced `.ravel()`-order mismatch
//!
//! Python's `_spectrum_smoothing(freqs_in, spec, ...)` ravels `freqs_in`
//! (shape `(nseg, nfreqs)`, so `.ravel()` groups it **segment-major**: all
//! of segment 0's bins, then all of segment 1's, ...) and separately ravels
//! `spec` (shape `(nfreqs, nseg)` — already `spec_db.T` at the call site —
//! so `.ravel()` groups it **frequency-major**: segment 0..nseg-1's value at
//! bin 0, then bin 1, ...). Both flat arrays are then indexed with the same
//! `stop`/`bin_index` machinery, on the implicit (and for `nseg > 1`,
//! false) assumption that a flat position means the same thing in both.
//! This is very likely an unintentional bug — but it's real MoSQITo
//! behaviour with no standard to contradict it, so it is reproduced exactly
//! here rather than "fixed": `freqs_flat` is built segment-major,
//! `spec_flat` frequency-major, on purpose.
//!
//! # A verified simplification: `freqs_out` as a single shared row
//!
//! Python separately computes `low[i,:] = argmin(abs(freqs_out -
//! filter_freqs[i,0]))` — an argmin over the **whole 2-D `freqs_out`
//! array**, flattened, then broadcasts the resulting scalar across every
//! segment's slot in that row. Because every real caller builds `freqs_out`
//! with identical rows (the same shared frequency axis, one copy per
//! segment), and numpy's `argmin` breaks ties by taking the first
//! (row-major) occurrence, this always resolves to a position within row 0
//! — equivalent to computing the argmin once against a single shared row.
//! Confirmed empirically against real MoSQITo output (`golden_tonality.rs`,
//! both `nseg == 1` and `nseg > 1` cases) rather than assumed; see
//! `DEVIATIONS.md`.
//!
//! # Not reproduced: uninitialised-memory gaps in the placed output
//!
//! Python's final placement loop (`smooth_spec[low:high, i] = ...`,
//! `smooth_spec = numpy.empty(...)`) can leave some output rows untouched —
//! confirmed directly: coarse low/high snapping near the low-frequency edge
//! collapses several logical 1/24-octave bands onto the same one or two
//! output positions, leaving neighbouring positions uncovered by any band's
//! slice. Real MoSQITo then returns whatever `numpy.empty()` happened to
//! leave in memory there (a subnormal float and other implausible values
//! were observed directly, not a stable or algorithmically meaningful
//! result). This port fills any uncovered position with `0.0` instead — a
//! defined, deterministic choice, verified inert for every signal
//! `golden_tonality.rs` exercises: the screening criterion this feeds
//! (`spec_db[temp] > smooth_spec[temp] + 6`) only ever reads it at genuine
//! local-maxima positions, none of which fall in the handful of affected
//! low-frequency/Nyquist-edge positions observed. See `DEVIATIONS.md`.

use super::get_frequencies::get_frequencies;
use crate::dsp::nearest_index as argmin_abs_diff;

/// Matches `_spectrum_smoothing(freqs_in, spec, noct, low_freq, high_freq,
/// freqs_out)`.
///
/// - `freqs_by_seg`: `freqs_in`, one row per segment (`nseg` rows), each
///   `nfreqs` long. Real callers always pass the same row for every
///   segment, but this is kept per-segment to preserve the ravel-order
///   mismatch above.
/// - `spec_by_freq_then_seg`: `spec` (already `spec_db.T` at the call
///   site) — `nfreqs` rows, each `nseg` long.
/// - `freqs_out_row`: the shared frequency axis used for output placement
///   (see the simplification above) — `nfreqs` long.
///
/// Returns the smoothed spectrum, `nfreqs` rows x `nseg` columns (Python's
/// `smooth_spec`, pre-transpose/pre-squeeze).
pub fn spectrum_smoothing(
    freqs_by_seg: &[Vec<f64>],
    spec_by_freq_then_seg: &[Vec<f64>],
    noct: u32,
    low_freq: f64,
    high_freq: f64,
    freqs_out_row: &[f64],
) -> Vec<Vec<f64>> {
    let nperseg = spec_by_freq_then_seg.len();
    let nseg = freqs_by_seg.len();

    // spec.ravel(): frequency-major (row = freq bin, inner loop over segment).
    let mut spec_flat = Vec::with_capacity(nperseg * nseg);
    for row in spec_by_freq_then_seg {
        spec_flat.extend_from_slice(row);
    }
    // freqs_in.ravel(): segment-major (row = segment, inner loop over freq bin).
    let mut freqs_in_flat = Vec::with_capacity(nseg * nperseg);
    for row in freqs_by_seg {
        freqs_in_flat.extend_from_slice(row);
    }

    let stop: Vec<usize> = (1..=nseg).map(|i| i * nperseg).collect();

    let mut filter_freqs = get_frequencies(low_freq, high_freq, noct, 10, 1000.0);
    if let Some(last) = filter_freqs.last_mut() {
        last.f2 = high_freq;
    }
    if let Some(first) = filter_freqs.first_mut() {
        first.f1 = low_freq;
    }

    let mut smoothed_spectrum: Vec<Vec<f64>> = vec![vec![0.0; nseg]; filter_freqs.len()];

    // Literal translation of the Python `while nb_bands > 0` loop, double
    // decrement on an empty band included (see module doc: `_spectrum_smoothing`
    // discards/skips a band without stepping `i` back, which is preserved here).
    let mut nb_bands: i64 = filter_freqs.len() as i64;
    let mut i: usize = 0;
    while nb_bands > 0 {
        let band = filter_freqs[i];
        let bin_index: Vec<usize> = freqs_in_flat
            .iter()
            .enumerate()
            .filter(|&(_, &f)| f >= band.f1 && f <= band.f2)
            .map(|(idx, _)| idx)
            .collect();

        if bin_index.is_empty() {
            smoothed_spectrum.remove(i);
            filter_freqs.remove(i);
            nb_bands -= 1;
        } else {
            for (j, row) in smoothed_spectrum[i].iter_mut().enumerate() {
                let lo = stop[j] - nperseg;
                let hi = stop[j];
                let sel: Vec<usize> = bin_index
                    .iter()
                    .copied()
                    .filter(|&idx| idx < hi && idx > lo)
                    .collect();
                let spec_sum = if !sel.is_empty() {
                    sel.iter()
                        .map(|&idx| 10f64.powf(spec_flat[idx] / 10.0))
                        .sum::<f64>()
                        / sel.len() as f64
                } else {
                    1e-12
                };
                *row = 10.0 * spec_sum.log10();
            }
        }
        nb_bands -= 1;
        i += 1;
    }

    let mut smooth_spec = vec![vec![0.0_f64; nseg]; nperseg];
    for (j, band) in filter_freqs.iter().enumerate() {
        let low = argmin_abs_diff(freqs_out_row, band.f1);
        let high = argmin_abs_diff(freqs_out_row, band.f2).min(nperseg);
        if low < high {
            for row in &mut smooth_spec[low..high] {
                row.copy_from_slice(&smoothed_spectrum[j]);
            }
        }
    }

    smooth_spec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_segment_smoothing_is_finite_and_covers_the_range() {
        let nfreqs = 2000;
        let freqs: Vec<f64> = (0..nfreqs).map(|i| 50.0 + i as f64 * 6.0).collect();
        let spec: Vec<f64> = (0..nfreqs)
            .map(|i| 40.0 + (i as f64 * 0.01).sin() * 10.0)
            .collect();

        let freqs_by_seg = vec![freqs.clone()];
        let spec_by_freq_then_seg: Vec<Vec<f64>> = spec.iter().map(|&v| vec![v]).collect();

        let out = spectrum_smoothing(
            &freqs_by_seg,
            &spec_by_freq_then_seg,
            24,
            90.0,
            11200.0,
            &freqs,
        );
        assert_eq!(out.len(), nfreqs);
        assert!(out.iter().flatten().all(|v| v.is_finite()));
    }
}
