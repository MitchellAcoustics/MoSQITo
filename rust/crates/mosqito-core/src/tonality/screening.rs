//! Port of `_screening_for_tones`'s `"smoothed"` method (the Bray & Caspary
//! SQS 2008 candidate-detection criteria) — the only method any real caller
//! (`_tnr_main_calc`/`_pr_main_calc`) ever passes; the `"not-smoothed"`
//! (Aures/Terhardt) method is dead code from the public API's perspective
//! and is not ported, matching this project's precedent of narrowing to
//! what's actually reachable (see `DEVIATIONS.md`).

use super::critical_band::critical_band;
use super::lth::lth;
use super::spectrum_smoothing::spectrum_smoothing;

/// Finds tonal candidates in `spec_db_by_seg`, per `_screening_for_tones`.
///
/// `freqs_by_seg`/`spec_db_by_seg` each hold one row per segment (`nseg`
/// rows), all the same length (`nfreqs`) — for a single spectrum (`nseg ==
/// 1`), pass a single row. Every real caller uses the same frequency axis
/// for every segment.
///
/// Returns, per segment, the candidate tone indices local to that segment's
/// own `nfreqs`-long arrays (matching Python's `peak_index - block*m`).
pub fn screening_for_tones(
    freqs_by_seg: &[Vec<f64>],
    spec_db_by_seg: &[Vec<f64>],
    low_freq: f64,
    high_freq: f64,
) -> Vec<Vec<usize>> {
    let n = freqs_by_seg.len();
    let m = freqs_by_seg[0].len();

    // spec_db.T, matching the shape `spectrum_smoothing` expects.
    let mut spec_t: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
    for (s, row) in spec_db_by_seg.iter().enumerate() {
        for (f, &v) in row.iter().enumerate() {
            spec_t[f][s] = v;
        }
    }
    let freqs_out_row = &freqs_by_seg[0];
    let smooth_spec_2d = spectrum_smoothing(
        freqs_by_seg,
        &spec_t,
        24,
        low_freq,
        high_freq,
        freqs_out_row,
    );

    // screening's own `.T.ravel()`: segment-major, consistently with
    // `spec_db`/`freqs` below (unlike `spectrum_smoothing`'s internal
    // mismatch, which stays encapsulated there).
    let mut smooth_spec_flat = Vec::with_capacity(n * m);
    for s in 0..n {
        smooth_spec_flat.extend(smooth_spec_2d.iter().map(|row| row[s]));
    }
    let mut spec_db_flat = Vec::with_capacity(n * m);
    for row in spec_db_by_seg {
        spec_db_flat.extend_from_slice(row);
    }
    let mut freqs_flat = Vec::with_capacity(n * m);
    for row in freqs_by_seg {
        freqs_flat.extend_from_slice(row);
    }

    let total = spec_db_flat.len();

    // Criteria 1: local maxima (diff(sign(diff(spec_db))) < 0).
    let d1: Vec<f64> = spec_db_flat.windows(2).map(|w| w[1] - w[0]).collect();
    let sign: Vec<i32> = d1
        .iter()
        .map(|&v| {
            if v > 0.0 {
                1
            } else if v < 0.0 {
                -1
            } else {
                0
            }
        })
        .collect();
    let d2: Vec<i32> = sign.windows(2).map(|w| w[1] - w[0]).collect();
    let maxima: Vec<usize> = d2
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v < 0)
        .map(|(idx, _)| idx + 1)
        .collect();

    // Criteria 2: >= 6 dB above the smoothed spectrum.
    let after_c2: Vec<usize> = maxima
        .into_iter()
        .filter(|&p| spec_db_flat[p] > smooth_spec_flat[p] + 6.0)
        .collect();

    // Criteria 3: >= 10 dB above the threshold of hearing.
    let threshold = lth(&freqs_flat);
    let mut index: Vec<usize> = after_c2
        .into_iter()
        .filter(|&p| spec_db_flat[p] > threshold[p] + 10.0)
        .collect();
    index.sort_unstable();

    let mut tones: Vec<Vec<usize>> = vec![Vec::new(); n];

    while !index.is_empty() {
        let mut peak_index = index[0];
        // The segment this candidate belongs to. Python finds it by scanning
        // a `stop` table (`arange(1, n+1) * m - 1` for several segments,
        // `[m]` for one) and keeping the `i` whose `stop[i] - m < peak_index
        // < stop[i]` — which is exactly `peak_index / m` for every index the
        // scan matches. The scan leaves one index per segment unmatched (the
        // segment's *last* bin), where Python falls through to a stale
        // `block` from the previous candidate, or raises `UnboundLocalError`
        // on the first one; dividing gives that bin its own (correct)
        // segment instead of a wrong or undefined one. Identical to Python
        // wherever Python is defined — see `DEVIATIONS.md`.
        let block = peak_index / m;

        // `low_limit` tracks the *original* peak, but the left-hand scan
        // below starts from the peak position the right-hand scan may have
        // already moved rightward, so it can be decremented further than the
        // original peak's distance from 0 and go negative. Python lets it,
        // then negative-indexes `freqs` with it (wrapping to the end of the
        // flat array); `isize` plus `python_index` below reproduces that
        // rather than underflowing a `usize`.
        let mut low_limit: isize = peak_index as isize;
        let mut high_limit = peak_index;

        let mut temp = peak_index + 1;
        while temp < total
            && spec_db_flat[temp] > smooth_spec_flat[temp] + 6.0
            && temp + 1 < (block + 1) * m
        {
            if spec_db_flat[temp] > spec_db_flat[peak_index] {
                peak_index = temp;
            }
            high_limit += 1;
            temp += 1;
        }

        if peak_index > 0 {
            let mut left: isize = peak_index as isize - 1;
            while left >= 0 {
                let lt = left as usize;
                if !(spec_db_flat[lt] > smooth_spec_flat[lt] + 6.0 && lt + 1 > block * m) {
                    break;
                }
                if spec_db_flat[lt] > spec_db_flat[peak_index] {
                    peak_index = lt;
                }
                low_limit -= 1;
                left -= 1;
            }
        }

        let (f1, f2) = critical_band(freqs_flat[peak_index]);
        let cb_width = f2 - f1;
        let t_width = freqs_flat[high_limit] - python_index(&freqs_flat, low_limit);

        if t_width < cb_width {
            tones[block].push(peak_index - block * m);
        }

        index.retain(|&p| p > high_limit);
    }

    tones
}

/// Reads `arr[idx]` with Python's list-indexing semantics: a negative `idx`
/// counts back from the end. Only reachable via `low_limit` above, which
/// Python also allows to go negative.
///
/// # Panics
/// Panics if `idx` is out of range even after wrapping, as Python's own
/// `IndexError` would.
fn python_index(arr: &[f64], idx: isize) -> f64 {
    let wrapped = if idx < 0 {
        arr.len() as isize + idx
    } else {
        idx
    };
    assert!(
        wrapped >= 0 && (wrapped as usize) < arr.len(),
        "index {idx} out of range for a {}-element array",
        arr.len()
    );
    arr[wrapped as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_index_counts_back_from_the_end_for_a_negative_index() {
        // `low_limit` can go negative (the left-hand scan starts from a peak
        // the right-hand scan already moved rightward), and Python then
        // wraps rather than failing. Holding it in a `usize` underflowed
        // instead — the bug this helper exists to avoid.
        let arr = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(python_index(&arr, 0), 10.0);
        assert_eq!(python_index(&arr, 3), 40.0);
        assert_eq!(python_index(&arr, -1), 40.0);
        assert_eq!(python_index(&arr, -4), 10.0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn python_index_rejects_an_index_that_does_not_wrap_into_range() {
        python_index(&[1.0, 2.0], -5);
    }

    #[test]
    fn finds_a_pure_tone_injected_into_broadband_noise() {
        let m = 2000usize;
        let freqs: Vec<f64> = (0..m)
            .map(|i| 90.0 + i as f64 * (11100.0 / m as f64))
            .collect();
        let tone_bin = m / 3;
        let spec: Vec<f64> = (0..m)
            .map(|i| if i == tone_bin { 80.0 } else { 40.0 })
            .collect();

        let tones = screening_for_tones(&[freqs], &[spec], 90.0, 11200.0);
        assert_eq!(tones.len(), 1);
        assert!(tones[0].contains(&tone_bin));
    }
}
