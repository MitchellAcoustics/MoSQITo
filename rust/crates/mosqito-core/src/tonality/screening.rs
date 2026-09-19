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

    let stop: Vec<usize> = (1..=n).map(|i| i * m).collect();
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
        let mut block = 0usize;
        for (i, &s) in stop.iter().enumerate() {
            if peak_index < s && peak_index > s - m {
                block = i;
            }
        }

        let mut low_limit = peak_index;
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
        let t_width = freqs_flat[high_limit] - freqs_flat[low_limit];

        if t_width < cb_width {
            tones[block].push(peak_index - block * m);
        }

        index.retain(|&p| p > high_limit);
    }

    tones
}

#[cfg(test)]
mod tests {
    use super::*;

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
