//! Port of `_find_highest_tone`: recursive within-critical-band
//! tie-breaking between multiple tonal candidates.

use super::critical_band::critical_band;
use crate::dsp::nearest_index as argmin_abs_diff;

/// Finds the two highest-level tones within the critical band centred on
/// `freqs[ind]`, matching `_find_highest_tone`.
///
/// Returns `(ind_p, ind_s, remaining_candidates)`: `ind_p` is the highest
/// tone's index, `ind_s` the second-highest's (`None` if the band holds
/// only one candidate), and `remaining_candidates` is `candidates` with
/// every candidate outside the winning band's top two removed.
///
/// Python threads an explicit `nb_tones` counter alongside the candidate
/// array through the recursion; it is always exactly `candidates.len()`
/// (every place that changes one changes the other identically), so this
/// port drops it and lets callers read `.len()` instead — a
/// behaviour-preserving simplification, not a change.
pub fn find_highest_tone(
    freqs: &[f64],
    spec_db: &[f64],
    mut candidates: Vec<usize>,
    ind: usize,
) -> (usize, Option<usize>, Vec<usize>) {
    let f = freqs[ind];
    let (f1, f2) = critical_band(f);
    let low_limit_idx = argmin_abs_diff(freqs, f1);
    let high_limit_idx = argmin_abs_diff(freqs, f2);

    let multiple_idx: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&i| i > low_limit_idx && i < high_limit_idx)
        .collect();

    if multiple_idx.len() > 1 {
        let mut order: Vec<usize> = (0..multiple_idx.len()).collect();
        order.sort_by(|&a, &b| {
            spec_db[multiple_idx[b]]
                .partial_cmp(&spec_db[multiple_idx[a]])
                .unwrap()
        });

        let ind_p = multiple_idx[order[0]];
        let ind_s = multiple_idx[order[1]];

        for &s in &order[2..] {
            let sup = multiple_idx[s];
            candidates.retain(|&c| c != sup);
        }

        if ind_p != ind {
            return find_highest_tone(freqs, spec_db, candidates, ind_p);
        }
        (ind_p, Some(ind_s), candidates)
    } else {
        (ind, None, candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_candidate_in_its_band_has_no_second_tone() {
        let freqs: Vec<f64> = (0..500).map(|i| 100.0 + i as f64 * 20.0).collect();
        let spec: Vec<f64> = vec![40.0; 500];
        let (ind_p, ind_s, remaining) = find_highest_tone(&freqs, &spec, vec![10], 10);
        assert_eq!(ind_p, 10);
        assert_eq!(ind_s, None);
        assert_eq!(remaining, vec![10]);
    }

    #[test]
    fn the_louder_of_two_candidates_in_the_same_band_wins() {
        let freqs: Vec<f64> = (0..500).map(|i| 1000.0 + i as f64 * 2.0).collect();
        let mut spec = vec![40.0; 500];
        spec[100] = 60.0;
        spec[102] = 70.0;
        let (ind_p, ind_s, _) = find_highest_tone(&freqs, &spec, vec![100, 102], 100);
        assert_eq!(ind_p, 102);
        assert_eq!(ind_s, Some(100));
    }
}
