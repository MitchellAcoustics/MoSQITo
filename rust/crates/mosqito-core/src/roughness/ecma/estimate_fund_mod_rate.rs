//! ECMA-418-2 (2nd Ed, 2022) §7.1.5.3: estimates the fundamental modulation
//! rate from a peak list by finding the harmonic complex (a set of peaks at
//! near-integer-ratio frequencies) with the most energy. Matches
//! `_estimate_fund_mod_rate.py`.

/// Estimates the fundamental modulation rate.
///
/// `f_p` and `ai_tilde` are the refined peak frequencies and their
/// high-modulation-rate-weighted amplitudes (same length, `N_peak <= 10`).
///
/// Returns `(mod_rate, a_hat)`: the fundamental rate, and the
/// low-modulation-rate weighting's per-peak amplitude input — the winning
/// harmonic complex's amplitudes, each further scaled by Eq. 93's
/// centre-of-gravity weighting (so `a_hat.len()` is the winning complex's
/// size, not `N_peak`).
pub fn estimate_fund_mod_rate(f_p: &[f64], ai_tilde: &[f64]) -> (f64, Vec<f64>) {
    let n_peak = f_p.len();

    // For each candidate fundamental i0, find the harmonic complex: peaks
    // whose frequency is within 4% of an integer multiple of f_p[i0], with
    // at most one peak kept per integer ratio (the one whose ratio is
    // closest to that integer, Eq. 89).
    let mut complexes: Vec<Vec<usize>> = Vec::with_capacity(n_peak);
    let mut energies: Vec<f64> = Vec::with_capacity(n_peak);

    for i0 in 0..n_peak {
        // Integer ratio of every peak's frequency to this candidate's
        // (Eq. 88). `f64::round` rounds half away from zero, where numpy
        // rounds half to even; both are continuous-data results of the
        // refinement step, so landing exactly on `x.5` (the only case where
        // they'd disagree) does not occur in practice.
        let ratios: Vec<f64> = f_p.iter().map(|&f| (f / f_p[i0]).round()).collect();

        // One candidate index per distinct ratio value: the sole occurrence
        // if it's unique, or (Eq. 89) whichever occurrence's frequency sits
        // closest to that integer multiple of f_p[i0], among duplicates.
        //
        // Order matters here, not just membership: `_estimate_fund_mod_rate.py`
        // builds this from `numpy.unique`, which groups by *ascending ratio
        // value* — first every unique (count-1) ratio in that order, then
        // every resolved duplicate group in that same ascending order — and
        // a later step (`i_peak`, see below) indexes into this exact
        // sequence positionally. Grouping by original peak order instead
        // would silently reorder it and change that later step's result.
        let mut value_groups: Vec<(f64, Vec<usize>)> = Vec::new();
        for (i, &r) in ratios.iter().enumerate() {
            match value_groups.iter_mut().find(|(v, _)| *v == r) {
                Some((_, members)) => members.push(i),
                None => value_groups.push((r, vec![i])),
            }
        }
        value_groups.sort_by(|(a, _), (b, _)| a.total_cmp(b));

        let mut candidates: Vec<usize> = Vec::new();
        for (_, members) in &value_groups {
            if members.len() == 1 {
                candidates.push(members[0]);
            }
        }
        for (_, members) in &value_groups {
            if members.len() > 1 {
                let best = members
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        let crit_a = (f_p[a] / (ratios[a] * f_p[i0]) - 1.0).abs();
                        let crit_b = (f_p[b] / (ratios[b] * f_p[i0]) - 1.0).abs();
                        crit_a.total_cmp(&crit_b)
                    })
                    .expect("non-empty group");
                candidates.push(best);
            }
        }

        // Eq. 90-91: keep candidates within the harmonic-complex tolerance,
        // and sum their weighted amplitudes.
        let harmonic: Vec<usize> = candidates
            .into_iter()
            .filter(|&i| (f_p[i] / (ratios[i] * f_p[i0] + 1e-9) - 1.0).abs() < 0.04)
            .collect();
        let energy: f64 = harmonic.iter().map(|&i| ai_tilde[i]).sum();

        complexes.push(harmonic);
        energies.push(energy);
    }

    // The harmonic complex with the highest total energy wins.
    let i_max = energies
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .expect("at least one peak");
    let i_complex = &complexes[i_max];
    let mod_rate = f_p[i_max];

    // Eq. 93: weight by distance between mod_rate and the complex's
    // amplitude-weighted centre of gravity.
    //
    // `_estimate_fund_mod_rate.py:59-61` computes `i_peak =
    // argmax(Ai_tilde[I_max])` — an index *local* to the winning complex's
    // own sub-array — and then uses it to index `f_p` directly
    // (`f_p[i_peak]`), rather than `f_p[I_max[i_peak]]`. That looks like an
    // array-indexing slip rather than anything ECMA-418-2 Eq. 93 intends,
    // but it is reproduced here rather than "fixed": `c_R`
    // (`non_linear_transform.rs`) was re-fit against the ECMA-418-2 Annex C
    // reference corpus using this exact behaviour (via MoSQITo's own
    // unmodified `_estimate_fund_mod_rate`), so silently changing it here
    // would invalidate that calibration without being sure it's actually
    // wrong. See `DEVIATIONS.md`.
    let i_peak_local = i_complex
        .iter()
        .enumerate()
        .max_by(|(_, &a), (_, &b)| ai_tilde[a].total_cmp(&ai_tilde[b]))
        .map(|(local, _)| local)
        .expect("winning complex is non-empty");

    let sum_weighted_f: f64 = i_complex.iter().map(|&i| f_p[i] * ai_tilde[i]).sum();
    let sum_amp: f64 = i_complex.iter().map(|&i| ai_tilde[i]).sum();
    let centre_of_gravity = sum_weighted_f / sum_amp;
    let w_peak = 1.0 + 0.1 * (centre_of_gravity - f_p[i_peak_local]).abs().powf(0.749);

    let a_hat: Vec<f64> = i_complex.iter().map(|&i| ai_tilde[i] * w_peak).collect();

    (mod_rate, a_hat)
}
