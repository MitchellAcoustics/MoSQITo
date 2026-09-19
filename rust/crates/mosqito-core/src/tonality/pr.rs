//! Port of `_pr_main_calc`: prominence ratio per ECMA-74 Annex D, with the
//! T-PR total per ECMA TR/108. Scope matches [`super::tnr`]'s: the
//! 1-D-spectrum and (2-D spectrum, 1-D shared frequency axis) cases only.

use super::critical_band::{critical_band, lower_critical_band, upper_critical_band};
use super::find_highest_tone::find_highest_tone;
use super::screening::screening_for_tones;

fn argmin_abs_diff(row: &[f64], target: f64) -> usize {
    let mut best_idx = 0;
    let mut best_diff = (row[0] - target).abs();
    for (idx, &v) in row.iter().enumerate().skip(1) {
        let d = (v - target).abs();
        if d < best_diff {
            best_diff = d;
            best_idx = idx;
        }
    }
    best_idx
}

/// Per-segment PR results, matching `_pr_main_calc`'s return tuple.
pub struct PrResult {
    pub tones_freqs: Vec<Vec<f64>>,
    pub pr: Vec<Vec<f64>>,
    pub prominence: Vec<Vec<bool>>,
    pub t_pr: Vec<f64>,
}

/// Matches `_pr_main_calc(spectrum_db, freq_axis)`. See [`super::tnr::tnr_main_calc`]
/// for the shape convention `spectrum_db_by_seg` follows.
pub fn pr_main_calc(spectrum_db_by_seg: &[Vec<f64>], freq_axis: &[f64]) -> PrResult {
    let nseg = spectrum_db_by_seg.len();

    let freq_index: Vec<usize> = freq_axis
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f > 89.1 && f < 11200.0)
        .map(|(i, _)| i)
        .collect();
    let freqs: Vec<f64> = freq_index.iter().map(|&i| freq_axis[i]).collect();

    let spec_db_filtered: Vec<Vec<f64>> = spectrum_db_by_seg
        .iter()
        .map(|seg| freq_index.iter().map(|&i| seg[i]).collect())
        .collect();

    let freqs_by_seg: Vec<Vec<f64>> = (0..nseg).map(|_| freqs.clone()).collect();
    let candidates = screening_for_tones(&freqs_by_seg, &spec_db_filtered, 90.0, 11200.0);

    let mut tones_freqs = Vec::with_capacity(nseg);
    let mut pr_all = Vec::with_capacity(nseg);
    let mut prom_all = Vec::with_capacity(nseg);
    let mut t_pr = Vec::with_capacity(nseg);

    for s in 0..nseg {
        let (tf, pr, prom) = evaluate_segment(&freqs, &spec_db_filtered[s], candidates[s].clone());
        let sum_lin: f64 = pr
            .iter()
            .zip(&prom)
            .filter(|&(_, &p)| p)
            .map(|(&t, _)| 10f64.powf(t / 10.0))
            .sum();
        let t = if sum_lin != 0.0 {
            10.0 * sum_lin.log10()
        } else {
            0.0
        };
        tones_freqs.push(tf);
        pr_all.push(pr);
        prom_all.push(prom);
        t_pr.push(t);
    }

    PrResult {
        tones_freqs,
        pr: pr_all,
        prominence: prom_all,
        t_pr,
    }
}

fn band_level(freqs: &[f64], spec: &[f64], f1: f64, f2: f64) -> (f64, usize, usize) {
    let low = argmin_abs_diff(freqs, f1);
    let high = argmin_abs_diff(freqs, f2);
    let spec_sum: f64 = spec[low..high].iter().map(|&v| 10f64.powf(v / 10.0)).sum();
    let level = if spec_sum != 0.0 {
        10.0 * spec_sum.log10()
    } else {
        0.0
    };
    (level, low, high)
}

fn evaluate_segment(
    freqs: &[f64],
    spec: &[f64],
    mut peaks: Vec<usize>,
) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut tones_freqs = Vec::new();
    let mut pr = Vec::new();
    let mut prominence = Vec::new();

    while !peaks.is_empty() {
        let ind0 = peaks[0];
        let ind = if peaks.len() > 1 {
            let (p, _s, remaining) = find_highest_tone(freqs, spec, peaks.clone(), ind0);
            peaks = remaining;
            p
        } else {
            ind0
        };

        let ft = freqs[ind];

        let (f1, f2) = critical_band(ft);
        let (lm, low_limit_idx, high_limit_idx) = band_level(freqs, spec, f1, f2);

        let (f1l, f2l) = lower_critical_band(ft);
        let (ll, _, _) = band_level(freqs, spec, f1l, f2l);
        let delta_f = f2l - f1l;

        let (f1u, f2u) = upper_critical_band(ft);
        let (lu, _, _) = band_level(freqs, spec, f1u, f2u);

        let delta = if ft <= 171.4 {
            10.0 * (10f64.powf(0.1 * lm)).log10()
                - 10.0
                    * (((100.0 / delta_f) * 10f64.powf(0.1 * ll) + 10f64.powf(0.1 * lu)) * 0.5)
                        .log10()
        } else {
            10.0 * (10f64.powf(0.1 * lm)).log10()
                - 10.0 * ((10f64.powf(0.1 * ll) + 10f64.powf(0.1 * lu)) * 0.5).log10()
        };

        if delta > 0.0 {
            tones_freqs.push(ft);
            pr.push(delta);
            let prom = if ft <= 1000.0 {
                delta >= 9.0 + 10.0 * (1000.0 / ft).log10()
            } else {
                delta >= 9.0
            };
            prominence.push(prom);
        }

        // Suppresses every remaining candidate within the *middle* critical
        // band (the one centred on `ft`), matching `_pr_main_calc`'s reuse
        // of `low_limit_idx`/`high_limit_idx` (not the lower/upper bands'
        // own bounds, which Python stores under different variable names).
        peaks.retain(|&p| !(p >= low_limit_idx && p <= high_limit_idx));
    }

    (tones_freqs, pr, prominence)
}
