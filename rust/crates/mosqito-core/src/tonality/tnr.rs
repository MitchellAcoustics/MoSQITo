//! Port of `_tnr_main_calc` and the `tnr_ecma_*` entry points: tone-to-noise
//! ratio per ECMA-74 Annex D, with the T-TNR total per ECMA TR/108.
//!
//! Scope: the `(spectrum_db 1-D)` and `(spectrum_db 2-D, freq_axis 1-D)`
//! branches of `_tnr_main_calc` — i.e. a single spectrum, or several
//! segments sharing one frequency axis. That covers every real caller
//! (`tnr_ecma_st`, `tnr_ecma_freq` with a 1-D `freqs`, and
//! `tnr_ecma_perseg`'s 1-D-signal path, whose internal `comp_spectrum` call
//! always produces exactly this shape). The remaining `(spectrum_db 2-D,
//! freq_axis 2-D)` branch — a per-segment-varying frequency axis, which no
//! entry point in this port constructs — is not ported, matching the
//! project's precedent of narrowing to what's actually reachable; see
//! `DEVIATIONS.md`.

use super::critical_band::critical_band;
use super::find_highest_tone::find_highest_tone;
use super::peak_level::peak_level;
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

/// Per-segment TNR results, matching `_tnr_main_calc`'s return tuple.
pub struct TnrResult {
    pub tones_freqs: Vec<Vec<f64>>,
    pub tnr: Vec<Vec<f64>>,
    pub prominence: Vec<Vec<bool>>,
    pub t_tnr: Vec<f64>,
}

/// Matches `_tnr_main_calc(spectrum_db, freq_axis)`.
///
/// `spectrum_db_by_seg` holds one row per segment (`nseg` rows, `nseg == 1`
/// for a single spectrum), each the same length as `freq_axis` and aligned
/// index-for-index with it.
pub fn tnr_main_calc(spectrum_db_by_seg: &[Vec<f64>], freq_axis: &[f64]) -> TnrResult {
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
    let mut tnr_all = Vec::with_capacity(nseg);
    let mut prom_all = Vec::with_capacity(nseg);
    let mut t_tnr = Vec::with_capacity(nseg);

    for s in 0..nseg {
        let (tf, tnr, prom) = evaluate_segment(
            &freqs,
            &spec_db_filtered[s],
            freq_axis,
            candidates[s].clone(),
        );
        let sum_lin: f64 = tnr
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
        tnr_all.push(tnr);
        prom_all.push(prom);
        t_tnr.push(t);
    }

    TnrResult {
        tones_freqs,
        tnr: tnr_all,
        prominence: prom_all,
        t_tnr,
    }
}

/// One segment's worth of `_tnr_main_calc`'s per-candidate evaluation loop.
///
/// `freqs`/`spec` are the frequency-of-interest-filtered arrays; `freq_axis_full`
/// is the *original*, unfiltered axis (Python's `frs`) — used, per a
/// reproduced indexing quirk, with indices computed against the filtered
/// `freqs` array (see the `delta_ftot` comment below).
fn evaluate_segment(
    freqs: &[f64],
    spec: &[f64],
    freq_axis_full: &[f64],
    mut peaks: Vec<usize>,
) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut tones_freqs = Vec::new();
    let mut tnr = Vec::new();
    let mut prominence = Vec::new();

    while !peaks.is_empty() {
        let ind = peaks[0];
        let (ind_p, ind_s) = if peaks.len() > 1 {
            let (p, s, remaining) = find_highest_tone(freqs, spec, peaks.clone(), ind);
            peaks = remaining;
            (p, s)
        } else {
            (ind, None)
        };

        let lt;
        let ltot;
        let f1;
        let f2;
        let low_limit_idx;
        let high_limit_idx;
        let delta_ft;
        let f;

        if let Some(ind_s_val) = ind_s {
            let fp = freqs[ind_p];
            let fs = freqs[ind_s_val];
            let delta_f = 21.0 * 10f64.powf(1.2 * (fp / 212.0).log10().abs().powf(1.8));

            if (fs - fp).abs() < delta_f {
                let lp = peak_level(freqs, spec, ind_p);
                let ls = peak_level(freqs, spec, ind_s_val);
                lt = 10.0 * (10f64.powf(lp / 10.0) + 10f64.powf(ls / 10.0)).log10();

                let (a, b) = critical_band(fp);
                f1 = a;
                f2 = b;
                low_limit_idx = argmin_abs_diff(freqs, f1);
                high_limit_idx = argmin_abs_diff(freqs, f2);
                let spec_sum: f64 = spec[low_limit_idx..high_limit_idx]
                    .iter()
                    .map(|&v| 10f64.powf(v / 10.0))
                    .sum();
                ltot = 10.0 * spec_sum.log10();

                peaks.retain(|&p| p != ind_s_val);

                delta_ft = 2.0 * (freq_axis_full[1] - freq_axis_full[0]);
                f = fp;
            } else {
                lt = spec[ind_p];
                let (a, b) = critical_band(freqs[ind_p]);
                f1 = a;
                f2 = b;
                low_limit_idx = argmin_abs_diff(freqs, f1);
                high_limit_idx = argmin_abs_diff(freqs, f2);
                let spec_sum: f64 = spec[low_limit_idx..high_limit_idx]
                    .iter()
                    .map(|&v| 10f64.powf(v / 10.0))
                    .sum();
                ltot = 10.0 * spec_sum.log10();
                delta_ft = freqs[1] - freqs[0];
                f = freqs[ind_p];
            }
        } else {
            lt = peak_level(freqs, spec, ind_p);
            let (a, b) = critical_band(freqs[ind_p]);
            f1 = a;
            f2 = b;
            low_limit_idx = argmin_abs_diff(freqs, f1);
            high_limit_idx = argmin_abs_diff(freqs, f2);
            let spec_sum: f64 = spec[low_limit_idx..high_limit_idx]
                .iter()
                .map(|&v| 10f64.powf(v / 10.0))
                .sum();
            ltot = 10.0 * spec_sum.log10();
            delta_ft = freqs[1] - freqs[0];
            f = freqs[ind_p];
        }

        let delta_fc = f2 - f1;
        // Reproduced quirk: `low_limit_idx`/`high_limit_idx` are positions
        // in the *filtered* `freqs` array, but Python indexes the
        // *unfiltered* `frs` with them directly rather than adjusting for
        // the filter's offset — real MoSQITo behaviour, not "fixed" here.
        let delta_ftot = freq_axis_full[high_limit_idx] - freq_axis_full[low_limit_idx];
        let ln = 10.0 * (10f64.powf(ltot / 10.0) - 10f64.powf(lt / 10.0)).log10()
            + 10.0 * (delta_fc / (delta_ftot - delta_ft)).log10();

        let delta_t = lt - ln;
        if delta_t > 0.0 {
            tones_freqs.push(f);
            tnr.push(delta_t);
            let prom = if f < 1000.0 {
                delta_t >= 8.0 + 8.33 * (1000.0 / f).log10()
            } else {
                delta_t >= 8.0
            };
            prominence.push(prom);
        }

        peaks.retain(|&p| p != ind_p);
    }

    (tones_freqs, tnr, prominence)
}
