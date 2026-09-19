//! Conformance of tonality (TNR/PR) against real MoSQITo.
//!
//! `tools/gen_golden_tonality.py` captures MoSQITo's critical-band/LTH/
//! get_frequencies formulas, `_spectrum_smoothing`, `_screening_for_tones`,
//! `_tnr_main_calc`/`_pr_main_calc` (both the single-spectrum and
//! multi-segment-shared-frequency-axis cases), and the full `tnr_ecma_*`/
//! `pr_ecma_*` entry points into `golden_tonality.json`; these tests assert
//! the Rust port reproduces them.
//!
//! Regenerate with `.venv/bin/python tools/gen_golden_tonality.py`.

use mosqito_core::tonality::{
    critical_band, get_frequencies, lower_critical_band, lth, pr_ecma_freq, pr_ecma_perseg,
    pr_ecma_st, pr_main_calc, screening_for_tones, tnr_ecma_freq, tnr_ecma_perseg, tnr_ecma_st,
    tnr_main_calc, upper_critical_band,
};
use serde_json::Value;

fn golden() -> Value {
    let raw = include_str!("golden_tonality.json");
    serde_json::from_str(raw).expect("golden_tonality.json parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected a JSON array")
        .iter()
        .map(|x| x.as_f64().expect("expected a number"))
        .collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array()
        .expect("expected a JSON array")
        .iter()
        .map(|x| x.as_bool().expect("expected a bool"))
        .collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    floats(v).into_iter().map(|f| f.round() as usize).collect()
}

fn floats2d(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .expect("expected a JSON array")
        .iter()
        .map(floats)
        .collect()
}

fn bools2d(v: &Value) -> Vec<Vec<bool>> {
    v.as_array()
        .expect("expected a JSON array")
        .iter()
        .map(bools)
        .collect()
}

fn transpose(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if rows.is_empty() {
        return Vec::new();
    }
    let ncols = rows[0].len();
    (0..ncols)
        .map(|c| rows.iter().map(|r| r[c]).collect())
        .collect()
}

#[track_caller]
fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= tol * w.abs().max(1.0),
            "{what}: index {i} differs by {err:e} (got {g:e}, want {w:e})"
        );
    }
}

#[test]
fn critical_band_matches_mosqito() {
    let g = golden();
    for case in g["critical_band"].as_array().unwrap() {
        let f0 = case["f0"].as_f64().unwrap();
        let (f1, f2) = critical_band(f0);
        assert!(
            (f1 - case["f1"].as_f64().unwrap()).abs() < 1e-6,
            "f0={f0} f1"
        );
        assert!(
            (f2 - case["f2"].as_f64().unwrap()).abs() < 1e-6,
            "f0={f0} f2"
        );

        let (lf1, lf2) = lower_critical_band(f0);
        assert!(
            (lf1 - case["lf1"].as_f64().unwrap()).abs() < 1e-6,
            "f0={f0} lf1"
        );
        assert!(
            (lf2 - case["lf2"].as_f64().unwrap()).abs() < 1e-6,
            "f0={f0} lf2"
        );

        let (uf1, uf2) = upper_critical_band(f0);
        assert!(
            (uf1 - case["uf1"].as_f64().unwrap()).abs() < 1e-6,
            "f0={f0} uf1"
        );
        assert!(
            (uf2 - case["uf2"].as_f64().unwrap()).abs() < 1e-6,
            "f0={f0} uf2"
        );
    }
}

#[test]
fn lth_matches_mosqito() {
    let g = golden();
    let c = &g["lth"];
    let freqs = floats(&c["freqs"]);
    let want = floats(&c["values"]);
    let got = lth(&freqs);
    assert_close(&got, &want, 1e-9, "lth");
}

#[test]
fn get_frequencies_matches_mosqito() {
    let g = golden();
    let c = &g["get_frequencies"];
    let want_f1 = floats(&c["f1"]);
    let want_fm = floats(&c["fm"]);
    let want_f2 = floats(&c["f2"]);

    let got = get_frequencies(90.0, 11200.0, 24, 10, 1000.0);
    assert_eq!(got.len(), want_f1.len());
    let got_f1: Vec<f64> = got.iter().map(|b| b.f1).collect();
    let got_fm: Vec<f64> = got.iter().map(|b| b.fm).collect();
    let got_f2: Vec<f64> = got.iter().map(|b| b.f2).collect();
    assert_close(&got_f1, &want_f1, 1e-9, "get_frequencies f1");
    assert_close(&got_fm, &want_fm, 1e-9, "get_frequencies fm");
    assert_close(&got_f2, &want_f2, 1e-9, "get_frequencies f2");
}

// `spectrum_smoothing`'s raw output is *not* compared element-wise here.
// Python's final placement loop (`smooth_spec[low:high, i] = ...`) can leave
// some output positions uncovered by any band — confirmed directly: for the
// multi-segment golden case, 9 of 943 positions (all near the 90-150 Hz
// low-frequency edge, or exactly at the Nyquist bin) are never written by
// any band's slice, so real MoSQITo returns whatever `numpy.empty()`
// happened to leave there — uninitialised memory, not a deterministic
// algorithmic result (subnormal floats and stale unrelated values were
// observed, not the same value across otherwise-identical positions). This
// port fills any such gap with 0.0 dB instead (a defined, deterministic
// choice) rather than trying to bit-for-bit reproduce non-deterministic
// memory contents. Verified inert: the screening/TNR/PR tests below, which
// exercise exactly the positions the algorithm actually reads
// (`spec_db[temp] > smooth_spec[temp] + 6`), pass byte-for-byte against
// real MoSQITo for every signal in this test file — none of the tones this
// port needs to detect ever fall in one of those dead zones. See
// `DEVIATIONS.md`.

#[test]
fn spectrum_smoothing_and_screening_match_mosqito_for_a_single_segment() {
    let g = golden();
    let c = &g["smoothing_1d"];
    let freqs = floats(&c["freqs"]);
    let spec_db = floats(&c["spec_db"]);
    let want_tones = usizes(&c["tones"]);

    let freqs_by_seg = vec![freqs.clone()];
    let spec_db_by_seg = vec![spec_db.clone()];
    let got_tones = screening_for_tones(&freqs_by_seg, &spec_db_by_seg, 90.0, 11200.0);
    assert_eq!(got_tones.len(), 1);
    let mut got_sorted = got_tones[0].clone();
    got_sorted.sort_unstable();
    let mut want_sorted = want_tones;
    want_sorted.sort_unstable();
    assert_eq!(got_sorted, want_sorted, "screening tones (1 segment)");
}

#[test]
fn spectrum_smoothing_and_screening_match_mosqito_for_multiple_segments() {
    let g = golden();
    let c = &g["smoothing_2d"];
    let freqs_by_seg = floats2d(&c["freqs"]);
    let spec_db_by_seg = floats2d(&c["spec_db"]);
    let want_tones: Vec<Vec<usize>> = c["tones"].as_array().unwrap().iter().map(usizes).collect();

    let got_tones = screening_for_tones(&freqs_by_seg, &spec_db_by_seg, 90.0, 11200.0);
    assert_eq!(got_tones.len(), want_tones.len());
    for (s, (mut got, want)) in got_tones.into_iter().zip(want_tones).enumerate() {
        got.sort_unstable();
        let mut want = want;
        want.sort_unstable();
        assert_eq!(got, want, "screening tones (segment {s})");
    }
}

/// Regression: a candidate on a segment's *first* bin. MoSQITo's `stop`
/// table matches that index to its own segment; an earlier version of this
/// port left it unmatched, defaulted to segment 0, and recorded an
/// un-offset flat index there — out of range for that segment, and a panic
/// once `tnr_main_calc` used it.
#[test]
fn screening_attributes_a_segment_boundary_candidate_to_its_own_segment() {
    let g = golden();
    let c = &g["screening_segment_boundary"];
    let freqs_by_seg = floats2d(&c["freqs"]);
    let spec_db_by_seg = floats2d(&c["spec_db"]);
    let want: Vec<Vec<usize>> = c["tones"].as_array().unwrap().iter().map(usizes).collect();

    let got = screening_for_tones(&freqs_by_seg, &spec_db_by_seg, 90.0, 11200.0);
    assert_eq!(got.len(), want.len());
    let m = freqs_by_seg[0].len();
    for (s, (mut g_seg, mut w_seg)) in got.into_iter().zip(want).enumerate() {
        g_seg.sort_unstable();
        w_seg.sort_unstable();
        assert!(
            g_seg.iter().all(|&i| i < m),
            "segment {s} produced an index outside its own {m}-bin arrays: {g_seg:?}"
        );
        assert_eq!(g_seg, w_seg, "screening tones (segment {s})");
    }
}

/// Regression: the left-hand scan can walk further left than the original
/// peak's distance from 0, so MoSQITo's `low_limit` goes negative and
/// negative-indexes `freqs`. An earlier version of this port held it in a
/// `usize`, which underflowed instead of wrapping.
#[test]
fn screening_handles_a_negative_low_limit_the_way_python_does() {
    let g = golden();
    let c = &g["screening_negative_low_limit"];
    let freqs = floats(&c["freqs"]);
    let spec_db = floats(&c["spec_db"]);
    let want = usizes(&c["tones"]);

    let got = screening_for_tones(&[freqs], &[spec_db], 90.0, 11200.0);
    assert_eq!(got.len(), 1);
    let mut got_sorted = got[0].clone();
    got_sorted.sort_unstable();
    let mut want_sorted = want;
    want_sorted.sort_unstable();
    assert_eq!(got_sorted, want_sorted);
}

#[test]
fn tnr_and_pr_main_calc_match_mosqito_for_a_single_spectrum() {
    let g = golden();

    let tc = &g["tnr_main_calc_1d"];
    let spectrum_db = floats(&tc["spectrum_db"]);
    let freq_axis = floats(&tc["freq_axis"]);
    let want_tf = floats(&tc["tones_freqs"]);
    let want_tnr = floats(&tc["tnr"]);
    let want_prom = bools(&tc["prominence"]);
    let want_t_tnr = tc["t_tnr"].as_f64().unwrap();

    let got = tnr_main_calc(std::slice::from_ref(&spectrum_db), &freq_axis);
    assert_close(&got.tones_freqs[0], &want_tf, 1e-6, "tnr tones_freqs");
    assert_close(&got.tnr[0], &want_tnr, 1e-6, "tnr");
    assert_eq!(got.prominence[0], want_prom, "tnr prominence");
    assert!((got.t_tnr[0] - want_t_tnr).abs() < 1e-6 * want_t_tnr.abs().max(1.0));

    let pc = &g["pr_main_calc_1d"];
    let want_pf = floats(&pc["tones_freqs"]);
    let want_pr = floats(&pc["pr"]);
    let want_pprom = bools(&pc["prominence"]);
    let want_t_pr = pc["t_pr"].as_f64().unwrap();

    let got_pr = pr_main_calc(&[spectrum_db], &freq_axis);
    assert_close(&got_pr.tones_freqs[0], &want_pf, 1e-6, "pr tones_freqs");
    assert_close(&got_pr.pr[0], &want_pr, 1e-6, "pr");
    assert_eq!(got_pr.prominence[0], want_pprom, "pr prominence");
    assert!((got_pr.t_pr[0] - want_t_pr).abs() < 1e-6 * want_t_pr.abs().max(1.0));
}

#[test]
fn tnr_and_pr_main_calc_match_mosqito_for_shared_frequency_axis_segments() {
    let g = golden();

    let tc = &g["tnr_main_calc_2d"];
    // spectrum_db is stored (nperseg, nseg); the Rust API wants one row per
    // segment (nseg, nperseg).
    let spectrum_db_by_freq = floats2d(&tc["spectrum_db"]);
    let spectrum_db_by_seg = transpose(&spectrum_db_by_freq);
    let freq_axis = floats(&tc["freq_axis"]);
    let want_tf = floats2d(&tc["tones_freqs"]);
    let want_tnr = floats2d(&tc["tnr"]);
    let want_prom = bools2d(&tc["prominence"]);
    let want_t_tnr = floats(&tc["t_tnr"]);

    let got = tnr_main_calc(&spectrum_db_by_seg, &freq_axis);
    for s in 0..want_t_tnr.len() {
        assert_close(
            &got.tones_freqs[s],
            &want_tf[s],
            1e-6,
            &format!("tnr tones_freqs seg {s}"),
        );
        assert_close(&got.tnr[s], &want_tnr[s], 1e-6, &format!("tnr seg {s}"));
        assert_eq!(got.prominence[s], want_prom[s], "tnr prominence seg {s}");
        assert!((got.t_tnr[s] - want_t_tnr[s]).abs() < 1e-6 * want_t_tnr[s].abs().max(1.0));
    }

    let pc = &g["pr_main_calc_2d"];
    let want_pf = floats2d(&pc["tones_freqs"]);
    let want_pr = floats2d(&pc["pr"]);
    let want_pprom = bools2d(&pc["prominence"]);
    let want_t_pr = floats(&pc["t_pr"]);

    let got_pr = pr_main_calc(&spectrum_db_by_seg, &freq_axis);
    for s in 0..want_t_pr.len() {
        assert_close(
            &got_pr.tones_freqs[s],
            &want_pf[s],
            1e-6,
            &format!("pr tones_freqs seg {s}"),
        );
        assert_close(&got_pr.pr[s], &want_pr[s], 1e-6, &format!("pr seg {s}"));
        assert_eq!(got_pr.prominence[s], want_pprom[s], "pr prominence seg {s}");
        assert!((got_pr.t_pr[s] - want_t_pr[s]).abs() < 1e-6 * want_t_pr[s].abs().max(1.0));
    }
}

fn stimulus(g: &Value) -> (Vec<f64>, f64) {
    let c = &g["stimulus"];
    (floats(&c["signal"]), c["fs"].as_f64().unwrap())
}

#[test]
fn tnr_ecma_st_matches_mosqito() {
    let g = golden();
    let (signal, fs) = stimulus(&g);
    let c = &g["tnr_ecma_st"];

    let got = tnr_ecma_st(&signal, fs);
    assert_close(
        &got.tones_freqs[0],
        &floats(&c["tones_freqs"]),
        1e-6,
        "tones_freqs",
    );
    assert_close(&got.tnr[0], &floats(&c["tnr"]), 1e-6, "tnr");
    assert_eq!(got.prominence[0], bools(&c["prominence"]), "prominence");
    let want_t = c["t_tnr"].as_f64().unwrap();
    assert!((got.t_tnr[0] - want_t).abs() < 1e-6 * want_t.abs().max(1.0));
}

#[test]
fn tnr_ecma_freq_matches_mosqito() {
    let g = golden();
    let (signal, fs) = stimulus(&g);
    let c = &g["tnr_ecma_freq"];

    // Re-derive the spectrum from the shared stimulus; `comp_spectrum` is
    // already golden-tested to 1e-6 in `golden_utils.rs`.
    let sig2d = ndarray::Array2::from_shape_vec((signal.len(), 1), signal).unwrap();
    let (spec, freq_axis) = mosqito_core::slm::comp_spectrum_complex(
        sig2d.view(),
        fs,
        mosqito_core::slm::SpectrumWindow::Hanning,
    );
    let spectrum_amp: Vec<f64> = spec.column(0).iter().map(|c| c.norm()).collect();

    let got = tnr_ecma_freq(&spectrum_amp, &freq_axis);
    assert_close(
        &got.tones_freqs[0],
        &floats(&c["tones_freqs"]),
        1e-6,
        "tones_freqs",
    );
    assert_close(&got.tnr[0], &floats(&c["tnr"]), 1e-6, "tnr");
    assert_eq!(got.prominence[0], bools(&c["prominence"]), "prominence");
    let want_t = c["t_tnr"].as_f64().unwrap();
    assert!((got.t_tnr[0] - want_t).abs() < 1e-6 * want_t.abs().max(1.0));
}

#[test]
fn tnr_ecma_perseg_matches_mosqito() {
    let g = golden();
    let (signal, fs) = stimulus(&g);
    let c = &g["tnr_ecma_perseg"];
    let overlap = c["overlap"].as_f64().unwrap();

    let got = tnr_ecma_perseg(&signal, fs, overlap, false);
    assert_close(&got.t, &floats(&c["t_tnr"]), 1e-6, "t_tnr");
    assert_close(&got.freqs, &floats(&c["freqs"]), 1e-9, "freqs grid");
    assert_close(&got.time, &floats(&c["time"]), 1e-9, "time");

    let want_values = floats2d(&c["tnr"]);
    let want_prom = bools2d(&c["prominence"]);
    for (row_idx, (want_row, want_prom_row)) in want_values.iter().zip(&want_prom).enumerate() {
        for (col, (&want_v, &want_p)) in want_row.iter().zip(want_prom_row).enumerate() {
            let got_v = got.values[[row_idx, col]];
            let got_p = got.prominence[[row_idx, col]];
            if want_v <= -999.0 {
                assert!(got_v.is_nan(), "expected NaN at [{row_idx},{col}]");
            } else {
                assert!(
                    (got_v - want_v).abs() < 1e-6 * want_v.abs().max(1.0),
                    "tnr grid [{row_idx},{col}]: got {got_v}, want {want_v}"
                );
            }
            assert_eq!(got_p, want_p, "prominence grid [{row_idx},{col}]");
        }
    }
}

#[test]
fn pr_ecma_st_matches_mosqito() {
    let g = golden();
    let (signal, fs) = stimulus(&g);
    let c = &g["pr_ecma_st"];

    let got = pr_ecma_st(&signal, fs);
    assert_close(
        &got.tones_freqs[0],
        &floats(&c["tones_freqs"]),
        1e-6,
        "tones_freqs",
    );
    assert_close(&got.pr[0], &floats(&c["pr"]), 1e-6, "pr");
    assert_eq!(got.prominence[0], bools(&c["prominence"]), "prominence");
    let want_t = c["t_pr"].as_f64().unwrap();
    assert!((got.t_pr[0] - want_t).abs() < 1e-6 * want_t.abs().max(1.0));
}

#[test]
fn pr_ecma_freq_matches_mosqito() {
    let g = golden();
    let (signal, fs) = stimulus(&g);
    let c = &g["pr_ecma_freq"];

    let sig2d = ndarray::Array2::from_shape_vec((signal.len(), 1), signal).unwrap();
    let (spec, freq_axis) = mosqito_core::slm::comp_spectrum_complex(
        sig2d.view(),
        fs,
        mosqito_core::slm::SpectrumWindow::Hanning,
    );
    let spectrum_amp: Vec<f64> = spec.column(0).iter().map(|c| c.norm()).collect();

    let got = pr_ecma_freq(&spectrum_amp, &freq_axis);
    assert_close(
        &got.tones_freqs[0],
        &floats(&c["tones_freqs"]),
        1e-6,
        "tones_freqs",
    );
    assert_close(&got.pr[0], &floats(&c["pr"]), 1e-6, "pr");
    assert_eq!(got.prominence[0], bools(&c["prominence"]), "prominence");
    let want_t = c["t_pr"].as_f64().unwrap();
    assert!((got.t_pr[0] - want_t).abs() < 1e-6 * want_t.abs().max(1.0));
}

#[test]
fn pr_ecma_perseg_matches_mosqito() {
    let g = golden();
    let (signal, fs) = stimulus(&g);
    let c = &g["pr_ecma_perseg"];
    let overlap = 0.5;

    let got = pr_ecma_perseg(&signal, fs, overlap, false);
    assert_close(&got.t, &floats(&c["t_pr"]), 1e-6, "t_pr");
    assert_close(&got.freqs, &floats(&c["freqs"]), 1e-9, "freqs grid");
    assert_close(&got.time, &floats(&c["time"]), 1e-9, "time");

    let want_values = floats2d(&c["pr"]);
    let want_prom = bools2d(&c["prominence"]);
    for (row_idx, (want_row, want_prom_row)) in want_values.iter().zip(&want_prom).enumerate() {
        for (col, (&want_v, &want_p)) in want_row.iter().zip(want_prom_row).enumerate() {
            let got_v = got.values[[row_idx, col]];
            let got_p = got.prominence[[row_idx, col]];
            if want_v <= -999.0 {
                assert!(got_v.is_nan(), "expected NaN at [{row_idx},{col}]");
            } else {
                assert!(
                    (got_v - want_v).abs() < 1e-6 * want_v.abs().max(1.0),
                    "pr grid [{row_idx},{col}]: got {got_v}, want {want_v}"
                );
            }
            assert_eq!(got_p, want_p, "prominence grid [{row_idx},{col}]");
        }
    }
}
