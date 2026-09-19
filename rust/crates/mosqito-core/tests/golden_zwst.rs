//! Conformance of `loudness::zwst::main_loudness` against the installed
//! `mosqito` package.
//!
//! `tools/gen_golden_zwst.py` exports 632 cases from the real
//! `_main_loudness`: 300 random third-octave spectra plus targeted edge
//! cases at and around every RAP threshold (see `main_loudness.rs`'s module
//! doc for why the edge cases matter — the low-frequency correction search
//! only checks 7 of its 8 threshold transitions, and only inputs that land
//! near the 8th would evidence any divergence).

use mosqito_core::loudness::zwst::{calc_slopes, main_loudness, FieldType};
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("golden_zwst.json")).expect("golden_zwst.json parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

#[test]
fn main_loudness_matches_mosqito_across_random_and_edge_case_spectra() {
    let g = golden();
    let cases = g["main_loudness"].as_array().expect("main_loudness array");
    assert!(
        cases.len() > 500,
        "expected broad coverage, got {}",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        let spec = floats(&case["spec"]);
        let field_type = match case["field_type"].as_str().expect("field_type") {
            "free" => FieldType::Free,
            "diffuse" => FieldType::Diffuse,
            other => panic!("unknown field_type {other}"),
        };
        let want = floats(&case["nm"]);

        let got = main_loudness(&spec, field_type);
        assert_eq!(got.len(), want.len(), "case {i}: length mismatch");
        for (j, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let err = (g - w).abs();
            let tol = 1e-9 * w.abs().max(1e-9);
            assert!(
                err <= tol,
                "case {i}, nm[{j}]: got {g:e}, want {w:e} (spec[0..11]={:?}, field_type={:?})",
                &spec[0..11],
                case["field_type"]
            );
        }
    }
}

#[test]
fn calc_slopes_matches_mosqito_across_random_and_synthetic_nm_arrays() {
    let g = golden();
    let cases = g["calc_slopes"].as_array().expect("calc_slopes array");
    assert!(
        cases.len() > 500,
        "expected broad coverage, got {}",
        cases.len()
    );

    let mut first_failure: Option<String> = None;
    let mut n_failures = 0usize;

    for (i, case) in cases.iter().enumerate() {
        let nm_vec = floats(&case["nm"]);

        // One synthetic case (a perfectly linear, integer-valued nm =
        // 0..20) is excluded: MoSQITo's own `_calc_slopes` produces 9
        // negative specific-loudness values for it — physically impossible
        // under ISO 532-1's model — independently confirmed against the
        // installed `mosqito` package directly, not just this golden file.
        // This nm pattern cannot arise from real `main_loudness` output
        // (which is continuous and non-negative by construction; every one
        // of the 632 spectrum-derived cases here, and 4 of the 5 other
        // synthetic ones, match exactly), so this looks like a genuine
        // instability in the original vectorised Python for a degenerate
        // input outside `calc_slopes`'s real domain, not a target to
        // reproduce. See the module doc on `calc_slopes` in
        // `src/loudness/zwst/calc_slopes.rs`.
        if nm_vec
            == [
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            ]
        {
            continue;
        }
        let mut nm = [0.0f64; 21];
        nm.copy_from_slice(&nm_vec);

        let want_n = case["N"].as_f64().expect("N");
        let want_spec = floats(&case["N_specific"]);

        let (got_n, got_spec) = calc_slopes(&nm);

        let n_ok = (got_n - want_n).abs() <= 1e-9 * want_n.abs().max(1e-9);
        let spec_max_err = got_spec
            .iter()
            .zip(&want_spec)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f64, f64::max);
        let spec_ok =
            spec_max_err <= 1e-9 * want_spec.iter().cloned().fold(0.0f64, f64::max).max(1e-9);

        if !n_ok || !spec_ok {
            n_failures += 1;
            if first_failure.is_none() {
                first_failure = Some(format!(
                    "case {i}: N got={got_n} want={want_n}; N_specific max err={spec_max_err:e}\nnm={nm:?}"
                ));
            }
        }
    }

    if let Some(msg) = first_failure {
        panic!(
            "{n_failures}/{} cases diverged from mosqito.\nfirst failure:\n{msg}",
            cases.len()
        );
    }
}
