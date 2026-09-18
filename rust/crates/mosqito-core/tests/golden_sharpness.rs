//! Conformance of `sharpness::din::sharpness_din_from_loudness{,_segmented}`
//! against the installed `mosqito` package.
//!
//! `tools/gen_golden_sharpness.py` exports 320 scalar cases (4 weightings x
//! 80 random (N, N_specific) pairs from real `_main_loudness`/`_calc_slopes`
//! output) and 80 segmented cases (6-segment batches, with two segments
//! forced below the `N < 0.1` masking threshold, including exactly `N = 0`
//! where MoSQITo's own Python divides `0/0` and relies on the mask to
//! discard the resulting NaN).

use mosqito_core::sharpness::din::{
    sharpness_din_from_loudness, sharpness_din_from_loudness_segmented, Weighting,
};
use ndarray::Array2;
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("golden_sharpness.json"))
        .expect("golden_sharpness.json parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn weighting(s: &str) -> Weighting {
    Weighting::parse(s).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn sharpness_din_from_loudness_matches_mosqito_scalar_cases() {
    let g = golden();
    let cases = g["scalar"].as_array().expect("scalar array");
    assert!(
        cases.len() > 250,
        "expected broad coverage, got {}",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        let n = case["N"].as_f64().expect("N");
        let n_specific_vec = floats(&case["N_specific"]);
        let mut n_specific = [0.0f64; 240];
        n_specific.copy_from_slice(&n_specific_vec);
        let w = weighting(case["weighting"].as_str().expect("weighting"));
        let want = case["S"].as_f64().expect("S");

        let got = sharpness_din_from_loudness(n, &n_specific, w);
        let tol = 1e-9 * want.abs().max(1e-9);
        assert!(
            (got - want).abs() <= tol,
            "case {i} ({:?}): got {got:e}, want {want:e} (N={n})",
            case["weighting"]
        );
    }
}

#[test]
fn sharpness_din_from_loudness_segmented_matches_mosqito_including_the_masking_threshold() {
    let g = golden();
    let cases = g["segmented"].as_array().expect("segmented array");
    assert!(
        cases.len() > 60,
        "expected broad coverage, got {}",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        let n = floats(&case["N"]);
        let nseg = n.len();
        let n_specific_flat = case["N_specific"]
            .as_array()
            .expect("N_specific rows")
            .iter()
            .map(floats)
            .collect::<Vec<_>>();
        assert_eq!(
            n_specific_flat.len(),
            240,
            "case {i}: expected 240 bark rows"
        );
        let mut n_specific = Array2::<f64>::zeros((240, nseg));
        for (row, vals) in n_specific_flat.iter().enumerate() {
            for (col, &v) in vals.iter().enumerate() {
                n_specific[[row, col]] = v;
            }
        }
        let w = weighting(case["weighting"].as_str().expect("weighting"));
        let want = floats(&case["S"]);

        let got = sharpness_din_from_loudness_segmented(&n, &n_specific, w);
        assert_eq!(got.len(), want.len(), "case {i}: length mismatch");
        for (seg, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
            let tol = 1e-9 * wv.abs().max(1e-9);
            assert!(
                (gv - wv).abs() <= tol,
                "case {i}, segment {seg} ({:?}): got {gv:e}, want {wv:e} (N={:?})",
                case["weighting"],
                n
            );
        }
    }
}
