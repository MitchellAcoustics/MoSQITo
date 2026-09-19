//! Conformance of the `loudness::zwtv` pipeline's own stages, and the full
//! pipeline on short signals, against the installed `mosqito` package.
//!
//! `tools/gen_golden_zwtv.py` exports these directly from real
//! `_nl_loudness`, `_third_octave_levels` and `loudness_zwtv`. `_nl_loudness`
//! in particular is checked on short (`ntime` as low as 2) matrices, where
//! the `col = 0` negative-index wraparound documented in
//! `nonlinear_decay.rs` is most visible in the output — if this port's
//! reproduction of that wraparound were wrong, these short cases would be
//! the ones to show it.

use mosqito_core::loudness::zwst::FieldType;
use mosqito_core::loudness::zwtv::{loudness_zwtv, nl_loudness, third_octave_levels};
use ndarray::Array2;
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("golden_zwtv.json")).expect("golden_zwtv.json parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn matrix(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("rows").iter().map(floats).collect()
}

fn to_array2(rows: &[Vec<f64>]) -> Array2<f64> {
    let nrows = rows.len();
    let ncols = rows[0].len();
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    for (r, row) in rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            out[[r, c]] = v;
        }
    }
    out
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    let err = (got - want).abs();
    let bound = tol * want.abs().max(tol);
    assert!(
        err <= bound,
        "{what}: got {got:e}, want {want:e} (err {err:e})"
    );
}

#[test]
fn nl_loudness_matches_mosqito_including_the_first_frame_wraparound() {
    let g = golden();
    let cases = g["nl_loudness"].as_array().expect("nl_loudness array");
    assert!(
        cases.len() > 20,
        "expected broad coverage, got {}",
        cases.len()
    );

    for (i, case) in cases.iter().enumerate() {
        let core = to_array2(&matrix(&case["core_loudness"]));
        let want = to_array2(&matrix(&case["nl_loudness"]));

        let got = nl_loudness(&core);
        assert_eq!(got.shape(), want.shape(), "case {i}: shape mismatch");
        for row in 0..got.nrows() {
            for col in 0..got.ncols() {
                assert_close(
                    got[[row, col]],
                    want[[row, col]],
                    1e-9,
                    &format!("case {i}, [{row},{col}]"),
                );
            }
        }
    }
}

#[test]
fn third_octave_levels_matches_mosqito() {
    let g = golden();
    let cases = g["third_octave_levels"]
        .as_array()
        .expect("third_octave_levels array");
    assert!(!cases.is_empty());

    for (i, case) in cases.iter().enumerate() {
        let sig = floats(&case["sig"]);
        let want_levels = to_array2(&matrix(&case["levels"]));
        let want_time = floats(&case["time_axis"]);

        let (levels, time_axis, _freq) = third_octave_levels(&sig, 48000.0).expect("valid fs");
        assert_eq!(
            levels.shape(),
            want_levels.shape(),
            "case {i}: shape mismatch"
        );
        for row in 0..levels.nrows() {
            for col in 0..levels.ncols() {
                assert_close(
                    levels[[row, col]],
                    want_levels[[row, col]],
                    1e-7,
                    &format!("case {i}, levels[{row},{col}]"),
                );
            }
        }
        assert_eq!(
            time_axis.len(),
            want_time.len(),
            "case {i}: time_axis length"
        );
        for (j, (&g, &w)) in time_axis.iter().zip(&want_time).enumerate() {
            assert_close(g, w, 1e-9, &format!("case {i}, time_axis[{j}]"));
        }
    }
}

#[test]
fn loudness_zwtv_matches_mosqito_end_to_end_on_short_signals() {
    let g = golden();
    let cases = g["loudness_zwtv"].as_array().expect("loudness_zwtv array");
    assert!(!cases.is_empty());

    for (i, case) in cases.iter().enumerate() {
        let signal = floats(&case["signal"]);
        let fs = case["fs"].as_f64().expect("fs");
        let want_n = floats(&case["N"]);
        let want_spec = to_array2(&matrix(&case["N_specific"]));
        let want_time = floats(&case["time_axis"]);

        let (n, n_specific, _bark, time_axis) =
            loudness_zwtv(&signal, fs, FieldType::Free).expect("valid pipeline");

        assert_eq!(n.len(), want_n.len(), "case {i}: N length");
        for (j, (&g, &w)) in n.iter().zip(&want_n).enumerate() {
            assert_close(g, w, 1e-6, &format!("case {i}, N[{j}]"));
        }
        assert_eq!(
            n_specific.shape(),
            want_spec.shape(),
            "case {i}: N_specific shape"
        );
        for row in 0..n_specific.nrows() {
            for col in 0..n_specific.ncols() {
                assert_close(
                    n_specific[[row, col]],
                    want_spec[[row, col]],
                    1e-6,
                    &format!("case {i}, N_specific[{row},{col}]"),
                );
            }
        }
        assert_eq!(
            time_axis.len(),
            want_time.len(),
            "case {i}: time_axis length"
        );
        for (j, (&g, &w)) in time_axis.iter().zip(&want_time).enumerate() {
            assert_close(g, w, 1e-9, &format!("case {i}, time_axis[{j}]"));
        }
    }
}
