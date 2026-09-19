//! Conformance of `loudness::ecma`'s own stages, and the full pipeline on
//! short signals, against the installed `mosqito` package.

use mosqito_core::loudness::ecma::loudness_ecma;
use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("golden_ecma_loudness.json"))
        .expect("golden_ecma_loudness.json parses")
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

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    let err = (got - want).abs();
    let bound = tol * want.abs().max(tol);
    assert!(
        err <= bound,
        "{what}: got {got:e}, want {want:e} (err {err:e})"
    );
}

#[test]
fn auditory_filters_centre_freq_matches_mosqito() {
    let g = golden();
    let want = floats(&g["centre_freq"]);
    let got = mosqito_core::loudness::ecma::auditory_filters_centre_freq();
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert_close(g, w, 1e-12, &format!("centre_freq[{i}]"));
    }
}

#[test]
fn gammatone_matches_mosqito() {
    let g = golden();
    let cases = g["gammatone"].as_array().expect("gammatone array");
    assert!(!cases.is_empty());

    for case in cases {
        let freq = case["freq"].as_f64().expect("freq");
        let want_bm = matrix(&case["bm"]);
        let want_am = matrix(&case["am"]);

        let (bm, am) = mosqito_core::loudness::ecma::gammatone(freq, 48000.0);
        assert_eq!(bm.len(), want_bm.len());
        for (i, (c, w)) in bm.iter().zip(&want_bm).enumerate() {
            assert_close(c.re, w[0], 1e-9, &format!("bm[{i}].re (freq={freq})"));
            assert_close(c.im, w[1], 1e-9, &format!("bm[{i}].im (freq={freq})"));
        }
        assert_eq!(am.len(), want_am.len());
        for (i, (c, w)) in am.iter().zip(&want_am).enumerate() {
            assert_close(c.re, w[0], 1e-9, &format!("am[{i}].re (freq={freq})"));
            assert_close(c.im, w[1], 1e-9, &format!("am[{i}].im (freq={freq})"));
        }
    }
}

#[test]
fn nonlinearity_matches_mosqito() {
    let g = golden();
    let cases = g["nonlinearity"].as_array().expect("nonlinearity array");
    assert!(cases.len() > 20);

    for case in cases {
        let p = case["p"].as_f64().expect("p");
        let want = case["a_prime"].as_f64().expect("a_prime");
        let got = mosqito_core::loudness::ecma::nonlinearity(p);
        assert_close(got, want, 1e-9, &format!("nonlinearity(p={p})"));
    }
}

#[test]
fn loudness_ecma_matches_mosqito_end_to_end_on_short_signals() {
    let g = golden();
    let cases = g["loudness_ecma"].as_array().expect("loudness_ecma array");
    assert!(!cases.is_empty());

    for (i, case) in cases.iter().enumerate() {
        let signal = floats(&case["signal"]);
        let fs = case["fs"].as_f64().expect("fs");
        let sb = case["sb"].as_u64().expect("sb") as usize;
        let sh = case["sh"].as_u64().expect("sh") as usize;
        let want_n = case["N"].as_f64().expect("N");
        let want_n_time = floats(&case["N_time"]);
        let want_n_specific = matrix(&case["N_specific"]);
        let want_time_axis = floats(&case["time_axis"]);

        let (n, n_time, n_specific, _bark, time_axis) = loudness_ecma(&signal, fs, sb, sh);

        assert_close(n, want_n, 1e-6, &format!("case {i}: N"));

        assert_eq!(n_time.len(), want_n_time.len(), "case {i}: N_time length");
        for (j, (&got, &want)) in n_time.iter().zip(&want_n_time).enumerate() {
            assert_close(got, want, 1e-6, &format!("case {i}, N_time[{j}]"));
        }

        assert_eq!(n_specific.nrows(), 53, "case {i}: N_specific band count");
        assert_eq!(
            n_specific.ncols(),
            want_n_time.len(),
            "case {i}: N_specific block count"
        );
        for band in 0..53 {
            let want_row = &want_n_specific[band];
            for (t, &want) in want_row.iter().enumerate() {
                assert_close(
                    n_specific[[band, t]],
                    want,
                    1e-6,
                    &format!("case {i}, N_specific[{band},{t}]"),
                );
            }
        }

        assert_eq!(
            time_axis.len(),
            want_time_axis.len(),
            "case {i}: time_axis length"
        );
        for (j, (&got, &want)) in time_axis.iter().zip(&want_time_axis).enumerate() {
            assert_close(got, want, 1e-9, &format!("case {i}, time_axis[{j}]"));
        }
    }
}
