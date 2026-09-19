//! Conformance of `roughness::ecma`'s individual stages — every one *not*
//! affected by `_lowpass_filter.py`'s bug (see `DEVIATIONS.md`) — against
//! the installed `mosqito` package.

use serde_json::Value;

fn golden() -> Value {
    serde_json::from_str(include_str!("golden_roughness_ecma.json"))
        .expect("golden_roughness_ecma.json parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
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
fn von_hann_window_matches_mosqito() {
    let want = floats(&golden()["von_hann_window"]);
    let got = mosqito_core::roughness::ecma::von_hann_window(512);
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert_close(g, w, 1e-12, &format!("von_hann_window[{i}]"));
    }
}

#[test]
fn weighting_functions_match_mosqito() {
    let g = golden();
    let cases = g["weighting"].as_array().expect("weighting array");
    assert!(!cases.is_empty());

    for case in cases {
        let cf = case["centre_freq"].as_f64().unwrap();
        let fmax = case["fmax"].as_f64().unwrap();
        let rmax = case["rmax"].as_f64().unwrap();
        let q2h = case["q2_high"].as_f64().unwrap();
        let q2l = case["q2_low"].as_f64().unwrap();
        let mod_rate = case["mod_rate"].as_f64().unwrap();
        let want_high = case["high_mod_rate_weighting"].as_f64().unwrap();
        let want_low = case["low_mod_rate_weighting"].as_f64().unwrap();

        assert_close(
            mosqito_core::roughness::ecma::f_max(cf),
            fmax,
            1e-9,
            "f_max",
        );
        assert_close(
            mosqito_core::roughness::ecma::r_max(cf),
            rmax,
            1e-9,
            "r_max",
        );
        assert_close(
            mosqito_core::roughness::ecma::q2_high(cf),
            q2h,
            1e-9,
            "q2_high",
        );
        assert_close(
            mosqito_core::roughness::ecma::q2_low(cf),
            q2l,
            1e-9,
            "q2_low",
        );

        let high =
            mosqito_core::roughness::ecma::high_mod_rate_weighting(mod_rate, 1.0, fmax, rmax, q2h);
        assert_close(high, want_high, 1e-9, "high_mod_rate_weighting");

        let low =
            mosqito_core::roughness::ecma::low_mod_rate_weighting(mod_rate, &[1.0, 0.5], fmax, q2l);
        assert_close(low, want_low, 1e-9, "low_mod_rate_weighting");
    }
}

#[test]
fn refinement_matches_mosqito() {
    let g = golden();
    let cases = g["refinement"].as_array().expect("refinement array");
    assert!(cases.len() > 10);

    for case in cases {
        let spec = floats(&case["spec"]);
        let kpi = case["kpi"].as_u64().unwrap() as usize;
        let want_mod_rate = case["mod_rate"].as_f64().unwrap();
        let want_amp = case["amp"].as_f64().unwrap();

        let (mod_rate, amp) = mosqito_core::roughness::ecma::refinement(kpi, &spec);
        assert_close(mod_rate, want_mod_rate, 1e-9, "refinement mod_rate");
        assert_close(amp, want_amp, 1e-9, "refinement amp");
    }
}

#[test]
fn peak_picking_matches_mosqito() {
    let g = golden();
    let cases = g["peak_picking"].as_array().expect("peak_picking array");
    assert!(cases.len() > 10);

    for (i, case) in cases.iter().enumerate() {
        let spec = floats(&case["spec"]);
        let want_f_p = floats(&case["f_p"]);
        let want_a = floats(&case["a"]);

        let (f_p, a) = mosqito_core::roughness::ecma::peak_picking(&spec);
        assert_eq!(f_p.len(), want_f_p.len(), "case {i}: peak count");
        for (j, (&got, &want)) in f_p.iter().zip(&want_f_p).enumerate() {
            assert_close(got, want, 1e-9, &format!("case {i}, f_p[{j}]"));
        }
        for (j, (&got, &want)) in a.iter().zip(&want_a).enumerate() {
            assert_close(got, want, 1e-9, &format!("case {i}, a[{j}]"));
        }
    }
}

#[test]
fn estimate_fund_mod_rate_matches_mosqito() {
    let g = golden();
    let cases = g["estimate_fund_mod_rate"]
        .as_array()
        .expect("estimate_fund_mod_rate array");
    assert!(cases.len() > 10);

    for (i, case) in cases.iter().enumerate() {
        let f_p = floats(&case["f_p"]);
        let ai_tilde = floats(&case["ai_tilde"]);
        let want_mod_rate = case["mod_rate"].as_f64().unwrap();
        let want_a_hat = floats(&case["a_hat"]);

        let (mod_rate, a_hat) =
            mosqito_core::roughness::ecma::estimate_fund_mod_rate(&f_p, &ai_tilde);
        assert_close(
            mod_rate,
            want_mod_rate,
            1e-9,
            &format!("case {i}: mod_rate"),
        );
        assert_eq!(a_hat.len(), want_a_hat.len(), "case {i}: a_hat length");
        for (j, (&got, &want)) in a_hat.iter().zip(&want_a_hat).enumerate() {
            assert_close(got, want, 1e-9, &format!("case {i}, a_hat[{j}]"));
        }
    }
}

#[test]
fn noise_reduction_matches_mosqito() {
    let g = golden();
    let cases = g["noise_reduction"]
        .as_array()
        .expect("noise_reduction array");
    assert!(!cases.is_empty());

    for (i, case) in cases.iter().enumerate() {
        // spectrum: [time][band][bin]
        let spectrum: Vec<Vec<Vec<f64>>> = case["spectrum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_array().unwrap().iter().map(floats).collect())
            .collect();
        let want: Vec<Vec<Vec<f64>>> = case["phi_e"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_array().unwrap().iter().map(floats).collect())
            .collect();

        let got = mosqito_core::roughness::ecma::noise_reduction(&spectrum);
        assert_eq!(got.len(), want.len(), "case {i}: time length");
        for t in 0..got.len() {
            for z in 0..got[t].len() {
                for k in 0..got[t][z].len() {
                    assert_close(
                        got[t][z][k],
                        want[t][z][k],
                        1e-9,
                        &format!("case {i}, phi_e[{t}][{z}][{k}]"),
                    );
                }
            }
        }
    }
}
