//! Conformance of Daniel & Weber roughness against real MoSQITo.
//!
//! `tools/gen_golden_roughness_dw.py` captures MoSQITo's `_H_weighting`,
//! `_gzi_weighting`, `_ear_filter_coeff`, `_roughness_dw_main_calc`, and the
//! two top-level entry points into `golden_roughness_dw.json`; these tests
//! assert the Rust port reproduces them.
//!
//! Regenerate with `.venv/bin/python tools/gen_golden_roughness_dw.py`.

use mosqito_core::roughness::dw::{
    ear_filter_coeff, gzi_weighting, h_weighting, roughness_dw, roughness_dw_freq,
    roughness_dw_main_calc,
};
use num_complex::Complex64;
use serde_json::Value;

fn golden() -> Value {
    let raw = include_str!("golden_roughness_dw.json");
    serde_json::from_str(raw).expect("golden_roughness_dw.json parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected a JSON array")
        .iter()
        .map(|x| x.as_f64().expect("expected a number"))
        .collect()
}

#[track_caller]
fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= tol * scale.max(w.abs()),
            "{what}: index {i} differs by {err:e} (got {g:e}, want {w:e})"
        );
    }
}

#[test]
fn h_weighting_matches_mosqito() {
    let g = golden();
    let c = &g["h_weighting"];
    let n = c["n"].as_u64().unwrap() as usize;
    let fs = c["fs"].as_f64().unwrap();
    let want_rows = c["h"].as_array().unwrap();

    let got = h_weighting(n, fs);
    assert_eq!(got.len(), 47);
    for (i, want_row) in want_rows.iter().enumerate() {
        let want = floats(want_row);
        assert_close(&got[i], &want, 1e-9, &format!("h_weighting row {i}"));
    }
}

#[test]
fn gzi_weighting_matches_mosqito() {
    let g = golden();
    let c = &g["gzi_weighting"];
    let zi = floats(&c["zi"]);
    let want = floats(&c["gzi"]);
    let got = gzi_weighting(&zi);
    assert_close(&got, &want, 1e-9, "gzi_weighting");
}

#[test]
fn ear_filter_coeff_matches_mosqito() {
    let g = golden();
    let c = &g["ear_filter_coeff"];
    let bark = floats(&c["bark_axis"]);
    let want = floats(&c["a0"]);
    let got = ear_filter_coeff(&bark);
    assert_close(&got, &want, 1e-9, "ear_filter_coeff");
}

#[test]
fn main_calc_matches_mosqito() {
    let g = golden();
    let c = &g["main_calc"];
    let spec_re = floats(&c["spec_re"]);
    let spec_im = floats(&c["spec_im"]);
    let freq_axis = floats(&c["freq_axis"]);
    let fs = c["fs"].as_f64().unwrap();
    let want_r = c["R"].as_f64().unwrap();
    let want_spec = floats(&c["R_spec"]);
    let want_bark = floats(&c["bark"]);

    let spec: Vec<Complex64> = spec_re
        .iter()
        .zip(&spec_im)
        .map(|(&re, &im)| Complex64::new(re, im))
        .collect();

    let zi: Vec<f64> = (1..=47).map(|i| i as f64 / 2.0).collect();
    let gzi = gzi_weighting(&zi);
    let h_weight = h_weighting(spec.len() * 2, fs);

    let (got_r, got_spec, got_bark) =
        roughness_dw_main_calc(&spec, &freq_axis, fs, &gzi, &h_weight);
    assert!((got_r - want_r).abs() < 1e-6 * want_r.abs().max(1.0));
    assert_close(&got_spec, &want_spec, 1e-6, "main_calc R_spec");
    assert_close(&got_bark, &want_bark, 1e-9, "main_calc bark");
}

#[test]
fn roughness_dw_matches_mosqito() {
    let g = golden();
    let c = &g["roughness_dw"];
    let signal = floats(&c["signal"]);
    let fs = c["fs"].as_f64().unwrap();
    let overlap = c["overlap"].as_f64().unwrap();
    let want_r = floats(&c["R"]);
    let want_bark = floats(&c["bark"]);
    let want_time = floats(&c["time"]);

    let (got_r, got_spec, got_bark, got_time) = roughness_dw(&signal, fs, overlap);
    assert_close(&got_r, &want_r, 1e-6, "roughness_dw R");
    assert_close(&got_bark, &want_bark, 1e-9, "roughness_dw bark");
    assert_close(&got_time, &want_time, 1e-9, "roughness_dw time");

    let want_rows = c["R_spec"].as_array().unwrap();
    assert_eq!(got_spec.nrows(), want_rows.len());
    for (z, want_row) in want_rows.iter().enumerate() {
        let want = floats(want_row);
        let got = got_spec.row(z).to_vec();
        assert_close(&got, &want, 1e-6, &format!("roughness_dw R_spec row {z}"));
    }
}

#[test]
fn roughness_dw_freq_matches_mosqito() {
    let g = golden();
    let c = &g["roughness_dw_freq"];
    let spectrum = floats(&c["spectrum"]);
    let freqs = floats(&c["freqs"]);
    let want_r = c["R"].as_f64().unwrap();
    let want_spec = floats(&c["R_spec"]);
    let want_bark = floats(&c["bark"]);

    let (got_r, got_spec, got_bark) = roughness_dw_freq(&spectrum, &freqs);
    assert!((got_r - want_r).abs() < 1e-6 * want_r.abs().max(1.0));
    assert_close(&got_spec, &want_spec, 1e-6, "roughness_dw_freq R_spec");
    assert_close(&got_bark, &want_bark, 1e-9, "roughness_dw_freq bark");
}
