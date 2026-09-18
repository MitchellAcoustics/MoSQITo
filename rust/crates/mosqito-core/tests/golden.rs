//! Conformance of the DSP primitives against SciPy.
//!
//! `tools/gen_golden.py` captures SciPy's output for each primitive into
//! `golden.json`; these tests assert the Rust implementations reproduce it.
//! This is the foundation the metric ports rest on — if a filter or resampler
//! drifts from SciPy here, every downstream conformance number moves with it.
//!
//! Regenerate with `.venv/bin/python tools/gen_golden.py` and note the
//! numpy/scipy versions recorded in the file when a value changes.

use mosqito_core::dsp::{
    butter_bandpass_sos, butter_lowpass_sos, cheby1_lowpass, decimate, filtfilt,
    find_peaks_with_prominence, hilbert_envelope, interp, lfilter, lfilter_complex, median, pchip,
    percentile_linear, resample, sosfilt, sosfiltfilt, sosfreqz, Sos,
};
use mosqito_core::num_complex::Complex64;
use serde_json::Value;

fn golden() -> Value {
    let raw = include_str!("golden.json");
    serde_json::from_str(raw).expect("golden.json parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected a JSON array")
        .iter()
        .map(|x| x.as_f64().expect("expected a number"))
        .collect()
}

fn sos_rows(v: &Value) -> Vec<Sos> {
    v.as_array()
        .expect("expected a JSON array of sections")
        .iter()
        .map(|row| {
            let r = floats(row);
            assert_eq!(r.len(), 6, "each section has 6 coefficients");
            [r[0], r[1], r[2], r[3], r[4], r[5]]
        })
        .collect()
}

/// Asserts element-wise agreement using a mixed absolute/relative bound, which
/// is what a signal spanning many orders of magnitude needs.
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
fn lfilter_matches_scipy() {
    let g = golden();
    let c = &g["lfilter"];
    let y = lfilter(&floats(&c["b"]), &floats(&c["a"]), &floats(&c["x"]));
    assert_close(&y, &floats(&c["y"]), 1e-12, "lfilter");
}

#[test]
fn lfilter_with_complex_coefficients_matches_scipy() {
    let g = golden();
    let c = &g["lfilter_complex"];
    let zip_c = |re: &str, im: &str| -> Vec<Complex64> {
        floats(&c[re])
            .into_iter()
            .zip(floats(&c[im]))
            .map(|(r, i)| Complex64::new(r, i))
            .collect()
    };
    let y = lfilter_complex(
        &zip_c("b_re", "b_im"),
        &zip_c("a_re", "a_im"),
        &floats(&c["x"]),
    );
    let (re, im): (Vec<f64>, Vec<f64>) = y.iter().map(|v| (v.re, v.im)).unzip();
    assert_close(&re, &floats(&c["y_re"]), 1e-12, "lfilter_complex real");
    assert_close(&im, &floats(&c["y_im"]), 1e-12, "lfilter_complex imag");
}

#[test]
fn filter_designs_match_scipy_including_section_ordering() {
    let g = golden();
    for case in g["designs"].as_array().expect("designs array") {
        let kind = case["kind"].as_str().expect("kind");
        let want = sos_rows(&case["sos"]);
        let got = match kind {
            "butter_lowpass" => butter_lowpass_sos(
                case["order"].as_u64().expect("order") as usize,
                case["wn"].as_f64().expect("wn"),
            ),
            "butter_bandpass" => butter_bandpass_sos(
                case["order"].as_u64().expect("order") as usize,
                case["low"].as_f64().expect("low"),
                case["high"].as_f64().expect("high"),
            ),
            "cheby1_lowpass" => cheby1_lowpass(
                case["order"].as_u64().expect("order") as usize,
                case["rp"].as_f64().expect("rp"),
                case["wn"].as_f64().expect("wn"),
            ),
            other => panic!("unknown design kind {other}"),
        };
        assert_eq!(got.len(), want.len(), "{kind}: section count");
        let flat_got: Vec<f64> = got.iter().flatten().copied().collect();
        let flat_want: Vec<f64> = want.iter().flatten().copied().collect();
        // Section ordering is part of what is being checked: a different
        // pairing would show up here, not just as a rounding difference.
        assert_close(
            &flat_got,
            &flat_want,
            1e-10,
            &format!("{kind} coefficients"),
        );
    }
}

#[test]
fn sosfilt_matches_scipy() {
    let g = golden();
    let c = &g["sosfilt"];
    let y = sosfilt(&sos_rows(&c["sos"]), &floats(&c["x"]));
    assert_close(&y, &floats(&c["y"]), 1e-12, "sosfilt");
}

#[test]
fn filtfilt_matches_scipy() {
    let g = golden();
    let c = &g["filtfilt"];
    let y = filtfilt(&floats(&c["b"]), &floats(&c["a"]), &floats(&c["x"]))
        .expect("signal is long enough for the padding");
    assert_close(&y, &floats(&c["y"]), 1e-12, "filtfilt");
}

#[test]
fn sosfiltfilt_matches_scipy() {
    let g = golden();
    let c = &g["sosfiltfilt"];
    let y = sosfiltfilt(&sos_rows(&c["sos"]), &floats(&c["x"])).expect("long enough");
    assert_close(&y, &floats(&c["y"]), 1e-12, "sosfiltfilt");
}

#[test]
fn decimate_matches_scipy() {
    let g = golden();
    for case in g["decimate"].as_array().expect("decimate array") {
        let q = case["q"].as_u64().expect("q") as usize;
        let y = decimate(&floats(&case["x"]), q).expect("long enough");
        assert_close(&y, &floats(&case["y"]), 1e-11, &format!("decimate q={q}"));
    }
}

#[test]
fn hilbert_envelope_matches_scipy() {
    let g = golden();
    for (i, case) in g["hilbert"]
        .as_array()
        .expect("hilbert array")
        .iter()
        .enumerate()
    {
        let env = hilbert_envelope(&floats(&case["x"]));
        assert_close(
            &env,
            &floats(&case["env"]),
            1e-12,
            &format!("hilbert case {i}"),
        );
    }
}

#[test]
fn resample_matches_scipy() {
    let g = golden();
    for case in g["resample"].as_array().expect("resample array") {
        let x = floats(&case["x"]);
        let num = case["num"].as_u64().expect("num") as usize;
        let y = resample(&x, num);
        assert_close(
            &y,
            &floats(&case["y"]),
            1e-11,
            &format!("resample {} -> {num}", x.len()),
        );
    }
}

#[test]
fn sosfreqz_matches_scipy() {
    let g = golden();
    let c = &g["sosfreqz"];
    let h = sosfreqz(&sos_rows(&c["sos"]), c["n"].as_u64().expect("n") as usize);
    let (re, im): (Vec<f64>, Vec<f64>) = h.iter().map(|v| (v.re, v.im)).unzip();
    assert_close(&re, &floats(&c["h_re"]), 1e-11, "sosfreqz real");
    assert_close(&im, &floats(&c["h_im"]), 1e-11, "sosfreqz imag");
}

#[test]
fn find_peaks_matches_scipy_indices_and_prominences() {
    let g = golden();
    for (i, case) in g["find_peaks"]
        .as_array()
        .expect("find_peaks array")
        .iter()
        .enumerate()
    {
        let peaks = find_peaks_with_prominence(&floats(&case["x"]));
        let got_idx: Vec<f64> = peaks.iter().map(|p| p.index as f64).collect();
        let want_idx = floats(&case["indices"]);
        assert_eq!(
            got_idx.len(),
            want_idx.len(),
            "find_peaks case {i}: peak count (got {got_idx:?}, want {want_idx:?})"
        );
        for (gi, wi) in got_idx.iter().zip(&want_idx) {
            assert_eq!(gi, wi, "find_peaks case {i}: peak index");
        }
        let got_prom: Vec<f64> = peaks.iter().map(|p| p.prominence).collect();
        assert_close(
            &got_prom,
            &floats(&case["prominences"]),
            1e-12,
            "prominences",
        );
    }
}

#[test]
fn pchip_matches_scipy() {
    let g = golden();
    for (i, case) in g["pchip"]
        .as_array()
        .expect("pchip array")
        .iter()
        .enumerate()
    {
        let yq = pchip(
            &floats(&case["x"]),
            &floats(&case["y"]),
            &floats(&case["xq"]),
        );
        assert_close(&yq, &floats(&case["yq"]), 1e-11, &format!("pchip case {i}"));
    }
}

#[test]
fn interp_matches_numpy() {
    let g = golden();
    let c = &g["interp"];
    let yq = interp(&floats(&c["xq"]), &floats(&c["xp"]), &floats(&c["fp"]));
    assert_close(&yq, &floats(&c["yq"]), 1e-13, "interp");
}

#[test]
fn percentile_and_median_match_numpy() {
    let g = golden();
    let c = &g["percentile"];
    let x = floats(&c["x"]);
    let got: Vec<f64> = floats(&c["q"])
        .iter()
        .map(|&q| percentile_linear(&x, q))
        .collect();
    assert_close(&got, &floats(&c["values"]), 1e-13, "percentile");
    assert_close(
        &[median(&x)],
        &[c["median"].as_f64().expect("median")],
        1e-13,
        "median",
    );
}

#[test]
fn golden_file_records_the_versions_it_came_from() {
    // A bare assertion that the provenance survived regeneration: a golden
    // mismatch should be attributable to a specific SciPy version.
    let g = golden();
    assert!(g["versions"]["scipy"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
    assert!(g["versions"]["numpy"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
}
