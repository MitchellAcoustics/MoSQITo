//! ISO 532-1:2017 section 6.1 conformance gate for time-varying loudness,
//! against all 20 Annex B.4 (synthetic) + B.5 (technical) reference signals.
//!
//! The reference values (`reference_loudness_zwtv_annex_b.json`) are the
//! standard's own published numbers — column B (N) and column L (specific
//! loudness at one reference Bark value) of each signal's xlsx sheet under
//! `validations/sq_metrics/loudness_zwtv/input/`, extracted once by
//! `tools/gen_reference_loudness_zwtv.py` — not MoSQITo's computed output.
//!
//! The compliance procedure (length trim, ±1-sample/±2 ms realignment,
//! ≤1% of samples allowed outside the wider of ±5%/±0.1 sone) matches
//! `validations/sq_metrics/loudness_zwtv/validation_loudness_zwtv.py`'s
//! `_check_compliance`, which is ISO 532-1 section 6.1's own compliance
//! procedure, not a bound chosen for this port.

use mosqito_core::loudness::zwst::FieldType;
use mosqito_core::loudness::zwtv::loudness_zwtv;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("tests/input").is_dir() {
            return dir;
        }
        if !dir.pop() {
            panic!(
                "could not find the MoSQITo repository root (looked for tests/input/) \
                 starting from {}",
                env!("CARGO_MANIFEST_DIR")
            );
        }
    }
}

fn read_wav_calibrated(path: &Path, wav_calib: f64) -> (Vec<f64>, f64) {
    let mut reader =
        hound::WavReader::open(path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
    let spec = reader.spec();
    let sig: Vec<f64> = reader
        .samples::<i16>()
        .map(|s| wav_calib * s.expect("sample") as f64 / (2f64.powi(15) - 1.0))
        .collect();
    (sig, spec.sample_rate as f64)
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

/// ISO 532-1 section 6.1's per-sample compliance band: within the wider of
/// ±5% or ±0.1 sone.
fn in_tolerance(actual: f64, reference: f64) -> bool {
    let lo = (reference * 0.95).min(reference - 0.1);
    let hi = (reference * 1.05).max(reference + 0.1);
    actual >= lo && actual <= hi
}

/// Trims to matching length (ISO 532-1's own reference and mosqito-rs's
/// output can differ by at most 1 sample), then tries a ±1-sample (±2 ms)
/// realignment if it reduces total absolute error, exactly mirroring
/// `validation_loudness_zwtv.py:287-317`.
fn align(actual: &[f64], reference: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let len_diff = (actual.len() as i64 - reference.len() as i64).abs();
    assert!(
        len_diff <= 1,
        "N length differs by {len_diff} (ISO 532-1's own tolerance is at most 1 sample)"
    );
    let n = actual.len().min(reference.len());
    let mut a: Vec<f64> = actual[..n].to_vec();
    let mut r: Vec<f64> = reference[..n].to_vec();

    let err = |a: &[f64], r: &[f64]| -> f64 { a.iter().zip(r).map(|(x, y)| (x - y).abs()).sum() };
    let base_err = err(&a, &r);

    if n > 1 {
        let shifted_left_err = err(&a[1..], &r[..n - 1]);
        let shifted_right_err = err(&a[..n - 1], &r[1..]);
        if shifted_left_err < base_err && shifted_left_err <= shifted_right_err {
            a = a[1..].to_vec();
            r = r[..n - 1].to_vec();
        } else if shifted_right_err < base_err {
            a = a[..n - 1].to_vec();
            r = r[1..].to_vec();
        }
    }

    (a, r)
}

fn assert_iso_6_1_compliant(actual: &[f64], reference: &[f64], what: &str) {
    let (a, r) = align(actual, reference);
    let n_outside = a
        .iter()
        .zip(&r)
        .filter(|(&x, &y)| !in_tolerance(x, y))
        .count();
    let frac_outside = n_outside as f64 / a.len() as f64;
    assert!(
        frac_outside <= 0.01,
        "{what}: {n_outside}/{} samples ({:.2}%) outside the ISO 532-1 5%/0.1 sone \
         tolerance band, want <=1%",
        a.len(),
        frac_outside * 100.0
    );
}

#[test]
fn loudness_zwtv_matches_iso_532_1_annex_b4_and_b5_reference_signals() {
    let root = repo_root();
    let reference_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/reference_loudness_zwtv_annex_b.json");
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(&reference_path)
            .unwrap_or_else(|e| panic!("reading {reference_path:?}: {e}")),
    )
    .expect("reference JSON parses");

    let signals = reference.as_array().expect("signal array");
    assert_eq!(signals.len(), 20, "expected all 20 Annex B.4+B.5 signals");

    let mut failures = Vec::new();
    for sig in signals {
        let tab = sig["tab"].as_str().expect("tab");
        let data_file = sig["data_file"].as_str().expect("data_file");
        let field = match sig["field"].as_str().expect("field") {
            "free" => FieldType::Free,
            "diffuse" => FieldType::Diffuse,
            other => panic!("unknown field_type {other}"),
        };
        let n_iso = floats(&sig["N_iso"]);

        let wav_path = root
            .join("validations/sq_metrics/loudness_zwtv/input")
            .join(data_file);
        let (signal, fs) = read_wav_calibrated(&wav_path, 2.0 * 2f64.sqrt());

        let result = std::panic::catch_unwind(|| {
            let (n, _n_specific, _bark, _time) =
                loudness_zwtv(&signal, fs, field).expect("valid pipeline");
            n
        });

        match result {
            Ok(n) => {
                let outcome = std::panic::catch_unwind(|| {
                    assert_iso_6_1_compliant(&n, &n_iso, tab);
                });
                if let Err(e) = outcome {
                    let msg = e
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "assertion failed".to_string());
                    failures.push(format!("{tab}: {msg}"));
                }
            }
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panicked".to_string());
                failures.push(format!("{tab}: computation panicked: {msg}"));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{}/20 signals failed ISO 532-1 section 6.1 compliance:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
