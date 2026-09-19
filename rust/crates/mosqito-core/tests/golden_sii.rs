//! Conformance of SII (ANSI S3.5) against real MoSQITo.
//!
//! `tools/gen_golden_sii.py` captures MoSQITo's `_main_sii`/`sii_ansi*`
//! output into `golden_sii.json`; these tests assert the Rust port
//! reproduces it — including the critical/equally-critical band procedures,
//! which no digitised standards corpus in this repository exercises (see
//! `tests/conformance_sii.rs` for the one ANSI-anchored worked example that
//! does, using the octave procedure).
//!
//! Regenerate with `.venv/bin/python tools/gen_golden_sii.py`.

use mosqito_core::speech_intelligibility::{
    main_sii, sii_ansi, sii_ansi_freq, sii_ansi_level, SiiMethod, SiiThreshold, SpeechLevel,
};
use serde_json::Value;

fn golden() -> Value {
    let raw = include_str!("golden_sii.json");
    serde_json::from_str(raw).expect("golden_sii.json parses")
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

fn parse_method(s: &str) -> SiiMethod {
    match s {
        "critical" => SiiMethod::Critical,
        "equally_critical" => SiiMethod::EquallyCritical,
        "third_octave" => SiiMethod::ThirdOctave,
        "octave" => SiiMethod::Octave,
        other => panic!("unknown method {other}"),
    }
}

fn parse_speech_level(s: &str) -> SpeechLevel {
    match s {
        "normal" => SpeechLevel::Normal,
        "raised" => SpeechLevel::Raised,
        "loud" => SpeechLevel::Loud,
        "shout" => SpeechLevel::Shout,
        other => panic!("unknown speech level {other}"),
    }
}

#[test]
fn main_sii_matches_mosqito_for_every_band_procedure() {
    let g = golden();
    let cases = g["main_sii"].as_object().unwrap();
    for (method_str, c) in cases {
        let method = parse_method(method_str);
        let speech = floats(&c["speech"]);
        let noise = floats(&c["noise"]);
        let want_sii = c["sii"].as_f64().unwrap();
        let want_spec = floats(&c["sii_spec"]);
        let want_freq = floats(&c["freq_axis"]);

        let (got_sii, got_spec, got_freq) = main_sii(method, &speech, &noise, SiiThreshold::Zero);
        assert!(
            (got_sii - want_sii).abs() < 1e-9,
            "{method_str}: SII differs (got {got_sii}, want {want_sii})"
        );
        assert_close(
            &got_spec,
            &want_spec,
            1e-9,
            &format!("{method_str} sii_spec"),
        );
        assert_close(
            &got_freq,
            &want_freq,
            1e-9,
            &format!("{method_str} freq_axis"),
        );

        let want_sii_z = c["sii_zwicker"].as_f64().unwrap();
        let want_spec_z = floats(&c["sii_spec_zwicker"]);
        let (got_sii_z, got_spec_z, _) = main_sii(method, &speech, &noise, SiiThreshold::Zwicker);
        assert!(
            (got_sii_z - want_sii_z).abs() < 1e-9,
            "{method_str} zwicker: SII differs (got {got_sii_z}, want {want_sii_z})"
        );
        assert_close(
            &got_spec_z,
            &want_spec_z,
            1e-9,
            &format!("{method_str} sii_spec zwicker"),
        );
    }
}

#[test]
fn sii_ansi_level_matches_mosqito() {
    let g = golden();
    let c = &g["sii_ansi_level"];
    let noise_level = c["noise_level"].as_f64().unwrap();
    let method = parse_method(c["method"].as_str().unwrap());
    let speech_level = parse_speech_level(c["speech_level"].as_str().unwrap());
    let want_sii = c["sii"].as_f64().unwrap();
    let want_spec = floats(&c["sii_spec"]);

    let (got_sii, got_spec, _) =
        sii_ansi_level(noise_level, method, speech_level, SiiThreshold::Zero);
    assert!((got_sii - want_sii).abs() < 1e-9);
    assert_close(&got_spec, &want_spec, 1e-9, "sii_ansi_level sii_spec");
}

#[test]
fn sii_ansi_freq_matches_mosqito() {
    let g = golden();
    let c = &g["sii_ansi_freq"];
    let spectrum = floats(&c["spectrum_db"]);
    let freqs = floats(&c["freqs"]);
    let method = parse_method(c["method"].as_str().unwrap());
    let speech_level = parse_speech_level(c["speech_level"].as_str().unwrap());
    let want_sii = c["sii"].as_f64().unwrap();
    let want_spec = floats(&c["sii_spec"]);

    let (got_sii, got_spec, _) =
        sii_ansi_freq(&spectrum, &freqs, method, speech_level, SiiThreshold::Zero);
    assert!(
        (got_sii - want_sii).abs() < 1e-6 * want_sii.abs().max(1.0),
        "got {got_sii}, want {want_sii}"
    );
    assert_close(&got_spec, &want_spec, 1e-6, "sii_ansi_freq sii_spec");
}

#[test]
fn sii_ansi_matches_mosqito() {
    let g = golden();
    let c = &g["sii_ansi"];
    let noise = floats(&c["noise"]);
    let fs = c["fs"].as_f64().unwrap();
    let method = parse_method(c["method"].as_str().unwrap());
    let speech_level = parse_speech_level(c["speech_level"].as_str().unwrap());
    let want_sii = c["sii"].as_f64().unwrap();
    let want_spec = floats(&c["sii_spec"]);

    let (got_sii, got_spec, _) = sii_ansi(&noise, fs, method, speech_level, SiiThreshold::Zero);
    assert!(
        (got_sii - want_sii).abs() < 1e-6 * want_sii.abs().max(1.0),
        "got {got_sii}, want {want_sii}"
    );
    assert_close(&got_spec, &want_spec, 1e-6, "sii_ansi sii_spec");
}
