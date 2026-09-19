//! DIN 45692:2009 chapter 6 conformance gate for `sharpness_din_st`, against
//! all 41 reference signals the standard publishes (20 broadband + 21
//! narrowband noise, `validations/sq_metrics/sharpness_din/input/`).
//!
//! Tolerance is the wider of ±5% or ±0.05 acum, matching
//! `validations/sq_metrics/sharpness_din/validation_sharpness_din.py`'s
//! `_check_compliance` — DIN 45692's own chapter 6 compliance criterion, not
//! an arbitrarily chosen bound. (The narrower ±5%-only tolerance in
//! `tests/sq_metrics/sharpness/test_sharpness_din.py`'s unit test is not
//! used here — see the plan's port-order notes on why the wider band is the
//! one the standard actually specifies.)

use mosqito_core::loudness::zwst::FieldType;
use mosqito_core::sharpness::din::{sharpness_din_st, Weighting};
use std::path::{Path, PathBuf};

/// Walks up from this crate's directory to the MoSQITo repository root,
/// identified by the `tests/input/` reference corpus (which `MANIFEST.in`
/// does not ship, so it only ever exists in a full checkout).
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

/// Reads a mono 16-bit PCM wav with MoSQITo's exact calibration:
/// `wav_calib * sample / (2^15 - 1)` (`mosqito/utils/load.py:56`).
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

/// DIN 45692 chapter 6's compliance criterion: the wider of ±5% or ±0.05 acum.
#[track_caller]
fn assert_din_close(actual: f64, desired: f64, what: &str) {
    let band = (0.05 * desired.abs()).max(0.05);
    assert!(
        (actual - desired).abs() <= band,
        "{what}: got {actual}, want {desired} +/- {band} (DIN 45692 ch.6 5% / 0.05 acum tolerance)"
    );
}

/// `(file name, reference S [acum])`, transcribed from
/// `validations/sq_metrics/sharpness_din/validation_sharpness_din.py`, which
/// in turn transcribes DIN 45692:2009's chapter 6 tables.
const BROADBAND: &[(&str, f64)] = &[
    ("broadband_250.wav", 2.70),
    ("broadband_350.wav", 2.74),
    ("broadband_450.wav", 2.78),
    ("broadband_570.wav", 2.85),
    ("broadband_700.wav", 2.91),
    ("broadband_840.wav", 2.96),
    ("broadband_1000.wav", 3.05),
    ("broadband_1170.wav", 3.12),
    ("broadband_1370.wav", 3.20),
    ("broadband_1600.wav", 3.30),
    ("broadband_1850.wav", 3.42),
    ("broadband_2150.wav", 3.53),
    ("broadband_2500.wav", 3.69),
    ("broadband_2900.wav", 3.89),
    ("broadband_3400.wav", 4.12),
    ("broadband_4000.wav", 4.49),
    ("broadband_4800.wav", 5.04),
    ("broadband_5800.wav", 5.69),
    ("broadband_7000.wav", 6.47),
    ("broadband_8500.wav", 7.46),
];

// Note the uppercase extension on `narrowband_250.WAV` — a case-sensitivity
// hazard on the real reference corpus, not a typo here.
const NARROWBAND: &[(&str, f64)] = &[
    ("narrowband_250.WAV", 0.38),
    ("narrowband_350.wav", 0.49),
    ("narrowband_450.wav", 0.60),
    ("narrowband_570.wav", 0.71),
    ("narrowband_700.wav", 0.82),
    ("narrowband_840.wav", 0.93),
    ("narrowband_1000.wav", 1.00),
    ("narrowband_1170.wav", 1.13),
    ("narrowband_1370.wav", 1.26),
    ("narrowband_1600.wav", 1.35),
    ("narrowband_1850.wav", 1.49),
    ("narrowband_2150.wav", 1.64),
    ("narrowband_2500.wav", 1.78),
    ("narrowband_2900.wav", 2.06),
    ("narrowband_3400.wav", 2.40),
    ("narrowband_4000.wav", 2.82),
    ("narrowband_4800.wav", 3.48),
    ("narrowband_5800.wav", 4.43),
    ("narrowband_7000.wav", 5.52),
    ("narrowband_8500.wav", 6.81),
    ("narrowband_10500.wav", 8.55),
];

fn run_corpus(name: &str, corpus: &[(&str, f64)]) {
    let root = repo_root();
    let dir = root.join("validations/sq_metrics/sharpness_din/input");
    assert_eq!(corpus.len(), if name == "broadband" { 20 } else { 21 });

    let mut failures = Vec::new();
    for &(file, want) in corpus {
        let (sig, fs) = read_wav_calibrated(&dir.join(file), 1.0);
        let s = sharpness_din_st(&sig, fs, Weighting::Din, FieldType::Free);
        let band = (0.05 * want).max(0.05);
        if (s - want).abs() > band {
            failures.push(format!("{file}: got {s}, want {want} +/- {band}"));
        }
    }

    assert!(
        failures.is_empty(),
        "{}/{} {name} signals failed DIN 45692 ch.6 compliance:\n{}",
        failures.len(),
        corpus.len(),
        failures.join("\n")
    );
}

#[test]
fn sharpness_din_st_matches_din_45692_broadband_noise_reference_values() {
    run_corpus("broadband", BROADBAND);
}

#[test]
fn sharpness_din_st_matches_din_45692_narrowband_noise_reference_values() {
    run_corpus("narrowband", NARROWBAND);
}

#[test]
fn sharpness_din_st_matches_a_worked_example_exactly() {
    // The one signal also embedded in `tests/input/` (used by MoSQITo's own
    // unit test), cross-checked precisely rather than just within tolerance.
    let (sig, fs) = read_wav_calibrated(&repo_root().join("tests/input/broadband_570.wav"), 1.0);
    let s = sharpness_din_st(&sig, fs, Weighting::Din, FieldType::Free);
    assert_din_close(s, 2.85, "S (broadband_570, worked example)");
}
