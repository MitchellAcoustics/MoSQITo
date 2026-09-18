//! ISO 532-1:2017 conformance gate for stationary loudness, against MoSQITo's
//! own reference corpus (which itself traces to the standard's Annex B).
//!
//! Tolerance is the wider of ±5% or ±0.1 sone, matching `isoclose` and
//! `test_loudness_zwst.py` — this is the actual ISO 532-1 §5.1 compliance
//! criterion MoSQITo's own tests use, not an arbitrarily chosen bound.

use mosqito_core::loudness::zwst::{loudness_zwst, loudness_zwst_freq, FieldType};
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

/// ISO 532-1 §5.1's compliance criterion: within the wider of ±5% or
/// ±0.1 sone.
#[track_caller]
fn assert_iso_close(actual: f64, desired: f64, what: &str) {
    let band = (0.05 * desired.abs()).max(0.1);
    assert!(
        (actual - desired).abs() <= band,
        "{what}: got {actual}, want {desired} +/- {band} (ISO 532-1 5% / 0.1 sone tolerance)"
    );
}

#[track_caller]
fn assert_iso_close_array(actual: &[f64], desired: &[f64], what: &str) {
    assert_eq!(actual.len(), desired.len(), "{what}: length mismatch");
    for (i, (&a, &d)) in actual.iter().zip(desired).enumerate() {
        let band = (0.05 * d.abs()).max(0.1);
        assert!(
            (a - d).abs() <= band,
            "{what}[{i}]: got {a}, want {d} +/- {band} (ISO 532-1 5% / 0.1 sone tolerance)"
        );
    }
}

/// Reads a MoSQITo reference CSV: UTF-8 BOM, one `#`-comment header line,
/// then one bare float per line — the 240-point specific-loudness format
/// used throughout `tests/input/` and `validations/`.
fn read_reference_csv(path: &Path) -> Vec<f64> {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
    let raw = raw.strip_prefix('\u{feff}').unwrap_or(&raw);
    raw.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect()
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

#[test]
fn loudness_zwst_matches_iso_532_1_annex_b2_from_a_third_octave_spectrum() {
    // ISO 532-1 Annex B2 reference third-octave spectrum, N = 83.296 sone.
    // The spectrum itself is `tests/input/Test_signal_1.py`'s `test_signal_1`
    // array; transcribed here rather than executing the Python (this test
    // only needs the 28 numbers, not a Python interpreter).
    let spec: [f64; 28] = [
        -60.0, -60.0, 78.0, 79.0, 89.0, 72.0, 80.0, 89.0, 75.0, 87.0, 85.0, 79.0, 86.0, 80.0, 71.0,
        70.0, 72.0, 71.0, 72.0, 74.0, 69.0, 65.0, 67.0, 77.0, 68.0, 58.0, 45.0, 30.0,
    ];

    let nm = mosqito_core::loudness::zwst::main_loudness(&spec, FieldType::Free);
    let (n, n_specific) = mosqito_core::loudness::zwst::calc_slopes(&nm);

    assert_iso_close(n, 83.296, "N (Annex B2)");

    let want_spec = read_reference_csv(&repo_root().join("tests/input/test_signal_1.csv"));
    assert_iso_close_array(&n_specific, &want_spec, "N_specific (Annex B2)");
}

#[test]
fn loudness_zwst_matches_iso_532_1_annex_b3_pink_noise_wav() {
    // ISO 532-1 Annex B3 reference recording, N = 10.498 sone.
    let root = repo_root();
    let (sig, fs) = read_wav_calibrated(
        &root.join("tests/input/Test signal 5 (pinknoise 60 dB).wav"),
        2.0 * 2f64.sqrt(),
    );

    let (n, n_specific, _bark) = loudness_zwst(&sig, fs, FieldType::Free);
    assert_iso_close(n, 10.498, "N (Annex B3, time domain)");

    let want_spec = read_reference_csv(&root.join("tests/input/test_signal_5.csv"));
    assert_iso_close_array(
        &n_specific,
        &want_spec,
        "N_specific (Annex B3, time domain)",
    );
}

#[test]
fn loudness_zwst_freq_matches_iso_532_1_annex_b3_pink_noise_wav() {
    // Same Annex B3 signal and target (N = 10.498 sone), through the
    // frequency-domain entry point: FFT magnitude spectrum in, matching
    // MoSQITo's own test_loudness_zwst_freq convention exactly
    // (2/sqrt(2)/n * fft(sig)[:n/2]).
    let root = repo_root();
    let (sig, fs) = read_wav_calibrated(
        &root.join("tests/input/Test signal 5 (pinknoise 60 dB).wav"),
        2.0 * 2f64.sqrt(),
    );

    let n = sig.len();
    let mut planner = realfft::RealFftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(n);
    let mut spectrum_c = fwd.make_output_vec();
    let mut input = sig.clone();
    fwd.process(&mut input, &mut spectrum_c).unwrap();
    let scale = 2.0 / std::f64::consts::SQRT_2 / n as f64;
    let magnitude: Vec<f64> = spectrum_c[..n / 2]
        .iter()
        .map(|c| c.norm() * scale)
        .collect();
    let freqs: Vec<f64> = (0..n / 2).map(|k| k as f64 * fs / n as f64).collect();

    let (n_loud, n_specific, _bark) = loudness_zwst_freq(&magnitude, &freqs, FieldType::Free);
    assert_iso_close(n_loud, 10.498, "N (Annex B3, frequency domain)");

    let want_spec = read_reference_csv(&root.join("tests/input/test_signal_5.csv"));
    assert_iso_close_array(
        &n_specific,
        &want_spec,
        "N_specific (Annex B3, frequency domain)",
    );
}

#[test]
fn loudness_zwst_matches_iso_532_1_annex_b3_44100hz_signal() {
    // ISO 532-1 Annex B3 signal 3, resampled from 44.1 kHz, N = 4.019 sone.
    // This is the fidelity-sensitive case the plan flagged: reachable only
    // through a Fourier resampler that matches scipy.signal.resample.
    let root = repo_root();
    let (sig, fs) = read_wav_calibrated(
        &root.join("tests/input/Test signal 3 (1 kHz 60 dB)_44100Hz.wav"),
        2.0 * 2f64.sqrt(),
    );
    assert_eq!(fs, 44100.0, "sanity check: this reference wav is 44.1 kHz");

    let (n, _n_specific, _bark) = loudness_zwst(&sig, fs, FieldType::Free);
    assert_iso_close(n, 4.019, "N (Annex B3, resampled from 44.1 kHz)");
}
