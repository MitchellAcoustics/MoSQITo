//! Conformance of `slm::noct` against MoSQITo's own reference corpus.
//!
//! `tools/gen_golden_noct.py` exports two kinds of reference: `center_freq`
//! and `filter_bandwidth` are checked against a formula transcription that the
//! script itself verifies against the installed `mosqito` package; the
//! `end_to_end` section is the strong gate — real `noct_spectrum` /
//! `noct_synthesis` output from MoSQITo's own reference corpus wav
//! (`tests/input/Test signal 5 (pinknoise 60 dB).wav`, the same file MoSQITo's
//! own `test_noct_synthesis_technical` uses). Only the small per-band outputs
//! are embedded in JSON; this test reads the wav itself and reproduces
//! MoSQITo's exact calibration and FFT construction using this crate's own
//! (already golden-tested) primitives, so it exercises the real pipeline
//! rather than comparing against a duplicated copy of the signal.
//!
//! Regenerate with `../.venv/bin/python tools/gen_golden_noct.py` from `rust/`.

use mosqito_core::slm::noct::{center_freq, filter_bandwidth, noct_spectrum, noct_synthesis};
use ndarray::Array2;
use serde_json::Value;
use std::path::PathBuf;

fn golden() -> Value {
    let raw = include_str!("golden_noct.json");
    serde_json::from_str(raw).expect("golden_noct.json parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected a JSON array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

#[track_caller]
fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        let scale = w.abs().max(1e-300);
        assert!(
            err <= tol * scale,
            "{what}: index {i} differs by {err:e} (got {g:e}, want {w:e})"
        );
    }
}

/// Walks up from this crate's directory to find the MoSQITo repository root,
/// identified by the `tests/input/` reference corpus. `MANIFEST.in` does not
/// ship this data in the package, so it only ever exists in a full checkout.
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

/// Reads a mono 16-bit PCM wav and applies MoSQITo's exact calibration:
/// `wav_calib * sample / (2^15 - 1)` — note 32767, not 32768
/// (`mosqito/utils/load.py:56`).
fn load_wav_calibrated(path: &std::path::Path, wav_calib: f64) -> (Vec<f64>, f64) {
    let mut reader = hound::WavReader::open(path).expect("open reference wav");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "reference wav is expected to be mono");
    assert_eq!(
        spec.bits_per_sample, 16,
        "reference wav is expected to be 16-bit PCM"
    );
    let sig: Vec<f64> = reader
        .samples::<i16>()
        .map(|s| wav_calib * s.expect("sample") as f64 / (2f64.powi(15) - 1.0))
        .collect();
    (sig, spec.sample_rate as f64)
}

/// Builds the one-sided FFT-magnitude spectrum the way MoSQITo's own
/// conformance test does: `2/sqrt(2)/n * fft(sig)[0:n/2]`.
fn one_sided_magnitude_spectrum(sig: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    let n = sig.len();
    let mut planner = realfft::RealFftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(n);
    let mut spectrum = fwd.make_output_vec();
    let mut input = sig.to_vec();
    fwd.process(&mut input, &mut spectrum).expect("real FFT");

    let scale = 2.0 / std::f64::consts::SQRT_2 / n as f64;
    let magnitude: Vec<f64> = spectrum[..n / 2].iter().map(|c| c.norm() * scale).collect();
    let freqs: Vec<f64> = (0..n / 2).map(|k| k as f64 * fs / n as f64).collect();
    (magnitude, freqs)
}

#[test]
fn center_freq_matches_mosqitos_transcribed_formula() {
    let g = golden();
    for case in g["center_freq"].as_array().expect("center_freq array") {
        let (fe, fn_) = center_freq(
            case["fmin"].as_f64().unwrap(),
            case["fmax"].as_f64().unwrap(),
            case["n"].as_u64().unwrap() as u32,
            case["g"].as_u64().unwrap() as u32,
            case["fr"].as_f64().unwrap(),
        );
        assert_close(&fe, &floats(&case["f_exact"]), 1e-9, "center_freq f_exact");
        assert_close(&fn_, &floats(&case["f_nom"]), 1e-9, "center_freq f_nom");
    }
}

#[test]
fn filter_bandwidth_matches_mosqitos_transcribed_formula() {
    let g = golden();
    for case in g["filter_bandwidth"]
        .as_array()
        .expect("filter_bandwidth array")
    {
        let fc = floats(&case["fc"]);
        let (alpha, f1, f2) = filter_bandwidth(&fc, case["n"].as_u64().unwrap() as u32);
        assert_close(
            &alpha,
            &floats(&case["alpha"]),
            1e-9,
            "filter_bandwidth alpha",
        );
        assert_close(&f1, &floats(&case["f1"]), 1e-9, "filter_bandwidth f1");
        assert_close(&f2, &floats(&case["f2"]), 1e-9, "filter_bandwidth f2");
    }
}

#[test]
fn noct_spectrum_matches_mosqito_on_the_reference_pink_noise_wav() {
    let g = golden();
    let e2e = &g["end_to_end"];
    let wav_path = repo_root().join(e2e["wav_path"].as_str().expect("wav_path"));
    let wav_calib = e2e["wav_calib"].as_f64().expect("wav_calib");

    let (sig, fs) = load_wav_calibrated(&wav_path, wav_calib);
    assert_eq!(sig.len(), e2e["n"].as_u64().expect("n") as usize);
    assert_eq!(fs, e2e["fs"].as_f64().expect("fs"));

    let sig2d = Array2::from_shape_vec((sig.len(), 1), sig).expect("column signal");

    for order in [1u32, 3] {
        let key = format!("noct_spectrum_n{order}");
        let case = &e2e[&key];
        let (spec, freq) = noct_spectrum(sig2d.view(), fs, 24.0, 12600.0, order, 10, 1000.0)
            .expect("valid design");
        assert_close(&freq, &floats(&case["freq"]), 1e-9, &format!("{key} freq"));
        // The full pipeline (decimate + Butterworth design + sosfilt, chained
        // across up to 28 bands) matches to a tight relative tolerance: every
        // stage was already golden-tested against SciPy in isolation, so this
        // is confirming the composition, not re-litigating the primitives.
        let got: Vec<f64> = spec.column(0).to_vec();
        assert_close(&got, &floats(&case["spec"]), 1e-6, &format!("{key} spec"));
    }
}

#[test]
fn noct_synthesis_matches_mosqito_on_the_reference_pink_noise_wav() {
    let g = golden();
    let e2e = &g["end_to_end"];
    let wav_path = repo_root().join(e2e["wav_path"].as_str().expect("wav_path"));
    let wav_calib = e2e["wav_calib"].as_f64().expect("wav_calib");

    let (sig, fs) = load_wav_calibrated(&wav_path, wav_calib);
    let (magnitude, freqs) = one_sided_magnitude_spectrum(&sig, fs);

    for order in [1u32, 3] {
        let key = format!("noct_synthesis_n{order}");
        let case = &e2e[&key];
        let (spec, freq) = noct_synthesis(&magnitude, &freqs, 24.0, 12600.0, order, 10, 1000.0)
            .expect("valid design");
        assert_close(&freq, &floats(&case["freq"]), 1e-9, &format!("{key} freq"));
        assert_close(&spec, &floats(&case["spec"]), 1e-6, &format!("{key} spec"));
    }
}
