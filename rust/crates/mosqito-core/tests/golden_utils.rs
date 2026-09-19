//! Conformance of Phase 2's foundation utilities against real MoSQITo.
//!
//! `tools/gen_golden_utils.py` captures MoSQITo's output for each function
//! into `golden_utils.json`; these tests assert the Rust ports reproduce it.
//!
//! Regenerate with `.venv/bin/python tools/gen_golden_utils.py`.

use mosqito_core::generators::{am_sine_generator, fm_sine_generator, sine_wave_generator};
use mosqito_core::loudness::{equal_loudness_contours, sone_to_phon};
use mosqito_core::slm::{
    comp_spectrum_complex, comp_spectrum_db, freq_band_synthesis, SpectrumWindow,
};
use mosqito_core::utils::{bark2freq, db2amp, freq2bark, ltq, spectrum2dba, LtqReference};
use ndarray::Array2;
use serde_json::Value;

fn golden() -> Value {
    let raw = include_str!("golden_utils.json");
    serde_json::from_str(raw).expect("golden_utils.json parses")
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
fn bark2freq_matches_mosqito() {
    let g = golden();
    let c = &g["bark2freq"];
    let bark = floats(&c["bark"]);
    let want = floats(&c["freq"]);
    let got = bark2freq(&bark);
    assert_close(&got, &want, 1e-9, "bark2freq");
}

#[test]
fn freq2bark_matches_mosqito() {
    let g = golden();
    let c = &g["freq2bark"];
    let freq = floats(&c["freq"]);
    let want = floats(&c["bark"]);
    let got = freq2bark(&freq);
    assert_close(&got, &want, 1e-9, "freq2bark");
}

#[test]
fn db2amp_matches_mosqito() {
    let g = golden();
    let c = &g["db2amp"];
    let db = floats(&c["db"]);
    let want1 = floats(&c["ref1"]);
    let want2 = floats(&c["ref2e-5"]);
    let got1: Vec<f64> = db.iter().map(|&d| db2amp(d, 1.0)).collect();
    let got2: Vec<f64> = db.iter().map(|&d| db2amp(d, 2e-5)).collect();
    assert_close(&got1, &want1, 1e-9, "db2amp ref=1");
    assert_close(&got2, &want2, 1e-9, "db2amp ref=2e-5");
}

#[test]
fn spectrum2dba_matches_mosqito() {
    let g = golden();
    let c = &g["spectrum2dBA"];
    let spectrum_db = floats(&c["spectrum_db"]);
    let fs = c["fs"].as_f64().unwrap();
    let want = floats(&c["dba"]);
    let got = spectrum2dba(&spectrum_db, fs);
    assert_close(&got, &want, 1e-9, "spectrum2dba");
}

#[test]
fn ltq_matches_mosqito() {
    let g = golden();
    let c = &g["ltq"];
    let bark = floats(&c["bark"]);
    let want_zwicker = floats(&c["zwicker"]);
    let want_roughness = floats(&c["roughness"]);
    let got_zwicker = ltq(&bark, LtqReference::Zwicker);
    let got_roughness = ltq(&bark, LtqReference::Roughness);
    assert_close(&got_zwicker, &want_zwicker, 1e-9, "ltq zwicker");
    assert_close(&got_roughness, &want_roughness, 1e-9, "ltq roughness");
}

#[test]
fn comp_spectrum_matches_mosqito_1d() {
    let g = golden();
    let c = &g["comp_spectrum_1d"];
    let signal = floats(&c["signal"]);
    let fs = c["fs"].as_f64().unwrap();
    let n = signal.len();
    let sig2d = Array2::from_shape_vec((n, 1), signal).unwrap();

    let want_hanning_db = floats(&c["hanning_db"]);
    let want_hanning_freq = floats(&c["hanning_freq"]);
    let (got_db, got_freq) = comp_spectrum_db(sig2d.view(), fs, SpectrumWindow::Hanning);
    // dB is a log of the FFT magnitude, so bins landing very near a spectral
    // null (this AM-tone test signal has many, between its sidebands)
    // amplify FFT-vs-rustfft floating-point noise far out of proportion to
    // its size in linear amplitude — a known characteristic of dB spectra,
    // not a port discrepancy; the non-dB (complex) comparison below is
    // exact to 1e-9 relative, which is where the actual computation is
    // validated.
    assert_close(
        &got_db.column(0).to_vec(),
        &want_hanning_db,
        1e-6,
        "comp_spectrum hanning db",
    );
    assert_close(
        &got_freq,
        &want_hanning_freq,
        1e-9,
        "comp_spectrum hanning freq",
    );

    let want_re = floats(&c["blackman_re"]);
    let want_im = floats(&c["blackman_im"]);
    let want_freq = floats(&c["blackman_freq"]);
    let (got_cplx, got_freq2) = comp_spectrum_complex(sig2d.view(), fs, SpectrumWindow::Blackman);
    let got_re: Vec<f64> = got_cplx.column(0).iter().map(|c| c.re).collect();
    let got_im: Vec<f64> = got_cplx.column(0).iter().map(|c| c.im).collect();
    assert_close(&got_re, &want_re, 1e-9, "comp_spectrum blackman re");
    assert_close(&got_im, &want_im, 1e-9, "comp_spectrum blackman im");
    assert_close(&got_freq2, &want_freq, 1e-9, "comp_spectrum blackman freq");
}

#[test]
fn comp_spectrum_matches_mosqito_2d() {
    let g = golden();
    let c = &g["comp_spectrum_2d"];
    let fs = c["fs"].as_f64().unwrap();
    let rows = c["signal"].as_array().unwrap();
    let nseg = rows[0].as_array().unwrap().len();
    let n = rows.len();
    let mut sig2d = Array2::<f64>::zeros((n, nseg));
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.as_array().unwrap().iter().enumerate() {
            sig2d[[i, j]] = v.as_f64().unwrap();
        }
    }

    let want_freq = floats(&c["hanning_freq"]);
    let (got_db, got_freq) = comp_spectrum_db(sig2d.view(), fs, SpectrumWindow::Hanning);
    assert_close(&got_freq, &want_freq, 1e-9, "comp_spectrum 2d freq");

    let want_rows = c["hanning_db"].as_array().unwrap();
    for (k, want_row) in want_rows.iter().enumerate() {
        let want = floats(want_row);
        let got = got_db.row(k).to_vec();
        assert_close(&got, &want, 1e-6, &format!("comp_spectrum 2d db row {k}"));
    }
}

#[test]
fn freq_band_synthesis_matches_mosqito() {
    let g = golden();
    let c = &g["freq_band_synthesis"];
    let spectrum_db = floats(&c["spectrum_db"]);
    let freqs = floats(&c["freqs"]);
    let fmin = floats(&c["fmin"]);
    let fmax = floats(&c["fmax"]);
    let want_levels = floats(&c["band_levels"]);
    let want_centers = floats(&c["band_centers"]);

    let (got_levels, got_centers) = freq_band_synthesis(&spectrum_db, &freqs, &fmin, &fmax);
    assert_close(
        &got_levels,
        &want_levels,
        1e-9,
        "freq_band_synthesis levels",
    );
    assert_close(
        &got_centers,
        &want_centers,
        1e-9,
        "freq_band_synthesis centers",
    );
}

#[test]
fn sine_wave_generator_matches_mosqito() {
    let g = golden();
    let c = &g["sine_wave_generator"];
    let fs = c["fs"].as_f64().unwrap();
    let d = c["d"].as_f64().unwrap();
    let freq = c["freq"].as_f64().unwrap();
    let spl_level = c["spl_level"].as_f64().unwrap();
    let want_signal = floats(&c["signal"]);
    let want_time = floats(&c["time"]);

    let (got_signal, got_time) = sine_wave_generator(fs, d, freq, spl_level);
    assert_close(
        &got_signal,
        &want_signal,
        1e-9,
        "sine_wave_generator signal",
    );
    assert_close(&got_time, &want_time, 1e-9, "sine_wave_generator time");
}

#[test]
fn am_sine_generator_matches_mosqito() {
    let g = golden();
    let c = &g["am_sine_generator"];
    let xmod = floats(&c["xmod"]);
    let fs = c["fs"].as_f64().unwrap();
    let fc = c["fc"].as_f64().unwrap();
    let spl_level = c["spl_level"].as_f64().unwrap();
    let want_y = floats(&c["y_am"]);
    let want_m = c["m"].as_f64().unwrap();

    let (got_y, got_m) = am_sine_generator(&xmod, fs, fc, spl_level);
    assert_close(&got_y, &want_y, 1e-9, "am_sine_generator y_am");
    assert!((got_m - want_m).abs() < 1e-12, "am_sine_generator m");
}

#[test]
fn fm_sine_generator_matches_mosqito() {
    let g = golden();
    let c = &g["fm_sine_generator"];
    let xmod = floats(&c["xmod"]);
    let fs = c["fs"].as_f64().unwrap();
    let fc = c["fc"].as_f64().unwrap();
    let k = c["k"].as_f64().unwrap();
    let spl_level = c["spl_level"].as_f64().unwrap();
    let want_y = floats(&c["y_fm"]);
    let want_inst_freq = floats(&c["inst_freq"]);
    let want_f_delta = c["f_delta"].as_f64().unwrap();
    let want_m = c["m"].as_f64().unwrap();

    let (got_y, got_inst_freq, got_f_delta, got_m) = fm_sine_generator(&xmod, fs, fc, k, spl_level);
    assert_close(&got_y, &want_y, 1e-9, "fm_sine_generator y_fm");
    assert_close(
        &got_inst_freq,
        &want_inst_freq,
        1e-9,
        "fm_sine_generator inst_freq",
    );
    assert!(
        (got_f_delta - want_f_delta).abs() < 1e-9,
        "fm_sine_generator f_delta"
    );
    assert!((got_m - want_m).abs() < 1e-9, "fm_sine_generator m");
}

#[test]
fn sone_to_phon_matches_mosqito() {
    let g = golden();
    let c = &g["sone_to_phon"];
    let sones = floats(&c["sones"]);
    let want = floats(&c["phons"]);
    let got: Vec<f64> = sones.iter().map(|&s| sone_to_phon(s)).collect();
    assert_close(&got, &want, 1e-9, "sone_to_phon");
}

#[test]
fn equal_loudness_contours_matches_mosqito() {
    let g = golden();
    for (key, phon) in [("phon_40", 40.0), ("phon_80", 80.0)] {
        let c = &g["equal_loudness_contours"][key];
        let want_spl = floats(&c["spl"]);
        let want_freq = floats(&c["freq"]);
        let (got_spl, got_freq) = equal_loudness_contours(phon);
        assert_close(
            &got_spl,
            &want_spl,
            1e-9,
            &format!("equal_loudness_contours spl {key}"),
        );
        assert_close(
            &got_freq,
            &want_freq,
            1e-9,
            &format!("equal_loudness_contours freq {key}"),
        );
    }
}
