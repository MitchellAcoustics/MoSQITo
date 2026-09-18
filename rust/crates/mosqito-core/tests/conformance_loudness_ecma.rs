//! ECMA-418-2 (2nd Ed, 2022) §5 / Annex A conformance checks for stationary
//! loudness.
//!
//! Unlike ISO 532-1 (whose Annex B publishes exact numeric reference
//! values, checked bit-for-bit in `conformance_iso532_1.rs` and
//! `conformance_loudness_zwtv.rs`), this repository ships no digitized
//! numeric ECMA-418-2 reference corpus for loudness — MoSQITo's own
//! validation for this metric
//! (`validations/sq_metrics/loudness_ecma/hearing_model_validation.py`) is
//! itself only a qualitative plot against ISO 226 equal-loudness contours,
//! with no programmatic pass/fail assertion anywhere in the Python
//! codebase. The primary evidence this port's algorithm is correct is
//! `golden_ecma_loudness.rs`, which checks every stage (centre frequencies,
//! gammatone coefficients, the nonlinearity, and the full pipeline)
//! bit-for-bit against real `mosqito` output.
//!
//! This file adds the one independently-verifiable, standards-anchored
//! property available without a digitized corpus: ECMA-418-2's Sottek
//! Hearing Model shares ISO 532-1 Zwicker loudness's anchor that a 1 kHz
//! tone at 40 dB SPL (= 40 phon, by definition at 1 kHz) has a loudness of
//! approximately 1 sone (`_HMS` here) — the same anchor
//! `hearing_model_validation.py` implicitly relies on by tracing the 40
//! phon equal-loudness contour outward from 1 kHz. A generous ±25% band is
//! used (not the tight tolerances used where an exact published reference
//! exists) since this is a physically-motivated sanity bound, not a
//! standards-pinned value this port claims to hit exactly.

use mosqito_core::loudness::ecma::loudness_ecma;

fn tone(freq: f64, spl_db: f64, duration_s: f64, fs: f64) -> Vec<f64> {
    let n = (duration_s * fs) as usize;
    let amplitude = 2e-5 * 10f64.powf(spl_db / 20.0) * std::f64::consts::SQRT_2;
    (0..n)
        .map(|i| amplitude * (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
        .collect()
}

#[test]
fn loudness_ecma_of_a_1khz_tone_at_40db_spl_is_close_to_one_sone() {
    let fs = 48000.0;
    let sig = tone(1000.0, 40.0, 1.0, fs);
    let (n, _n_time, _n_specific, _bark, _time) = loudness_ecma(&sig, fs, 2048, 1024);

    assert!(
        (0.75..=1.25).contains(&n),
        "N = {n} sone_HMS for a 1 kHz / 40 dB SPL tone, want close to 1.0 \
         (ECMA-418-2's 40-phon anchor, matching ISO 532-1 Zwicker loudness's own)"
    );
}

#[test]
fn loudness_ecma_of_a_1khz_tone_increases_monotonically_with_spl() {
    let fs = 48000.0;
    let levels = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let mut prev = f64::NEG_INFINITY;
    for &spl in &levels {
        let sig = tone(1000.0, spl, 0.5, fs);
        let (n, _n_time, _n_specific, _bark, _time) = loudness_ecma(&sig, fs, 2048, 1024);
        assert!(
            n > prev,
            "N did not increase from {prev} to {n} going from a lower to a higher SPL \
             (at {spl} dB SPL, 1 kHz)"
        );
        prev = n;
    }
}
