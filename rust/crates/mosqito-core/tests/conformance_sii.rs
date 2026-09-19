//! ANSI S3.5-1997 conformance gate for SII, using the worked example
//! transcribed directly from the standard in
//! `validations/sq_metrics/speech_intelligibility/validation_sii.py`
//! (octave-band procedure).
//!
//! Tolerance is the wider of ±1% or ±0.01, matching that validation script's
//! own `amin`/`amax` tolerance-band construction — this is the one Phase 2
//! metric with a real standards-anchored numeric oracle (not a value
//! digitised off a published plot, unlike `roughness_dw`'s or
//! `roughness_ecma`'s Annex C corpora).

use mosqito_core::speech_intelligibility::{main_sii, SiiMethod, SiiThreshold, SpeechLevel};

/// ±1%/±0.01, whichever is wider, matching the validation script's
/// `amin([ref*0.99, ref-0.01])` / `amax([ref*1.01, ref+0.01])` bounds.
#[track_caller]
fn assert_within_ansi_tolerance(got: f64, want: f64, what: &str) {
    let lo = (want * 0.99).min(want - 0.01);
    let hi = (want * 1.01).max(want + 0.01);
    assert!(
        got >= lo && got <= hi,
        "{what}: {got} outside ANSI tolerance [{lo}, {hi}] (want {want})"
    );
}

#[test]
fn sii_octave_matches_the_ansi_s3_5_worked_example() {
    // ANSI S3.5-1997's own worked example, octave-band procedure.
    let noise_spectrum = [70.0, 65.0, 45.0, 25.0, 1.0, -15.0];
    let speech_spectrum = [50.0, 40.0, 40.0, 30.0, 20.0, 0.0];
    let want_sii = 0.504;
    let want_spec = [0.0, 0.0, 0.08, 0.17, 0.21, 0.04];

    let (sii, sii_spec, _) = main_sii(
        SiiMethod::Octave,
        &speech_spectrum,
        &noise_spectrum,
        SiiThreshold::Zero,
    );

    assert_within_ansi_tolerance(sii, want_sii, "SII");
    for (i, (&got, &want)) in sii_spec.iter().zip(&want_spec).enumerate() {
        assert_within_ansi_tolerance(got, want, &format!("SII_specific[{i}]"));
    }
}

#[test]
fn sii_is_higher_for_a_quieter_noise_floor() {
    // Sanity check independent of the ANSI worked example: a uniformly
    // quieter noise spectrum must not reduce intelligibility.
    let speech = [50.0, 40.0, 40.0, 30.0, 20.0, 0.0];
    let loud_noise = [70.0, 65.0, 45.0, 25.0, 1.0, -15.0];
    let quiet_noise: [f64; 6] = loud_noise.map(|n| n - 20.0);

    let (sii_loud, _, _) = main_sii(SiiMethod::Octave, &speech, &loud_noise, SiiThreshold::Zero);
    let (sii_quiet, _, _) = main_sii(SiiMethod::Octave, &speech, &quiet_noise, SiiThreshold::Zero);

    assert!(sii_quiet > sii_loud);
    assert!((0.0..=1.0).contains(&sii_quiet));
    assert!((0.0..=1.0).contains(&sii_loud));

    // Speech levels used by every entry point, spanning all four band
    // procedures, all sit in [0, 1] — asserted here as a broad sanity net
    // rather than repeated per-metric everywhere SpeechLevel is used.
    for method in [
        SiiMethod::Critical,
        SiiMethod::EquallyCritical,
        SiiMethod::ThirdOctave,
        SiiMethod::Octave,
    ] {
        for level in [
            SpeechLevel::Normal,
            SpeechLevel::Raised,
            SpeechLevel::Loud,
            SpeechLevel::Shout,
        ] {
            let speech = mosqito_core::speech_intelligibility::speech_spectrum(method, level);
            let noise: Vec<f64> = speech.iter().map(|&s| s - 20.0).collect();
            let (sii, _, _) = main_sii(method, speech, &noise, SiiThreshold::Zero);
            assert!(
                (0.0..=1.0).contains(&sii),
                "{method:?}/{level:?}: SII {sii} out of [0,1]"
            );
        }
    }
}
