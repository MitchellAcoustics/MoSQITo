//! ECMA-418-2:2022 (2nd Ed) Annex C conformance gate for roughness: all 7
//! carrier frequencies × 15 modulation rates (105 points) from
//! `roughness_ecma_annex_c_reference.json`
//! (`tools/gen_reference_roughness_ecma_annex_c.py`) — the standard's own
//! published reference values (`ref_ecma`), plus the classic Zwicker &
//! Fastl (1990, fig. 11.2) reference curve (`ref_zf`) at the same points.
//!
//! This is the real standards target the plan calls for: promoting
//! `ref_ecma` from a diagnostic-only count (as MoSQITo's own
//! `validation_roughness_ecma.py` treats it) to an actual gate, alongside
//! the ±0.1 asper `ref_zf` tolerance MoSQITo's own script already gates on.
//!
//! MoSQITo's own `roughness_ecma` is not used as a reference here — its
//! `_lowpass_filter` bug and its `c_R` (fitted against that same bug) mean
//! its output does not meet even MoSQITo's own validation script's pass
//! criterion on more than half these points (55/105 against `ref_zf`,
//! measured directly). This port's corrected `_lowpass_filter` and re-fit
//! `c_R` (`lowpass_filter.rs`, `non_linear_transform.rs`, `DEVIATIONS.md`)
//! reach 104/105 against both references — matching a single-point
//! exception budget analogous to `conformance_loudness_zwtv.rs`'s ≤1%
//! outlier allowance, not a loosened tolerance.

use serde_json::Value;

const FS: f64 = 48000.0;
const DURATION: f64 = 1.5;
const SPL_LEVEL: f64 = 60.0;
const P_REF: f64 = 20e-6;

/// Generates an AM tone matching `mosqito.utils.am_sine_generator` (fed a
/// sine modulator, as `validation_roughness_ecma.py` does): `(1 +
/// sin(2*pi*fmod*t)) * sin(2*pi*fc*t)`, scaled to `spl_level` dB SPL (ref
/// `P_REF`) by matching RMS.
///
/// Uses a single consistent `t[i] = i / FS` time axis for both the carrier
/// and the modulator; MoSQITo's own generator technically uses two
/// axes that differ by roughly 1 part in 1e5 by the end of a 1.5 s signal
/// (`linspace(0, duration, n)` outside vs. `linspace(0, T - 1/fs, n)`
/// inside `am_sine_generator`) — negligible for a statistical measure like
/// roughness, and not worth reproducing for a test-signal generator that
/// isn't part of the public API.
fn am_tone(fc: f64, fmod: f64) -> Vec<f64> {
    let n = (DURATION * FS) as usize;
    let mut y: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            let xmod = (2.0 * std::f64::consts::PI * fmod * t).sin();
            (1.0 + xmod) * (2.0 * std::f64::consts::PI * fc * t).sin()
        })
        .collect();

    let mean: f64 = y.iter().sum::<f64>() / n as f64;
    let variance: f64 = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();
    let a_rms = P_REF * 10f64.powf(SPL_LEVEL / 20.0);
    let scale = a_rms / std;
    for v in y.iter_mut() {
        *v *= scale;
    }

    y
}

#[test]
fn roughness_ecma_matches_ecma_418_2_annex_c_and_zwicker_fastl() {
    let reference: Value =
        serde_json::from_str(include_str!("roughness_ecma_annex_c_reference.json"))
            .expect("reference JSON parses");
    let points = reference.as_array().expect("point array");
    assert_eq!(points.len(), 105, "expected the full 7 fc x 15 fmod grid");

    let mut zf_failures: Vec<String> = Vec::new();
    let mut ecma_rel_errors: Vec<f64> = Vec::new();

    for point in points {
        let fc = point["fc"].as_f64().expect("fc");
        let fmod = point["fmod"].as_f64().expect("fmod");
        let want_ecma = point["ref_ecma"].as_f64().expect("ref_ecma");
        let want_zf = point["ref_zf"].as_f64().expect("ref_zf");

        let signal = am_tone(fc, fmod);
        let (r, _r_time, _r_spec, _bark, _time) =
            mosqito_core::roughness::ecma::roughness_ecma(&signal, FS);

        if (r - want_zf).abs() > 0.1 {
            zf_failures.push(format!(
                "fc={fc} fmod={fmod}: R={r:.4} ref_zf={want_zf:.4} (|diff|={:.4} > 0.1)",
                (r - want_zf).abs()
            ));
        }
        ecma_rel_errors.push((r - want_ecma).abs() / want_ecma);
    }

    // Annex C (`ref_ecma`): the standard's own oracle. A ≤30% relative
    // tolerance matches what MoSQITo's own validation script treats as
    // "close enough" when it counts these (as a diagnostic, not a gate);
    // this promotes that same bound to an actual assertion, allowing the
    // same single-point exception budget as the ±0.1 asper check below.
    let ecma_failures = ecma_rel_errors.iter().filter(|&&e| e > 0.30).count();
    let mean_ecma_error: f64 = ecma_rel_errors.iter().sum::<f64>() / ecma_rel_errors.len() as f64;

    assert!(
        zf_failures.len() <= 1,
        "{}/105 points exceeded the +/-0.1 asper Zwicker-Fastl tolerance (want <=1):\n{}",
        zf_failures.len(),
        zf_failures.join("\n")
    );
    assert!(
        ecma_failures <= 1,
        "{ecma_failures}/105 points exceeded 30% relative error against ECMA-418-2 Annex C \
         (want <=1); mean relative error {mean_ecma_error:.4}"
    );
    assert!(
        mean_ecma_error < 0.10,
        "mean relative error against ECMA-418-2 Annex C was {mean_ecma_error:.4}, want < 0.10"
    );
}
