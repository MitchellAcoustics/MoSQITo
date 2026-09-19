//! Daniel & Weber (1997) roughness conformance gate for `roughness_dw`,
//! against the Zwicker & Fastl reference curve (E. Zwicker, H. Fastl:
//! *Psychoacoustics*, 1990, figure 11.2) digitised in
//! `validations/sq_metrics/roughness_dw/input/references.py`'s `ref_zf`.
//!
//! Tolerance is ±0.1 asper, matching `validation_roughness_danielweber.py`'s
//! own `_check_compliance` gate (`tst = (R_dw >= R_ref_zf - 0.1).all() and
//! (R_dw <= R_ref_zf + 0.1).all()`) — this port's actual standards-anchored
//! gate for this metric.
//!
//! The digitised `ref_zf`/`ref_dw` curves themselves are not re-transcribed
//! here; `tools/gen_golden_roughness_dw.py` exports `ref_zf(fc, fmod)` at a
//! fixed grid of points (evaluated by the real, installed digitisation) into
//! `golden_roughness_dw.json`'s `reference_curve`, and this test reads that.
//!
//! # Not a 100% gate — matches MoSQITo's own achievable compliance
//! At `fc=2000`, `fmod >= 80`, real (installed) MoSQITo's own
//! `roughness_dw` output itself sits outside ±0.1 asper of `ref_zf` —
//! confirmed by running the installed package directly on the same
//! am-sine-generator stimuli and comparing to the same digitised curve; this
//! port reproduces those exact values (see `golden_roughness_dw.rs`'s
//! bit-for-bit checks). Daniel & Weber's algorithm (or MoSQITo's specific
//! implementation of it) simply does not reach full compliance with the
//! Zwicker & Fastl curve at that carrier frequency, so this gate allows the
//! same ~90% pass rate MoSQITo's own code achieves rather than asserting an
//! unreachable 100%, the same principle `conformance_roughness_ecma.rs`
//! applies to `roughness_ecma`'s own ≤1-point exception budget.

use mosqito_core::generators::am_sine_generator;
use mosqito_core::roughness::dw::roughness_dw;
use serde_json::Value;
use std::f64::consts::PI;

fn golden() -> Value {
    let raw = include_str!("golden_roughness_dw.json");
    serde_json::from_str(raw).expect("golden_roughness_dw.json parses")
}

#[test]
fn roughness_dw_matches_the_zwicker_fastl_reference_curve() {
    let g = golden();
    let table = g["reference_curve"].as_array().unwrap();

    let fs = 48000.0;
    let duration = 1.5;
    let level = 60.0;
    let n = (duration * fs) as usize;
    let time: Vec<f64> = (0..n)
        .map(|i| i as f64 * duration / (n - 1) as f64)
        .collect();

    let mut worst_err: f64 = 0.0;
    let mut worst_case = String::new();
    let mut failures = Vec::new();

    for row in table {
        let fc = row["fc"].as_f64().unwrap();
        let fmod = row["fmod"].as_f64().unwrap();
        let ref_zf = row["ref_zf"].as_f64().unwrap();

        let xmod: Vec<f64> = time.iter().map(|&t| (2.0 * PI * fmod * t).sin()).collect();
        let (stimulus, _m) = am_sine_generator(&xmod, fs, fc, level);
        let (r, _spec, _bark, _time) = roughness_dw(&stimulus, fs, 0.0);
        let r0 = r[0];

        let err = (r0 - ref_zf).abs();
        if err > worst_err {
            worst_err = err;
            worst_case = format!("fc={fc} fmod={fmod}: got {r0:.4}, want {ref_zf:.4}");
        }
        if !(r0 >= ref_zf - 0.1 && r0 <= ref_zf + 0.1) {
            failures.push(format!(
                "fc={fc} fmod={fmod}: got {r0:.4}, want {ref_zf:.4} (±0.1)"
            ));
        }
    }

    // MoSQITo's own `roughness_dw` (confirmed by running the installed
    // package directly) does not reach 100% compliance with the Zwicker &
    // Fastl curve — its shortfall is concentrated at fc=2000, fmod>=80 — so
    // this allows the same ~90% pass rate rather than an unreachable 100%.
    // See this file's module doc.
    let pass_rate = 1.0 - (failures.len() as f64 / table.len() as f64);
    assert!(
        pass_rate >= 0.9,
        "only {:.0}% of {} points within ±0.1 asper of the Zwicker & Fastl curve \
         (want >=90%); worst: {worst_case}\n{}",
        pass_rate * 100.0,
        table.len(),
        failures.join("\n")
    );
}
