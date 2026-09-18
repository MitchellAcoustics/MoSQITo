//! Core loudness computation, per ISO 532-1:2017 Annex A (the equal-loudness
//! and critical-band correction stage of the Zwicker stationary method).
//!
//! This reproduces `_main_loudness.py`'s actual computed values, including one
//! subtlety worth flagging rather than silently fixing: the low-frequency
//! correction search (`dll_result` below) only checks for a threshold
//! transition across `RAP` levels 1 through 7 (0-indexed 0..6), never the
//! transition into the 8th (last) level. Whether that is a genuine gap in the
//! original BASIC-to-numpy vectorisation or an intentional simplification is
//! not resolvable from the Python source alone, and this crate has no access
//! to the ISO 532-1 text to settle it independently. What *is* verifiable is
//! that MoSQITo's own validation passes ISO 532-1's published reference
//! values at the required tolerance with this behaviour in place
//! (`validations/sq_metrics/loudness_zwst/`), so reproducing it exactly —
//! checked here against the installed `mosqito` package across hundreds of
//! random spectra, not just the reference corpus — is the safe choice; see
//! `DEVIATIONS.md`.

/// Ranges of 1/3-octave band levels for the low-frequency correction, per
/// equal-loudness contours (ISO 532-1:2017 Table A.3 column headers).
const RAP: [f64; 8] = [45.0, 55.0, 65.0, 71.0, 80.0, 90.0, 100.0, 120.0];

/// Reduction of 1/3-octave band levels at low frequencies within each of the
/// 8 `RAP` ranges, for the first 11 bands (25 Hz–250 Hz).
const DLL: [[f64; 11]; 8] = [
    [
        -32.0, -24.0, -16.0, -10.0, -5.0, 0.0, -7.0, -3.0, 0.0, -2.0, 0.0,
    ],
    [
        -29.0, -22.0, -15.0, -10.0, -4.0, 0.0, -7.0, -2.0, 0.0, -2.0, 0.0,
    ],
    [
        -27.0, -19.0, -14.0, -9.0, -4.0, 0.0, -6.0, -2.0, 0.0, -2.0, 0.0,
    ],
    [
        -25.0, -17.0, -12.0, -9.0, -3.0, 0.0, -5.0, -2.0, 0.0, -2.0, 0.0,
    ],
    [
        -23.0, -16.0, -11.0, -7.0, -3.0, 0.0, -4.0, -1.0, 0.0, -1.0, 0.0,
    ],
    [
        -20.0, -14.0, -10.0, -6.0, -3.0, 0.0, -4.0, -1.0, 0.0, -1.0, 0.0,
    ],
    [
        -18.0, -12.0, -9.0, -6.0, -2.0, 0.0, -3.0, -1.0, 0.0, -1.0, 0.0,
    ],
    [
        -15.0, -10.0, -8.0, -4.0, -2.0, 0.0, -3.0, -1.0, 0.0, -1.0, 0.0,
    ],
];

/// Critical band level at the absolute threshold of hearing, ignoring the
/// ear's transmission characteristics — one entry per critical band from
/// ~150 Hz up (20 bands: 3 combined low-frequency bands + 17 third-octave
/// bands from 400 Hz to 12500 Hz).
const LTQ: [f64; 20] = [
    30.0, 18.0, 12.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    3.0, 3.0,
];

/// Correction for the ear's transmission characteristics, per critical band.
const A0: [f64; 20] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -1.6, -3.2, -5.4, -5.6, -4.0, -1.5,
    2.0, 5.0, 12.0,
];

/// Level difference between free and diffuse sound fields, per critical band.
const DDF: [f64; 20] = [
    0.0, 0.0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3.0, 2.0, 0.0, -1.4, -2.0, -1.9, -1.0, 0.5, 3.0, 4.0,
    4.3, 4.0,
];

/// Adaptation of 1/3-octave band levels to critical band levels.
const DCB: [f64; 20] = [
    -0.25, -0.6, -0.8, -0.8, -0.5, 0.0, 0.5, 1.1, 1.5, 1.7, 1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 0.8, 0.5,
    0.0, -0.5,
];

/// Sound field type `_main_loudness` corrects for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Free,
    Diffuse,
}

/// Computes core loudness for one third-octave band spectrum.
///
/// `spec_third` is 28 third-octave band levels in dB (25 Hz–12500 Hz, ISO
/// 532-1's fixed band layout). Returns 21 core loudness values (per critical
/// band, plus a trailing zero anchoring the slope-attachment stage at 24
/// Bark — see `calc_slopes`).
///
/// # Panics
/// Panics if `spec_third` has fewer than 28 entries, or if any of the first
/// 11 bands (25 Hz–250 Hz) exceeds 120 dB — ISO 532-1's low-frequency
/// correction table is only specified up to that level.
pub fn main_loudness(spec_third: &[f64], field_type: FieldType) -> [f64; 21] {
    assert!(
        spec_third.len() >= 28,
        "expected at least 28 third-octave bands"
    );
    let max_low_band = spec_third[0..11]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_low_band <= 120.0,
        "1/3 octave band value {max_low_band} dB exceeds 120 dB, for which the Zwicker method is no longer valid"
    );

    // Low-frequency correction (ISO 532-1 Annex A.3): for each of the first
    // 11 bands, find where the band level crosses a RAP threshold and use the
    // corresponding DLL correction. See the module doc on the range here.
    let mut dll_result = [0.0f64; 11];
    for j in 0..11 {
        let logic: [bool; 8] = std::array::from_fn(|i| spec_third[j] > RAP[i] - DLL[i][j]);
        let mut val = if logic[0] { 0.0 } else { DLL[0][j] };
        for i in 1..7 {
            if logic[i - 1] != logic[i] {
                val = DLL[i][j];
            }
        }
        dll_result[j] = val;
    }

    // Intensities for the first 11 bands after correction, combined into 3
    // critical-band levels (LCB1: 25-80 Hz, LCB2: 100-160 Hz, LCB3: 200-250 Hz).
    let mut ti = [0.0f64; 11];
    for j in 0..11 {
        ti[j] = (10f64).powf((dll_result[j] + spec_third[j]) / 10.0);
    }
    let gi = [
        ti[0..6].iter().sum::<f64>(),
        ti[6..9].iter().sum::<f64>(),
        ti[9..11].iter().sum::<f64>(),
    ];
    let lcb: [f64; 3] = std::array::from_fn(|k| {
        if gi[k] > 0.0 {
            10.0 * gi[k].log10()
        } else {
            0.0
        }
    });

    // Assemble the 20 critical-band levels: the 3 combined low-frequency
    // bands, then the 17 third-octave bands from 400 Hz to 12500 Hz
    // (spec_third indices 11..28) unchanged.
    let mut le = [0.0f64; 20];
    le[0..3].copy_from_slice(&lcb);
    le[3..20].copy_from_slice(&spec_third[11..28]);
    for j in 0..20 {
        le[j] -= A0[j];
    }
    if field_type == FieldType::Diffuse {
        for j in 0..20 {
            le[j] += DDF[j];
        }
    }

    // Non-linear transform to loudness, ISO 532-1 eq. A.4-ish (Zwicker's
    // power-law loudness function), applied only above the threshold of
    // hearing.
    let s = 0.25;
    let mut nm = [0.0f64; 21];
    for j in 0..20 {
        if le[j] > LTQ[j] {
            let corrected = le[j] - DCB[j];
            let mp1 = 0.0635 * (10f64).powf(0.025 * LTQ[j]);
            let mp2 = (1.0 - s + s * (10f64).powf(0.1 * (corrected - LTQ[j]))).powf(0.25) - 1.0;
            nm[j] = (mp1 * mp2).max(0.0);
        }
    }
    // nm[20] stays 0.0 — the trailing anchor `calc_slopes` decays down to.

    // Correction of specific loudness in the lowest critical band for the
    // dependence of the absolute threshold within that band.
    let korry = 0.4 + 0.32 * nm[0].powf(0.2);
    if korry <= 1.0 {
        nm[0] *= korry;
    }

    nm
}
