//! ANSI S3.5-1997 §3.4 band procedure data: center/edge frequencies,
//! band-importance weights, reference internal noise spectra, and the
//! standard speech spectra at each of the four speech levels — one table set
//! per band procedure.
//!
//! Transcribed directly from `_band_procedure_data.py`/`_speech_data.py`.
//! `FREEFIELD2EARDRUM_TRANSFER_FUNCTION` is computed in the Python for each
//! band layout but never returned or used anywhere in the codebase — dead
//! data, not ported; see `DEVIATIONS.md`.

/// Which of ANSI S3.5's four frequency-band procedures to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiiMethod {
    /// 21 critical bands corresponding to the Bark scale.
    Critical,
    /// 17 equally-contributing critical bands.
    EquallyCritical,
    /// 18 third-octave bands.
    ThirdOctave,
    /// 6 octave bands.
    Octave,
}

/// The speech spectrum level to assess, matching ANSI S3.5's four standard
/// speech spectra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechLevel {
    Normal,
    Raised,
    Loud,
    Shout,
}

/// One band procedure's fixed table data.
pub struct BandData {
    pub center: &'static [f64],
    pub lower: &'static [f64],
    pub upper: &'static [f64],
    pub importance: &'static [f64],
    pub reference_internal_noise: &'static [f64],
}

const CRITICAL_CENTER: [f64; 21] = [
    150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1000.0, 1170.0, 1370.0, 1600.0, 1850.0,
    2150.0, 2500.0, 2900.0, 3400.0, 4000.0, 4800.0, 5800.0, 7000.0, 8500.0,
];
const CRITICAL_LOWER: [f64; 21] = [
    100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0,
    2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0,
];
const CRITICAL_UPPER: [f64; 21] = [
    200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0,
    2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0,
];
const CRITICAL_IMPORTANCE: [f64; 21] = [
    0.0103, 0.0261, 0.0419, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577,
    0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0460, 0.0343, 0.0226, 0.0110,
];
const CRITICAL_REF_NOISE: [f64; 21] = [
    1.50, -3.90, -7.20, -8.90, -10.30, -11.40, -12.00, -12.50, -13.20, -14.00, -15.40, -16.90,
    -18.80, -21.20, -23.20, -24.90, -25.90, -24.20, -19.00, -11.70, -6.00,
];
const CRITICAL_SPEECH_NORMAL: [f64; 21] = [
    31.44, 34.75, 34.14, 34.58, 33.17, 30.64, 27.59, 25.01, 23.52, 22.28, 20.15, 18.29, 16.37,
    13.80, 12.21, 11.09, 9.33, 5.84, 3.47, 1.78, -0.14,
];
const CRITICAL_SPEECH_RAISED: [f64; 21] = [
    34.06, 38.98, 38.62, 39.84, 39.44, 37.99, 35.85, 33.86, 32.56, 30.91, 28.58, 26.37, 24.34,
    22.35, 21.04, 19.56, 16.78, 12.14, 9.04, 6.36, 3.44,
];
const CRITICAL_SPEECH_LOUD: [f64; 21] = [
    34.21, 41.55, 43.68, 44.08, 45.34, 45.22, 43.60, 42.16, 41.07, 39.68, 37.70, 35.62, 33.17,
    30.98, 29.01, 27.71, 25.41, 19.20, 15.37, 12.61, 9.62,
];
const CRITICAL_SPEECH_SHOUT: [f64; 21] = [
    28.69, 42.50, 47.14, 48.46, 50.17, 51.68, 51.43, 51.31, 49.40, 49.03, 47.65, 45.47, 43.13,
    40.80, 39.15, 37.30, 34.41, 29.01, 25.17, 22.08, 18.76,
];

const EQUAL_CENTER: [f64; 17] = [
    350.0, 450.0, 570.0, 700.0, 840.0, 1000.0, 1170.0, 1370.0, 1600.0, 1850.0, 2150.0, 2500.0,
    2900.0, 3400.0, 4000.0, 4800.0, 5800.0,
];
const EQUAL_LOWER: [f64; 17] = [
    300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0,
    2700.0, 3150.0, 3700.0, 4400.0, 5300.0,
];
const EQUAL_UPPER: [f64; 17] = [
    400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0,
    3150.0, 3700.0, 4400.0, 5300.0, 6400.0,
];
const EQUAL_IMPORTANCE: [f64; 17] = [0.0588; 17];
const EQUAL_REF_NOISE: [f64; 17] = [
    -7.20, -8.90, -10.30, -11.40, -12.00, -12.50, -13.20, -14.00, -15.40, -16.90, -18.80, -21.20,
    -23.20, -24.90, -25.90, -24.20, -19.00,
];
const EQUAL_SPEECH_NORMAL: [f64; 17] = [
    34.14, 34.58, 33.17, 30.64, 27.59, 25.01, 23.52, 22.28, 20.15, 18.29, 16.37, 13.80, 12.21,
    11.09, 9.33, 5.84, 3.47,
];
const EQUAL_SPEECH_RAISED: [f64; 17] = [
    38.62, 39.84, 39.44, 37.99, 35.85, 33.86, 32.56, 30.91, 28.58, 26.37, 24.34, 22.35, 21.04,
    19.56, 16.78, 12.14, 9.04,
];
const EQUAL_SPEECH_LOUD: [f64; 17] = [
    43.68, 44.08, 45.34, 45.22, 43.60, 42.16, 41.07, 39.68, 37.70, 35.62, 33.17, 30.98, 29.01,
    27.71, 25.41, 19.20, 15.37,
];
const EQUAL_SPEECH_SHOUT: [f64; 17] = [
    47.14, 48.46, 50.17, 51.68, 51.43, 51.31, 49.40, 49.03, 47.65, 45.47, 43.13, 40.80, 39.15,
    37.30, 34.41, 29.01, 25.17,
];

const OCTAVE_CENTER: [f64; 6] = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
const OCTAVE_LOWER: [f64; 6] = [177.0, 355.0, 710.0, 1420.0, 2840.0, 5680.0];
const OCTAVE_UPPER: [f64; 6] = [355.0, 710.0, 1420.0, 2840.0, 5680.0, 11360.0];
const OCTAVE_IMPORTANCE: [f64; 6] = [0.0617, 0.1671, 0.2373, 0.2648, 0.2142, 0.0549];
const OCTAVE_REF_NOISE: [f64; 6] = [-3.90, -9.70, -12.50, -17.70, -25.90, -7.10];
const OCTAVE_SPEECH_NORMAL: [f64; 6] = [34.75, 34.27, 25.01, 17.32, 9.33, 1.13];
const OCTAVE_SPEECH_RAISED: [f64; 6] = [38.98, 40.15, 33.86, 25.32, 16.78, 5.07];
const OCTAVE_SPEECH_LOUD: [f64; 6] = [41.55, 44.85, 42.16, 34.39, 25.41, 11.39];
const OCTAVE_SPEECH_SHOUT: [f64; 6] = [42.50, 49.24, 51.31, 44.32, 34.41, 20.72];

const THIRD_OCTAVE_CENTER: [f64; 18] = [
    160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0,
    3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
];
const THIRD_OCTAVE_LOWER: [f64; 18] = [
    141.0, 178.0, 224.0, 282.0, 355.0, 447.0, 562.0, 708.0, 891.0, 1122.0, 1413.0, 1778.0, 2239.0,
    2818.0, 3548.0, 4467.0, 5623.0, 7079.0,
];
const THIRD_OCTAVE_UPPER: [f64; 18] = [
    178.0, 224.0, 282.0, 355.0, 447.0, 562.0, 708.0, 891.0, 1122.0, 1413.0, 1778.0, 2239.0, 2818.0,
    3548.0, 4467.0, 5623.0, 7079.0, 8913.0,
];
const THIRD_OCTAVE_IMPORTANCE: [f64; 18] = [
    0.0083, 0.0095, 0.0150, 0.0289, 0.0440, 0.0578, 0.0653, 0.0711, 0.0818, 0.0844, 0.0882, 0.0898,
    0.0868, 0.0844, 0.0771, 0.0527, 0.0364, 0.0185,
];
const THIRD_OCTAVE_REF_NOISE: [f64; 18] = [
    0.60, -1.70, -3.90, -6.10, -8.20, -9.70, -10.80, -11.90, -12.50, -13.50, -15.40, -17.70,
    -21.20, -24.20, -25.90, -23.60, -15.80, -7.10,
];
const THIRD_OCTAVE_SPEECH_NORMAL: [f64; 18] = [
    32.41, 34.48, 34.75, 33.98, 34.59, 34.27, 32.06, 28.30, 25.01, 23.00, 20.15, 17.32, 13.18,
    11.55, 9.33, 5.31, 2.59, 1.13,
];
const THIRD_OCTAVE_SPEECH_RAISED: [f64; 18] = [
    33.81, 33.92, 38.98, 38.57, 39.11, 40.15, 38.78, 36.37, 33.86, 31.89, 28.58, 25.32, 22.35,
    20.15, 16.78, 11.47, 7.67, 5.07,
];
const THIRD_OCTAVE_SPEECH_LOUD: [f64; 18] = [
    35.29, 37.76, 41.55, 43.78, 43.40, 44.85, 45.55, 44.05, 42.16, 40.53, 37.70, 34.39, 30.98,
    28.21, 25.41, 18.35, 13.87, 11.39,
];
const THIRD_OCTAVE_SPEECH_SHOUT: [f64; 18] = [
    30.77, 36.65, 42.50, 46.51, 47.40, 49.24, 51.21, 51.44, 51.31, 49.63, 47.65, 44.32, 40.80,
    38.13, 34.41, 28.24, 23.45, 20.72,
];

/// The fixed band-procedure table for `method`.
pub fn band_data(method: SiiMethod) -> BandData {
    match method {
        SiiMethod::Critical => BandData {
            center: &CRITICAL_CENTER,
            lower: &CRITICAL_LOWER,
            upper: &CRITICAL_UPPER,
            importance: &CRITICAL_IMPORTANCE,
            reference_internal_noise: &CRITICAL_REF_NOISE,
        },
        SiiMethod::EquallyCritical => BandData {
            center: &EQUAL_CENTER,
            lower: &EQUAL_LOWER,
            upper: &EQUAL_UPPER,
            importance: &EQUAL_IMPORTANCE,
            reference_internal_noise: &EQUAL_REF_NOISE,
        },
        SiiMethod::ThirdOctave => BandData {
            center: &THIRD_OCTAVE_CENTER,
            lower: &THIRD_OCTAVE_LOWER,
            upper: &THIRD_OCTAVE_UPPER,
            importance: &THIRD_OCTAVE_IMPORTANCE,
            reference_internal_noise: &THIRD_OCTAVE_REF_NOISE,
        },
        SiiMethod::Octave => BandData {
            center: &OCTAVE_CENTER,
            lower: &OCTAVE_LOWER,
            upper: &OCTAVE_UPPER,
            importance: &OCTAVE_IMPORTANCE,
            reference_internal_noise: &OCTAVE_REF_NOISE,
        },
    }
}

/// The standard speech spectrum (dB re. 2e-5 Pa) for `method`/`level`.
pub fn speech_spectrum(method: SiiMethod, level: SpeechLevel) -> &'static [f64] {
    match (method, level) {
        (SiiMethod::Critical, SpeechLevel::Normal) => &CRITICAL_SPEECH_NORMAL,
        (SiiMethod::Critical, SpeechLevel::Raised) => &CRITICAL_SPEECH_RAISED,
        (SiiMethod::Critical, SpeechLevel::Loud) => &CRITICAL_SPEECH_LOUD,
        (SiiMethod::Critical, SpeechLevel::Shout) => &CRITICAL_SPEECH_SHOUT,
        (SiiMethod::EquallyCritical, SpeechLevel::Normal) => &EQUAL_SPEECH_NORMAL,
        (SiiMethod::EquallyCritical, SpeechLevel::Raised) => &EQUAL_SPEECH_RAISED,
        (SiiMethod::EquallyCritical, SpeechLevel::Loud) => &EQUAL_SPEECH_LOUD,
        (SiiMethod::EquallyCritical, SpeechLevel::Shout) => &EQUAL_SPEECH_SHOUT,
        (SiiMethod::ThirdOctave, SpeechLevel::Normal) => &THIRD_OCTAVE_SPEECH_NORMAL,
        (SiiMethod::ThirdOctave, SpeechLevel::Raised) => &THIRD_OCTAVE_SPEECH_RAISED,
        (SiiMethod::ThirdOctave, SpeechLevel::Loud) => &THIRD_OCTAVE_SPEECH_LOUD,
        (SiiMethod::ThirdOctave, SpeechLevel::Shout) => &THIRD_OCTAVE_SPEECH_SHOUT,
        (SiiMethod::Octave, SpeechLevel::Normal) => &OCTAVE_SPEECH_NORMAL,
        (SiiMethod::Octave, SpeechLevel::Raised) => &OCTAVE_SPEECH_RAISED,
        (SiiMethod::Octave, SpeechLevel::Loud) => &OCTAVE_SPEECH_LOUD,
        (SiiMethod::Octave, SpeechLevel::Shout) => &OCTAVE_SPEECH_SHOUT,
    }
}
