//! DIN 45692:2009 sharpness from loudness, and the four time/frequency
//! entry points built on the ISO 532-1 stationary (`loudness::zwst`) and
//! time-varying (`loudness::zwtv`) loudness pipelines.

use ndarray::Array2;

use crate::dsp::interp;
use crate::loudness::zwst::{bark_axis, loudness_zwst, loudness_zwst_freq, FieldType};
use crate::loudness::zwtv::{loudness_zwtv, LoudnessZwtvError};

/// The four sharpness weighting functions `sharpness_din_from_loudness`
/// supports (`_weighting_fastl.py` plus the three closed-form curves in
/// `sharpness_din_from_loudness.py:123-136`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weighting {
    Din,
    Aures,
    Bismarck,
    Fastl,
}

impl Weighting {
    /// Parses MoSQITo's `weighting` string argument.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "din" => Ok(Weighting::Din),
            "aures" => Ok(Weighting::Aures),
            "bismarck" => Ok(Weighting::Bismarck),
            "fastl" => Ok(Weighting::Fastl),
            other => Err(format!(
                "weighting must be 'din', 'aures', 'bismarck' or 'fastl', got {other:?}"
            )),
        }
    }
}

// Zwicker & Fastl weighting curve (`_weighting_fastl.py`), used with linear
// interpolation (`numpy.interp` semantics: clamped, not extrapolated, at
// either end).
const FASTL_X: [f64; 51] = [
    0.0985854, 14.826764, 15.364039, 15.863559, 16.297388, 16.533346, 16.844551, 17.052244,
    17.335314, 17.599474, 17.816587, 18.014925, 18.231972, 18.38313, 18.543644, 18.704224,
    18.883648, 18.978205, 19.167118, 19.346476, 19.441233, 19.554567, 19.668102, 19.762926,
    19.923306, 20.036907, 20.188398, 20.320976, 20.462912, 20.60518, 20.78507, 20.908228,
    21.078962, 21.268276, 21.410543, 21.533966, 21.704834, 21.847036, 21.98957, 22.141394,
    22.312195, 22.464752, 22.62633, 22.759441, 22.89302, 23.054932, 23.236088, 23.38871, 23.5416,
    23.6848, 23.932978,
];
const FASTL_Y: [f64; 51] = [
    0.9783246, 0.99701804, 1.0092967, 1.0169811, 1.0438102, 1.0713409, 1.0891489, 1.1167798,
    1.1441435, 1.1668463, 1.1944437, 1.2268357, 1.2497056, 1.277537, 1.3006072, 1.3284053,
    1.3561363, 1.3794405, 1.411866, 1.4348694, 1.4723566, 1.4908663, 1.523559, 1.5657737,
    1.5793887, 1.6168091, 1.6682787, 1.7150875, 1.7571354, 1.8228214, 1.8836461, 1.9304883,
    2.0102572, 2.0710485, 2.1367345, 2.2024875, 2.2917116, 2.35267, 2.4372666, 2.5123746,
    2.5968711, 2.7239833, 2.8226962, 2.9073262, 3.02505, 3.147401, 3.2980514, 3.429891, 3.5806417,
    3.7125149, 3.9385738,
];

fn g(weighting: Weighting, z: f64, n: f64) -> f64 {
    match weighting {
        Weighting::Din => {
            if z > 15.8 {
                0.15 * (0.42 * (z - 15.8)).exp() + 0.85
            } else {
                1.0
            }
        }
        Weighting::Bismarck => {
            if z > 15.0 {
                0.2 * (0.308 * (z - 15.0)).exp() + 0.8
            } else {
                1.0
            }
        }
        Weighting::Fastl => interp(&[z], &FASTL_X, &FASTL_Y)[0],
        // Unlike the other three weightings, this one depends on the
        // segment's own overall loudness `N` (`sharpness_din_from_loudness.py:127`).
        Weighting::Aures => 0.078 * ((0.171 * z).exp() / z) * (n / (n * 0.05 + 1.0).ln()),
    }
}

/// Sharpness from a single loudness result, matching
/// `sharpness_din_from_loudness(N, N_specific, weighting)` for scalar `N`
/// (the `S.size == 1` branch, which — for every real caller, since it is
/// driven purely by how many segments there are, not by whether `N` started
/// out as a Python float or a length-1 array — never applies the `N < 0.1`
/// masking the segmented branch below does).
pub fn sharpness_din_from_loudness(n: f64, n_specific: &[f64; 240], weighting: Weighting) -> f64 {
    let z = bark_axis();
    let acc: f64 = (0..240)
        .map(|k| n_specific[k] * g(weighting, z[k], n) * z[k] * 0.1)
        .sum();
    0.11 * acc / n
}

/// Sharpness per segment, matching `sharpness_din_from_loudness(N,
/// N_specific, weighting)` for an `N` array of more than one segment (the
/// `else` branch of `sharpness_din_from_loudness.py:142-146`): additionally
/// zeroes any segment whose overall loudness is below 0.1 sone, where the
/// division would otherwise be dominated by noise.
///
/// `n_specific` is (240, nseg), one column per segment in `n`.
///
/// # Panics
/// Panics if `n_specific`'s column count does not match `n.len()`.
pub fn sharpness_din_from_loudness_segmented(
    n: &[f64],
    n_specific: &Array2<f64>,
    weighting: Weighting,
) -> Vec<f64> {
    assert_eq!(
        n_specific.ncols(),
        n.len(),
        "N_specific must have one column per segment in N"
    );
    let z = bark_axis();
    n.iter()
        .enumerate()
        .map(|(seg, &n_seg)| {
            if n_seg < 0.1 {
                return 0.0;
            }
            let acc: f64 = (0..240)
                .map(|k| n_specific[[k, seg]] * g(weighting, z[k], n_seg) * z[k] * 0.1)
                .sum();
            0.11 * acc / n_seg
        })
        .collect()
}

/// Sharpness from a time signal. Matches `sharpness_din_st(signal, fs,
/// weighting, field_type)`.
///
/// MoSQITo's Python resamples to 48 kHz itself before calling
/// `loudness_zwst` (`sharpness_din_st.py:97-113`, in a duplicated block that
/// only ever runs once — see `DEVIATIONS.md`); `loudness_zwst` here already
/// does the same resampling internally, so this calls it directly.
pub fn sharpness_din_st(
    signal: &[f64],
    fs: f64,
    weighting: Weighting,
    field_type: FieldType,
) -> f64 {
    let (n, n_specific, _bark) = loudness_zwst(signal, fs, field_type);
    sharpness_din_from_loudness(n, &n_specific, weighting)
}

/// Sharpness from a fine-band spectrum. Matches `sharpness_din_freq(spectrum,
/// freqs, weighting, field_type)` for a 1-D spectrum (the 2-D case in
/// MoSQITo's Python always raises `ValueError` after computing the loudness
/// anyway — see `sharpness_din_freq.py:164-165` — so it carries no real
/// behaviour to port).
pub fn sharpness_din_freq(
    spectrum: &[f64],
    freqs: &[f64],
    weighting: Weighting,
    field_type: FieldType,
) -> f64 {
    let (n, n_specific, _bark) = loudness_zwst_freq(spectrum, freqs, field_type);
    sharpness_din_from_loudness(n, &n_specific, weighting)
}

/// Sharpness per time segment. Matches `sharpness_din_perseg(signal, fs,
/// weighting, nperseg, noverlap, field_type)`.
pub fn sharpness_din_perseg(
    signal: &[f64],
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
    weighting: Weighting,
    field_type: FieldType,
) -> Result<(Vec<f64>, Vec<f64>), crate::loudness::zwst::LoudnessZwstError> {
    let (n, n_specific, _bark, time) =
        crate::loudness::zwst::loudness_zwst_perseg(signal, fs, nperseg, noverlap, field_type)?;
    let s = sharpness_din_from_loudness_segmented(&n, &n_specific, weighting);
    Ok((s, time))
}

/// Sharpness along time from a time-varying signal. Matches
/// `sharpness_din_tv(signal, fs, weighting, field_type, skip)`.
///
/// `skip` cuts the leading transient (`loudness_zwtv`'s nonlinear decay
/// stage takes a few frames to settle, per its own module doc): the
/// returned arrays start at the first output frame whose time is closest to
/// `skip` seconds (`sharpness_din_tv.py:123`, `argmin(abs(time_axis -
/// skip))` — nearest, not the first frame *at or after* `skip`).
pub fn sharpness_din_tv(
    signal: &[f64],
    fs: f64,
    weighting: Weighting,
    field_type: FieldType,
    skip: f64,
) -> Result<(Vec<f64>, Vec<f64>), LoudnessZwtvError> {
    let (n, n_specific, _bark, time) = loudness_zwtv(signal, fs, field_type)?;
    let s = sharpness_din_from_loudness_segmented(&n, &n_specific, weighting);

    let cut_index = time
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| (a - skip).abs().total_cmp(&(b - skip).abs()))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok((s[cut_index..].to_vec(), time[cut_index..].to_vec()))
}
