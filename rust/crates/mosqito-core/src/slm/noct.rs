//! N-th octave band spectrum analysis, per ANSI S1.1-1986 ("Specifications for
//! Octave-Band and Fractional-Octave-Band Analog and Digital Filters").
//!
//! Two entry points, matching MoSQITo's `noct_spectrum` and `noct_synthesis`:
//! [`noct_spectrum`] measures band levels directly from a time signal by
//! filtering it through a bank of bandpass filters; [`noct_synthesis`] derives
//! the same band levels from an existing frequency spectrum by evaluating each
//! filter's frequency response instead. Both use a fixed filter order of 3,
//! matching the only order MoSQITo ever designs with.
//!
//! `noct_spectrum` accepts a 2-D signal, axis 0 = samples and axis 1 =
//! segments, because `loudness_zwst_perseg` calls it directly on the output of
//! `time_segmentation` rather than looping itself. `noct_synthesis` is 1-D only
//! for now: its only tested and exercised use (`loudness_zwst_freq`'s 1-D path)
//! is 1-D, and the 2-D case in MoSQITo's Python has two further sub-cases
//! (shared vs. per-segment frequency axis) that are better added when a
//! concrete caller needs them.

use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::f64::consts::PI;

use crate::dsp::{butter_bandpass_sos, decimate, sosfilt, sosfreqz, Sos};

/// Filter order used throughout: MoSQITo always designs order-3 filters here
/// (`_n_oct_time_filter`'s `N=3` default, never overridden by its callers).
const FILTER_ORDER: usize = 3;

/// Errors from designing or applying an n-th octave filter bank.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoctError {
    /// A band's center frequency exceeds `0.88 * fs/2`, the limit ANSI S1.1
    /// filter design keeps clear of the Nyquist frequency.
    CenterFrequencyTooHigh { fc: f64, nyquist: f64 },
    /// The signal (after any decimation) is too short to filter.
    SignalTooShort,
    /// `noct_synthesis` requires the frequency axis to imply a 48 kHz sampling
    /// rate, matching the ISO 532-1 signals it is used for.
    SamplingFrequencyMismatch { implied_fs: f64 },
}

impl std::fmt::Display for NoctError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NoctError::CenterFrequencyTooHigh { fc, nyquist } => write!(
                f,
                "center frequency {fc} Hz exceeds 0.88 * Nyquist ({:.1} Hz)",
                0.88 * nyquist
            ),
            NoctError::SignalTooShort => {
                write!(f, "signal is too short for the required filter padding")
            }
            NoctError::SamplingFrequencyMismatch { implied_fs } => write!(
                f,
                "frequency axis implies fs = {implied_fs} Hz; noct_synthesis requires 48 kHz"
            ),
        }
    }
}

impl std::error::Error for NoctError {}

/// ANSI S1.1 nominal (preferred) 1/1-octave center frequencies, 31.5 Hz to 16 kHz.
pub const NOMINAL_OCTAVE_CENTER_FREQUENCIES: [f64; 10] = [
    31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
];

/// ANSI S1.1 nominal (preferred) 1/3-octave center frequencies, 25 Hz to 20 kHz.
pub const NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES: [f64; 30] = [
    25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
    630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
    10000.0, 12500.0, 16000.0, 20000.0,
];

/// Rounds to the nearest integer, ties to even — matching `numpy.round`, which
/// (unlike `f64::round`) breaks exact ties towards the even neighbour rather
/// than away from zero. The band-number calculation below is the one place
/// this could plausibly land on an exact tie.
fn round_half_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// Computes exact and nominal (preferred) band center frequencies, per ANSI
/// S1.1-1986 equations 1–4.
///
/// `g` selects the frequency ratio system: base 2 (`g == 2`) or base 10
/// (`g == 10`, ANSI's preferred system, and the only one MoSQITo's callers
/// use). `fr` is the reference frequency the band numbering is anchored to —
/// 1000 Hz for the audible range.
///
/// For `n` of 1 or 3, band numbers are also mapped onto the ANSI nominal
/// (rounded, "preferred") frequency table. MoSQITo's Python truncates bands
/// whose number falls outside the table (`_center_freq.py:71-73`), which can
/// leave the returned exact- and nominal-frequency arrays different lengths —
/// a latent bug, not something ANSI intends. This falls back to the exact
/// frequency for any band outside the table instead, so the two arrays this
/// returns are always the same length and always band-for-band aligned. Every
/// `fmin`/`fmax` pair used across MoSQITo's reference corpus stays within the
/// table (25 Hz–20 kHz for third-octave, 31.5 Hz–16 kHz for octave), so this
/// never actually changes a value there — see `DEVIATIONS.md`.
///
/// # Panics
/// Panics if `g` is neither 2 nor 10, or if `n` is 1 or 3 and `fr` is not
/// present in the corresponding nominal frequency table.
pub fn center_freq(fmin: f64, fmax: f64, n: u32, g: u32, fr: f64) -> (Vec<f64>, Vec<f64>) {
    let b = 1.0 / n as f64;
    let u = match g {
        2 => 2f64.powf(b),
        10 => 10f64.powf(3.0 * b / 10.0),
        _ => panic!("g must be 2 or 10, got {g}"),
    };
    let log_u = u.log10();
    let kmin = round_half_even((fmin / fr).log10() / log_u) as i64;
    let kmax = round_half_even((fmax / fr).log10() / log_u) as i64;
    let k: Vec<i64> = (kmin..=kmax).collect();

    let f_exact: Vec<f64> = k.iter().map(|&ki| fr * u.powi(ki as i32)).collect();

    let f_nom = if n == 1 || n == 3 {
        let table: &[f64] = if n == 1 {
            &NOMINAL_OCTAVE_CENTER_FREQUENCIES
        } else {
            &NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES
        };
        let i_ref = table
            .iter()
            .position(|&f| f == fr)
            .unwrap_or_else(|| panic!("reference frequency {fr} not in the nominal table"));
        let i_ref = i_ref as i64;
        k.iter()
            .zip(&f_exact)
            .map(|(&ki, &exact)| {
                if ki >= -i_ref && ki < table.len() as i64 - i_ref {
                    table[(ki + i_ref) as usize]
                } else {
                    exact
                }
            })
            .collect()
    } else {
        f_exact.clone()
    };

    (f_exact, f_nom)
}

/// Computes each band's bandwidth ratio `alpha` and edge frequencies, per ANSI
/// S1.1-1986 equations 5–9, for the fixed [`FILTER_ORDER`] of 3.
///
/// `alpha` is what `noct_spectrum`/`noct_synthesis` turn into normalised
/// bandpass cutoffs: `w1 = fc/(fs/2)/alpha`, `w2 = fc/(fs/2)*alpha`.
pub fn filter_bandwidth(fc: &[f64], n: u32) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let b = 1.0 / n as f64;
    let order = FILTER_ORDER as f64;
    let mut alpha = Vec::with_capacity(fc.len());
    let mut f1v = Vec::with_capacity(fc.len());
    let mut f2v = Vec::with_capacity(fc.len());
    for &fc_i in fc {
        let f1 = fc_i / 2f64.powf(b / 2.0);
        let f2 = fc_i * 2f64.powf(b / 2.0);
        let qr = fc_i / (f2 - f1);
        let qd = (PI / 2.0 / order) / (PI / 2.0 / order).sin() * qr;
        let a = (1.0 + (1.0 + 4.0 * qd * qd).sqrt()) / (2.0 * qd);
        alpha.push(a);
        f1v.push(f1);
        f2v.push(f2);
    }
    (alpha, f1v, f2v)
}

/// A band's time-domain filter design (decimation factor + bandpass SOS),
/// shared across every segment `noct_spectrum` filters in that band — the
/// design depends only on `fs`/`fc`/`alpha`, none of which vary by segment.
struct TimeFilterDesign {
    /// Decimation factor applied before filtering (1 = no decimation).
    q: usize,
    sos: Vec<Sos>,
}

/// Designs one band's time-domain filter. Mirrors `_n_oct_time_filter`:
/// decimates first when `fc` is far below `fs`, to keep the bandpass design
/// well conditioned.
fn time_filter_design(fs: f64, fc: f64, alpha: f64) -> Result<TimeFilterDesign, NoctError> {
    if fc > 0.88 * (fs / 2.0) {
        return Err(NoctError::CenterFrequencyTooHigh {
            fc,
            nyquist: fs / 2.0,
        });
    }

    let mut q = 1usize;
    let mut fs_eff = fs;
    if fc < fs / 200.0 {
        q = 2usize;
        while fc < fs / q as f64 / 200.0 {
            q += 1;
        }
        fs_eff = fs / q as f64;
    }

    let w1 = fc / (fs_eff / 2.0) / alpha;
    let w2 = fc / (fs_eff / 2.0) * alpha;
    Ok(TimeFilterDesign {
        q,
        sos: butter_bandpass_sos(FILTER_ORDER, w1, w2),
    })
}

/// RMS level of one channel in the band `design` was built for.
fn n_oct_time_filter_column(
    sig: ArrayView1<f64>,
    design: &TimeFilterDesign,
) -> Result<f64, NoctError> {
    let signal = if design.q > 1 {
        decimate(&sig.to_vec(), design.q).ok_or(NoctError::SignalTooShort)?
    } else {
        sig.to_vec()
    };

    let filtered = sosfilt(&design.sos, &signal);
    let rms = (filtered.iter().map(|v| v * v).sum::<f64>() / filtered.len() as f64).sqrt();
    Ok(rms)
}

/// RMS level in the band centred at `fc`, evaluated against an existing
/// spectrum via the filter's frequency response. Mirrors `_n_oct_freq_filter`.
fn n_oct_freq_filter(spectrum: &[f64], fs: f64, fc: f64, alpha: f64) -> f64 {
    let w1 = fc / (fs / 2.0) / alpha;
    let w2 = fc / (fs / 2.0) * alpha;
    let sos = butter_bandpass_sos(FILTER_ORDER, w1, w2);
    let h = sosfreqz(&sos, spectrum.len());
    let sum_sq: f64 = h
        .iter()
        .zip(spectrum)
        .map(|(hi, &s)| (hi * s).norm_sqr())
        .sum();
    sum_sq.sqrt()
}

/// Measures the RMS level of `sig` in each n-th octave band between `fmin` and
/// `fmax`, by time-domain filtering. Matches `noct_spectrum(sig, fs, fmin,
/// fmax, n, G, fr)`.
///
/// `sig` is (samples, segments): axis 0 is time, axis 1 is one column per
/// segment. Returns `(spec, freq)` where `spec` is (bands, segments) and
/// `freq` holds each band's nominal (preferred) center frequency.
///
/// Runs the bands in parallel; each band filters its segments sequentially.
pub fn noct_spectrum(
    sig: ArrayView2<f64>,
    fs: f64,
    fmin: f64,
    fmax: f64,
    n: u32,
    g: u32,
    fr: f64,
) -> Result<(Array2<f64>, Vec<f64>), NoctError> {
    let (fc_exact, f_nom) = center_freq(fmin, fmax, n, g, fr);
    let (alpha, _f1, _f2) = filter_bandwidth(&fc_exact, n);
    let nseg = sig.ncols();
    let nbands = fc_exact.len();

    let rows: Vec<Vec<f64>> = fc_exact
        .par_iter()
        .zip(alpha.par_iter())
        .map(|(&fc, &al)| -> Result<Vec<f64>, NoctError> {
            let design = time_filter_design(fs, fc, al)?;
            (0..nseg)
                .map(|c| n_oct_time_filter_column(sig.column(c), &design))
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut spec = Array2::<f64>::zeros((nbands, nseg));
    for (i, row) in rows.into_iter().enumerate() {
        for (j, v) in row.into_iter().enumerate() {
            spec[[i, j]] = v;
        }
    }
    Ok((spec, f_nom))
}

/// Converts a frequency spectrum to n-th octave band levels between `fmin`
/// and `fmax`, by evaluating each band filter's frequency response. Matches
/// `noct_synthesis(spectrum, freqs, fmin, fmax, n, G, fr)` for a 1-D
/// spectrum.
///
/// `freqs` must imply a 48 kHz sampling rate (`max(freqs) * 2 ≈ 48000`), as in
/// MoSQITo — this function exists to feed ISO 532-1 loudness, which is defined
/// only at 48 kHz. Bands whose upper edge would exceed the Nyquist frequency
/// are dropped, as `noct_synthesis.py:88-93` does.
pub fn noct_synthesis(
    spectrum: &[f64],
    freqs: &[f64],
    fmin: f64,
    fmax: f64,
    n: u32,
    g: u32,
    fr: f64,
) -> Result<(Vec<f64>, Vec<f64>), NoctError> {
    let fs = freqs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 2.0;
    if (fs.round() - 48000.0).abs() > 0.5 {
        return Err(NoctError::SamplingFrequencyMismatch { implied_fs: fs });
    }

    let (fc_exact, f_nom) = center_freq(fmin, fmax, n, g, fr);
    let (alpha, _f1, f_high) = filter_bandwidth(&fc_exact, n);

    let bands: Vec<(f64, f64, f64)> = fc_exact
        .into_iter()
        .zip(alpha)
        .zip(f_high)
        .zip(f_nom)
        .filter(|&(((_, _), fh), _)| fh <= fs / 2.0)
        .map(|(((fc, al), _fh), nom)| (fc, al, nom))
        .collect();

    let levels: Vec<f64> = bands
        .par_iter()
        .map(|&(fc, al, _)| n_oct_freq_filter(spectrum, fs, fc, al))
        .collect();
    let freq_out: Vec<f64> = bands.iter().map(|&(_, _, nom)| nom).collect();

    Ok((levels, freq_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn round_half_even_matches_numpy_convention() {
        assert_relative_eq!(round_half_even(2.5), 2.0);
        assert_relative_eq!(round_half_even(3.5), 4.0);
        assert_relative_eq!(round_half_even(-2.5), -2.0);
        assert_relative_eq!(round_half_even(2.4), 2.0);
        assert_relative_eq!(round_half_even(2.6), 3.0);
    }

    #[test]
    fn center_freq_third_octave_spans_the_standard_range() {
        // fmin/fmax exactly on the table: band numbers should map onto
        // 25 Hz and 20000 Hz without needing the fallback.
        let (exact, nom) = center_freq(25.0, 20000.0, 3, 10, 1000.0);
        assert_eq!(exact.len(), nom.len());
        assert_relative_eq!(nom[0], 25.0, epsilon = 1e-9);
        assert_relative_eq!(nom[nom.len() - 1], 20000.0, epsilon = 1e-9);
        assert_relative_eq!(exact[0], 25.0, max_relative = 0.01);
    }

    #[test]
    fn center_freq_reference_frequency_maps_to_itself() {
        let (exact, nom) = center_freq(500.0, 2000.0, 3, 10, 1000.0);
        let idx = exact
            .iter()
            .position(|&f| (f - 1000.0).abs() < 1e-6)
            .expect("1 kHz band present");
        assert_relative_eq!(nom[idx], 1000.0, epsilon = 1e-9);
    }

    #[test]
    fn center_freq_octave_uses_the_octave_table() {
        // fmin/fmax exactly on the octave table's own bounds (31.5 Hz to
        // 16 kHz): every band number stays in range, so every nominal
        // frequency comes from the table itself rather than the
        // below-the-table fallback (which is exercised separately below).
        let (_exact, nom) = center_freq(31.5, 16000.0, 1, 10, 1000.0);
        for f in &nom {
            assert!(
                NOMINAL_OCTAVE_CENTER_FREQUENCIES.contains(f),
                "{f} should be one of the nominal octave frequencies"
            );
        }
    }

    #[test]
    fn center_freq_falls_back_to_the_exact_frequency_below_the_table() {
        // fmin below the octave table's 31.5 Hz floor: the lowest band number
        // has no entry in NOMINAL_OCTAVE_CENTER_FREQUENCIES, so it must fall
        // back to the exact (unrounded) frequency rather than panicking or
        // silently dropping the band.
        let (exact, nom) = center_freq(20.0, 16000.0, 1, 10, 1000.0);
        assert_eq!(
            exact.len(),
            nom.len(),
            "fallback must keep the arrays aligned"
        );
        assert_relative_eq!(nom[0], exact[0], epsilon = 1e-9);
        assert!(
            !NOMINAL_OCTAVE_CENTER_FREQUENCIES.contains(&nom[0]),
            "the lowest band should be below every table entry"
        );
    }

    #[test]
    fn filter_bandwidth_alpha_exceeds_one() {
        // alpha is a ratio of upper/lower edge to center; it must exceed 1 for
        // a band with positive width.
        let (alpha, f1, f2) = filter_bandwidth(&[1000.0], 3);
        assert!(alpha[0] > 1.0);
        assert!(f1[0] < 1000.0 && 1000.0 < f2[0]);
    }

    #[test]
    fn noct_spectrum_isolates_a_pure_tone_in_its_band() {
        let fs = 48000.0;
        let n = 48000;
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / fs).sin())
            .collect();
        let sig2d = Array2::from_shape_vec((n, 1), sig).unwrap();

        let (spec, freq) =
            noct_spectrum(sig2d.view(), fs, 100.0, 10000.0, 3, 10, 1000.0).expect("valid design");
        let idx = freq
            .iter()
            .position(|&f| f == 1000.0)
            .expect("1 kHz band present");

        // A unit-amplitude sine has RMS 1/sqrt(2); every other band should be
        // far quieter.
        assert_relative_eq!(
            spec[[idx, 0]],
            std::f64::consts::FRAC_1_SQRT_2,
            max_relative = 0.05
        );
        for (i, &f) in freq.iter().enumerate() {
            if (f - 1000.0).abs() > 1.0 {
                assert!(
                    spec[[i, 0]] < 0.1,
                    "band at {f} Hz leaked energy: {}",
                    spec[[i, 0]]
                );
            }
        }
    }

    #[test]
    fn noct_spectrum_processes_every_segment_independently() {
        let fs = 48000.0;
        let n = 4096;
        let seg1: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / fs).sin())
            .collect();
        let seg2: Vec<f64> = vec![0.0; n];
        let mut sig2d = Array2::<f64>::zeros((n, 2));
        sig2d.column_mut(0).assign(&Array1::from(seg1));
        sig2d.column_mut(1).assign(&Array1::from(seg2));

        let (spec, freq) =
            noct_spectrum(sig2d.view(), fs, 500.0, 2000.0, 3, 10, 1000.0).expect("valid design");
        let idx = freq
            .iter()
            .position(|&f| f == 1000.0)
            .expect("1 kHz band present");
        assert!(spec[[idx, 0]] > 0.1, "segment 0 should carry the tone");
        assert_relative_eq!(spec[[idx, 1]], 0.0, epsilon = 1e-9);
    }

    #[test]
    fn noct_spectrum_rejects_a_center_frequency_above_the_nyquist_limit() {
        let sig2d = Array2::<f64>::zeros((4096, 1));
        let err = noct_spectrum(sig2d.view(), 8000.0, 3000.0, 3600.0, 3, 10, 1000.0).unwrap_err();
        assert!(matches!(err, NoctError::CenterFrequencyTooHigh { .. }));
    }

    #[test]
    fn noct_synthesis_rejects_a_non_48k_frequency_axis() {
        let freqs = [0.0, 1000.0, 2000.0];
        let spectrum = [0.0, 1.0, 0.0];
        let err = noct_synthesis(&spectrum, &freqs, 24.0, 1000.0, 3, 10, 1000.0).unwrap_err();
        assert!(matches!(err, NoctError::SamplingFrequencyMismatch { .. }));
    }

    #[test]
    fn noct_synthesis_time_and_frequency_domain_agree_on_broadband_noise() {
        // Measure a signal with noct_spectrum (time domain), then measure its
        // FFT-magnitude spectrum with noct_synthesis (frequency domain). The
        // two must agree closely — this self-consistency is the actual
        // conformance gate MoSQITo tests with (test_noct_synthesis_technical).
        //
        // The stimulus must be broadband, not a pure tone. noct_spectrum
        // decimates before filtering when fc is low (fc < fs/200) so the
        // bandpass design stays well-conditioned, and noct_synthesis never
        // decimates — so for a low fc the two paths design genuinely
        // different filters, sharing only the same nominal Hz passband. Where
        // a band holds real signal, both filters recover the same energy;
        // where it holds none, each path is measuring its own filter's
        // stopband leakage, and two different filters leak differently. A
        // pure 1 kHz tone left every other band comparing leakage, not
        // signal, and failed by tens of dB. MoSQITo's own gate uses a
        // pink-noise recording for the same reason; a small xorshift64 PRNG
        // stands in for noise here so the test has no added dependency.
        // n also needs to be long enough that `freqs.max() * 2` rounds to
        // 48000: keeping only the first n/2 FFT bins leaves the last one
        // fs/n short of the true Nyquist frequency, and noct_synthesis's
        // 48 kHz check needs that shortfall under 0.5 Hz. MoSQITo's own
        // reference wav is ~10 s for the same reason.
        let fs = 48000.0;
        let n = 240_000;
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let sig: Vec<f64> = (0..n).map(|_| 0.1 * noise()).collect();
        let sig2d = Array2::from_shape_vec((n, 1), sig.clone()).unwrap();

        let (spec_t, freq_t) =
            noct_spectrum(sig2d.view(), fs, 100.0, 10000.0, 3, 10, 1000.0).expect("valid design");

        // Real one-sided FFT magnitude spectrum, matching MoSQITo's test
        // convention: 2/sqrt(2)/n * fft(sig)[:n/2].
        let mut planner = realfft::RealFftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(n);
        let mut spectrum_c = fwd.make_output_vec();
        let mut input = sig;
        fwd.process(&mut input, &mut spectrum_c).unwrap();
        let scale = 2.0 / std::f64::consts::SQRT_2 / n as f64;
        let magnitude: Vec<f64> = spectrum_c[..n / 2]
            .iter()
            .map(|c| c.norm() * scale)
            .collect();
        let freqs: Vec<f64> = (0..n / 2).map(|k| k as f64 * fs / n as f64).collect();

        let (spec_f, freq_f) = noct_synthesis(&magnitude, &freqs, 100.0, 10000.0, 3, 10, 1000.0)
            .expect("valid design");

        assert_eq!(freq_t.len(), freq_f.len());
        for (i, (&ft, &ff)) in freq_t.iter().zip(&freq_f).enumerate() {
            assert_relative_eq!(ft, ff, epsilon = 1e-9, max_relative = 1e-9);
            let db_t = 20.0 * (spec_t[[i, 0]] / 2e-5).log10();
            let db_f = 20.0 * (spec_f[i] / 2e-5).log10();
            assert!(
                (db_t - db_f).abs() < 0.3,
                "band {ft} Hz: time {db_t:.3} dB vs freq {db_f:.3} dB"
            );
        }
    }
}
