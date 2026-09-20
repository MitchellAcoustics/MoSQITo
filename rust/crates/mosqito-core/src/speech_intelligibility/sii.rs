//! ANSI S3.5-1997 Speech Intelligibility Index (SII), matching
//! `mosqito.sq_metrics.speech_intelligibility.sii_ansi`.

use ndarray::Array2;

use super::band_data::{band_data, speech_spectrum, SiiMethod, SpeechLevel};
use crate::slm::{comp_spectrum_db, freq_band_synthesis, SpectrumWindow};
use crate::utils::{freq2bark, ltq, LtqReference};

/// The hearing threshold to subtract in SII's step 4, matching `_main_sii`'s
/// `threshold` argument (`None`/`'zwicker'`/an explicit array).
///
/// # A Python bug this variant fixes, not reproduces
/// `_main_sii.py` dispatches on `threshold` with `elif threshold ==
/// "zwicker":` — for an array threshold, comparing it to a string raises
/// `ValueError: The truth value of an array with more than one element is
/// ambiguous`, so real MoSQITo cannot actually accept an explicit array
/// threshold at all despite documenting the parameter as `array_like or
/// 'zwicker'` (confirmed directly: `_main_sii(method, speech, noise,
/// threshold=some_array)` raises in the installed package, not just through
/// the `sii_ansi*` wrappers). `Custom` here works as documented instead of
/// reproducing that crash — there is no ambiguity to preserve, since every
/// non-crashing code path in the standard is unaffected. See `DEVIATIONS.md`.
pub enum SiiThreshold<'a> {
    /// `threshold=None`: zero on every band.
    Zero,
    /// `threshold='zwicker'`: `LTQ(freq2bark(CENTER_FREQUENCIES))`.
    Zwicker,
    /// An explicit per-band threshold, one value per band.
    Custom(&'a [f64]),
}

/// `(SII, SII_specific, freq_axis)` — every SII entry point's result.
type SiiResult = (f64, Vec<f64>, Vec<f64>);

/// Core SII computation from a speech and noise spectrum, matching
/// `_main_sii(method, speech_spectrum, noise_spectrum, threshold)`.
///
/// `speech_spectrum`/`noise_spectrum` are dB re. 2e-5 Pa, one value per band
/// of `method`'s procedure.
///
/// # A confirmed dead branch, not reproduced
/// Python's `_main_sii.py` guards a per-band bandwidth adjustment
/// (`noise_spectrum -= 10*log10(upper - lower)`) behind
/// `method in {"critical_bands", "equal_critical_bands"}` — but the only
/// method strings any caller can ever pass are `"critical"`/
/// `"equally_critical"` (checked and rejected earlier in the same call
/// chain), so this branch is unreachable and the adjustment never fires.
/// Neither of MoSQITo's own validated reference cases
/// (`validations/sq_metrics/speech_intelligibility/validation_sii.py`) uses
/// the critical-band procedures, so there is no corpus here to confirm
/// whether ANSI S3.5 actually requires this adjustment for critical bands;
/// not ported, matching the dead code as written. See `DEVIATIONS.md`.
///
/// # A reproduced asymmetry between band procedures
/// The cumulative-masking sum (§4.3.2.4) runs `k in 0..i` for the
/// third-octave procedure but `k in 0..(i-1)` (one term short) for the
/// critical/equally-critical procedures — an asymmetry present in the
/// Python as written. Reproduced verbatim for the same reason: no validated
/// corpus exercises the critical-band procedures to confirm which is
/// intended.
pub fn main_sii(
    method: SiiMethod,
    speech_spectrum_in: &[f64],
    noise_spectrum: &[f64],
    threshold: SiiThreshold,
) -> SiiResult {
    let data = band_data(method);
    let nbands = data.center.len();

    let t: Vec<f64> = match threshold {
        SiiThreshold::Zero => vec![0.0; nbands],
        SiiThreshold::Zwicker => {
            let bark = freq2bark(data.center);
            ltq(&bark, LtqReference::Zwicker)
        }
        SiiThreshold::Custom(arr) => {
            // Without this, a short array would silently truncate every
            // `zip` below it and drop trailing bands from the SII sum — a
            // plausible-looking but wrong answer rather than an error.
            assert_eq!(
                arr.len(),
                nbands,
                "a custom threshold needs one value per band ({nbands} for this method)"
            );
            arr.to_vec()
        }
    };

    let z: Vec<f64> = if matches!(method, SiiMethod::Octave) {
        noise_spectrum.to_vec()
    } else {
        let b: Vec<f64> = noise_spectrum
            .iter()
            .zip(speech_spectrum_in)
            .map(|(&n, &s)| n.max(s - 24.0))
            .collect();

        let mut z = vec![0.0f64; nbands];
        if matches!(method, SiiMethod::ThirdOctave) {
            let c: Vec<f64> = (0..nbands)
                .map(|k| -80.0 + 0.6 * (b[k] + 10.0 * data.center[k].log10() - 6.353))
                .collect();
            for i in 0..nbands {
                let s: f64 = (0..i)
                    .map(|k| {
                        10f64.powf(
                            0.1 * (b[k]
                                + 3.32 * c[k] * (0.89 * data.center[i] / data.center[k]).log10()),
                        )
                    })
                    .sum();
                z[i] = 10.0 * (10f64.powf(0.1 * noise_spectrum[i]) + s).log10();
            }
        } else {
            let c: Vec<f64> = (0..nbands)
                .map(|k| -80.0 + 0.6 * (b[k] + 10.0 * (data.upper[k] - data.lower[k]).log10()))
                .collect();
            for i in 0..nbands {
                // Python: `for k in range(i - 1)`, reproduced verbatim — see
                // this function's doc comment.
                let end = i.saturating_sub(1);
                let s: f64 = (0..end)
                    .map(|k| {
                        10f64.powf(
                            0.1 * (b[k] + 3.32 * c[k] * (data.center[i] / data.center[k]).log10()),
                        )
                    })
                    .sum();
                z[i] = 10.0 * (10f64.powf(0.1 * noise_spectrum[i]) + s).log10();
            }
        }
        z[0] = b[0];
        z
    };

    let x: Vec<f64> = data
        .reference_internal_noise
        .iter()
        .zip(&t)
        .map(|(&r, &tt)| r + tt)
        .collect();
    let d: Vec<f64> = z.iter().zip(&x).map(|(&zz, &xx)| zz.max(xx)).collect();

    let standard_normal = speech_spectrum(method, SpeechLevel::Normal);
    let l: Vec<f64> = speech_spectrum_in
        .iter()
        .zip(standard_normal)
        .map(|(&sp, &std)| (1.0 - (sp - std - 10.0) / 160.0).min(1.0))
        .collect();
    let k: Vec<f64> = speech_spectrum_in
        .iter()
        .zip(&d)
        .map(|(&sp, &dd)| ((sp - dd + 15.0) / 30.0).clamp(0.0, 1.0))
        .collect();
    let a: Vec<f64> = l.iter().zip(&k).map(|(&ll, &kk)| ll * kk).collect();

    let sii_specific: Vec<f64> = data
        .importance
        .iter()
        .zip(&a)
        .map(|(&imp, &aa)| imp * aa)
        .collect();
    let sii: f64 = sii_specific.iter().sum();

    (sii, sii_specific, data.center.to_vec())
}

/// SII from a noise time signal, matching `sii_ansi(noise, fs, method,
/// speech_level, threshold)`.
pub fn sii_ansi(
    noise: &[f64],
    fs: f64,
    method: SiiMethod,
    speech_level: SpeechLevel,
    threshold: SiiThreshold,
) -> SiiResult {
    let data = band_data(method);
    let speech = speech_spectrum(method, speech_level);

    let n = noise.len();
    let sig2d = Array2::from_shape_vec((n, 1), noise.to_vec()).expect("noise is non-empty");
    let (spec_db, freqs) = comp_spectrum_db(sig2d.view(), fs, SpectrumWindow::Blackman);
    let spec_db_col = spec_db.column(0).to_vec();
    let (noise_spectrum, _) = freq_band_synthesis(&spec_db_col, &freqs, data.lower, data.upper);

    main_sii(method, speech, &noise_spectrum, threshold)
}

/// SII from a noise spectrum in dB, matching `sii_ansi_freq(spectrum, freqs,
/// method, speech_level, threshold)`.
pub fn sii_ansi_freq(
    spectrum: &[f64],
    freqs: &[f64],
    method: SiiMethod,
    speech_level: SpeechLevel,
    threshold: SiiThreshold,
) -> SiiResult {
    let data = band_data(method);
    let speech = speech_spectrum(method, speech_level);
    let nbands = speech.len();

    let noise_spectrum: Vec<f64> = if spectrum.len() != nbands || freqs != data.center {
        let (levels, _) = freq_band_synthesis(spectrum, freqs, data.lower, data.upper);
        levels
    } else {
        spectrum.to_vec()
    };

    main_sii(method, speech, &noise_spectrum, threshold)
}

/// SII from an overall noise SPL, spread uniformly across every band,
/// matching `sii_ansi_level(noise_level, method, speech_level, threshold)`.
pub fn sii_ansi_level(
    noise_level: f64,
    method: SiiMethod,
    speech_level: SpeechLevel,
    threshold: SiiThreshold,
) -> SiiResult {
    let speech = speech_spectrum(method, speech_level);
    let nbands = speech.len();
    let band_level = 10.0 * (10f64.powf(noise_level / 10.0) / nbands as f64).log10();
    let noise_spectrum = vec![band_level; nbands];

    main_sii(method, speech, &noise_spectrum, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "one value per band")]
    fn a_custom_threshold_of_the_wrong_length_is_rejected() {
        // Silently zipping a short threshold would drop trailing bands from
        // the SII sum and return a plausible but wrong number.
        let speech = [50.0, 40.0, 40.0, 30.0, 20.0, 0.0];
        let noise = [70.0, 65.0, 45.0, 25.0, 1.0, -15.0];
        let short = [0.0; 3];
        main_sii(
            SiiMethod::Octave,
            &speech,
            &noise,
            SiiThreshold::Custom(&short),
        );
    }

    #[test]
    fn custom_threshold_matches_zero_threshold_when_all_zero() {
        let speech = [50.0, 40.0, 40.0, 30.0, 20.0, 0.0];
        let noise = [70.0, 65.0, 45.0, 25.0, 1.0, -15.0];
        let zeros = [0.0; 6];

        let (sii_zero, spec_zero, _) =
            main_sii(SiiMethod::Octave, &speech, &noise, SiiThreshold::Zero);
        let (sii_custom, spec_custom, _) = main_sii(
            SiiMethod::Octave,
            &speech,
            &noise,
            SiiThreshold::Custom(&zeros),
        );

        assert_eq!(sii_zero, sii_custom);
        assert_eq!(spec_zero, spec_custom);
    }

    #[test]
    fn custom_threshold_matches_zwicker_when_set_to_the_same_curve() {
        let speech = [50.0, 40.0, 40.0, 30.0, 20.0, 0.0];
        let noise = [70.0, 65.0, 45.0, 25.0, 1.0, -15.0];
        let center = band_data(SiiMethod::Octave).center;
        let bark = freq2bark(center);
        let zwicker_curve = ltq(&bark, LtqReference::Zwicker);

        let (sii_zwicker, _, _) =
            main_sii(SiiMethod::Octave, &speech, &noise, SiiThreshold::Zwicker);
        let (sii_custom, _, _) = main_sii(
            SiiMethod::Octave,
            &speech,
            &noise,
            SiiThreshold::Custom(&zwicker_curve),
        );

        assert_eq!(sii_zwicker, sii_custom);
    }

    #[test]
    fn a_higher_custom_threshold_never_increases_sii() {
        // Raising the hearing threshold can only make speech-in-noise harder
        // to perceive, never easier.
        let speech = [50.0, 40.0, 40.0, 30.0, 20.0, 0.0];
        let noise = [70.0, 65.0, 45.0, 25.0, 1.0, -15.0];
        let low = [-10.0; 6];
        let high = [30.0; 6];

        let (sii_low, _, _) = main_sii(
            SiiMethod::Octave,
            &speech,
            &noise,
            SiiThreshold::Custom(&low),
        );
        let (sii_high, _, _) = main_sii(
            SiiMethod::Octave,
            &speech,
            &noise,
            SiiThreshold::Custom(&high),
        );

        assert!(sii_high <= sii_low);
    }
}
