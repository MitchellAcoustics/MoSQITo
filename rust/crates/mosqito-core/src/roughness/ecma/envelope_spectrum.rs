//! ECMA-418-2 (2nd Ed, 2022) §7.1.2-7.1.3: envelope extraction, downsampling
//! to 1500 Hz, and the scaled power spectrum, for one critical band. Matches
//! the relevant part of `roughness_ecma.py:125-149`.

use ndarray::Array2;
use realfft::RealFftPlanner;

use super::von_hann_window::von_hann_window;
use crate::dsp::{decimate, hilbert_envelope};
use crate::loudness::ecma::{block_sample_index, block_step, n_blocks};

/// Downsampled envelope block length (`sbb` in the standard): 16384 / 32.
pub const SBB: usize = 512;
/// Number of retained (one-sided) spectrum bins: `SBB / 2`.
pub const N_BINS: usize = SBB / 2;

/// One band's per-block DFT (`Phi_E0`-scaled power spectrum, before the
/// cross-band loudness scaling in `roughness_ecma.rs`) and normalisation
/// energy.
pub struct BandSpectrum {
    /// `(n_blocks, N_BINS)`: the raw scaled power spectrum, Eq. 85's `Phi`
    /// before the `N_specific`-dependent scaling factor.
    pub dft: Array2<f64>,
    /// `(n_blocks,)`: Eq. 85's denominator term `Phi_E0`.
    pub phi_e0: Vec<f64>,
}

/// Computes one band's envelope power spectrum from its (already
/// gammatone-filtered) band-pass signal.
///
/// Per block: gathers the `sb`-sample block (the same index formula
/// `loudness::ecma` uses), extracts its envelope via the Hilbert transform,
/// downsamples 32x (as `8*4`, matching `roughness_ecma.py:133-134`'s split)
/// to `SBB` = 512 samples, applies the ECMA Von Hann window, and computes
/// its scaled power spectrum.
pub fn band_spectrum(band_pass_signal: &[f64], sb: usize, sh: usize, n_new: usize) -> BandSpectrum {
    let blocks = n_blocks(n_new, sh);
    let step = block_step(sb);
    let hann = von_hann_window(SBB);

    let mut dft = Array2::<f64>::zeros((blocks, N_BINS));
    let mut phi_e0 = vec![0.0f64; blocks];

    let mut fft_planner = RealFftPlanner::<f64>::new();
    let fft = fft_planner.plan_fft_forward(SBB);

    for l in 0..blocks {
        let block: Vec<f64> = (0..sb)
            .map(|k| band_pass_signal[block_sample_index(l, k, sh, step)])
            .collect();

        let envelope = hilbert_envelope(&block);
        let downsampled_8 = decimate(&envelope, 8).expect("valid decimation factor");
        let downsampled = decimate(&downsampled_8, 4).expect("valid decimation factor");
        debug_assert_eq!(downsampled.len(), SBB);

        let mut windowed: Vec<f64> = downsampled
            .iter()
            .zip(&hann)
            .map(|(&e, &w)| e * w)
            .collect();

        phi_e0[l] = windowed.iter().map(|&v| v * v).sum();

        let mut spectrum = fft.make_output_vec();
        fft.process(&mut windowed, &mut spectrum)
            .expect("fixed-size FFT plan matches input length");

        for k in 0..N_BINS {
            let mag = spectrum[k].norm();
            dft[[l, k]] = (mag / 2.0 * std::f64::consts::SQRT_2).powi(2);
        }
    }

    BandSpectrum { dft, phi_e0 }
}
