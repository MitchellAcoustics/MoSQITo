//! Sound level meter primitives: n-th octave band analysis, windowed FFT
//! spectra, and frequency-band energy synthesis.

pub mod comp_spectrum;
pub mod freq_band_synthesis;
pub mod noct;

pub use comp_spectrum::{comp_spectrum_complex, comp_spectrum_db, SpectrumWindow};
pub use freq_band_synthesis::freq_band_synthesis;
pub use noct::{
    center_freq, filter_bandwidth, noct_spectrum, noct_synthesis, NoctError,
    NOMINAL_OCTAVE_CENTER_FREQUENCIES, NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
};
