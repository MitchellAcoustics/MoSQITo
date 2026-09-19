//! Signal-processing primitives matching SciPy's semantics.
//!
//! The metrics in this crate are validated against reference values published
//! with their standards, but the reference corpus and the Python implementation
//! that produced it both go through SciPy. Where a standard leaves a processing
//! step unspecified — the anti-alias filter inside a decimation, say — matching
//! SciPy is what makes the published reference values reachable. Each submodule
//! documents the specific conventions it reproduces.

pub mod design;
pub mod fft;
pub mod filter;
pub mod interp;
pub mod peaks;
pub mod stats;
pub mod windows;

pub use design::{butter_bandpass_sos, butter_lowpass_sos, cheby1_lowpass, sosfreqz, Sos};
pub use fft::{hilbert_envelope, resample, resample_to, resample_up_to};
pub use filter::{decimate, filtfilt, lfilter, lfilter_complex, sosfilt, sosfiltfilt};
pub use interp::{interp, interp_zero_fill, nearest_index, pchip};
pub use peaks::{find_peaks_with_prominence, Peak};
pub use stats::{median, percentile_linear};
pub use windows::{blackman, hanning};
