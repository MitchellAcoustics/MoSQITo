//! Standards-conformant psychoacoustic sound quality metrics.
//!
//! This crate is a Rust implementation of the metrics provided by the
//! [MoSQITo](https://github.com/Eomys/MoSQITo) Python toolbox. It targets the
//! published standards directly — ISO 532-1, ECMA-418-2 (2nd edition, 2022) and
//! DIN 45692 — rather than bit-parity with the Python implementation. Every
//! deliberate divergence is recorded in `DEVIATIONS.md` at the repository root.
//!
//! The [`dsp`] module provides the signal-processing primitives the metrics are
//! built from. Its semantics deliberately follow SciPy's, because the reference
//! implementation and its validation corpus were produced with SciPy: see the
//! module documentation for the specific conventions that are matched.

pub mod dsp;

/// Re-exported so callers and tests share this crate's complex number type.
pub use num_complex;

/// Sampling rate required by ISO 532-1 and ECMA-418-2, in Hz.
pub const FS_STANDARD: f64 = 48000.0;
