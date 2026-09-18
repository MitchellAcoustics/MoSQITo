//! Sound level meter primitives: n-th octave band analysis.

pub mod noct;

pub use noct::{
    center_freq, filter_bandwidth, noct_spectrum, noct_synthesis, NoctError,
    NOMINAL_OCTAVE_CENTER_FREQUENCIES, NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES,
};
