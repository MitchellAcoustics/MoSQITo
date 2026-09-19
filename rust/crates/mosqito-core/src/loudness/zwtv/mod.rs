//! ISO 532-1:2017 time-varying loudness (Zwicker method).

mod loudness_zwtv;
mod lowpass_intp;
mod nonlinear_decay;
mod square_and_smooth;
mod tables;
mod temporal_weighting;
mod third_octave_levels;

pub use loudness_zwtv::{loudness_zwtv, LoudnessZwtvError};
pub use nonlinear_decay::nl_loudness;
pub use third_octave_levels::{third_octave_levels, ThirdOctaveLevelsError};
