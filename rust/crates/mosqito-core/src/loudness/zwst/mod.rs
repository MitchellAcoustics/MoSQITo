//! Loudness for stationary signals, per ISO 532-1:2017 (the Zwicker method).

pub mod calc_slopes;
pub mod loudness_zwst;
pub mod main_loudness;

pub use calc_slopes::calc_slopes;
pub use loudness_zwst::{
    bark_axis, loudness_zwst, loudness_zwst_freq, loudness_zwst_perseg, LoudnessZwstError,
};
pub use main_loudness::{main_loudness, FieldType};
