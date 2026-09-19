//! Tonality: tone-to-noise ratio (TNR) and prominence ratio (PR), per
//! ECMA-74 Annex D with the T-TNR/T-PR totals from ECMA TR/108.
//!
//! `tone_to_noise_ecma` and `prominence_ratio_ecma` share almost all of
//! their machinery (critical-band-edge formulas, the threshold of hearing,
//! spectrum smoothing, tonal-candidate screening, and within-band
//! tie-breaking) — ported once here, under one module, with [`tnr`] and
//! [`pr`] as thin orchestration on top, mirroring how `sharpness::din`'s
//! four weightings share one integral and differ only in formula.

mod critical_band;
mod entry_points;
mod find_highest_tone;
mod get_frequencies;
mod lth;
mod peak_level;
mod pr;
mod screening;
mod spectrum_smoothing;
mod tnr;

pub use critical_band::{critical_band, lower_critical_band, upper_critical_band};
pub use entry_points::{
    pr_ecma_freq, pr_ecma_perseg, pr_ecma_st, tnr_ecma_freq, tnr_ecma_perseg, tnr_ecma_st,
    PersegGrid,
};
pub use find_highest_tone::find_highest_tone;
pub use get_frequencies::{get_frequencies, BandEdges};
pub use lth::lth;
pub use peak_level::peak_level;
pub use pr::{pr_main_calc, PrResult};
pub use screening::screening_for_tones;
pub use spectrum_smoothing::spectrum_smoothing;
pub use tnr::{tnr_main_calc, TnrResult};
