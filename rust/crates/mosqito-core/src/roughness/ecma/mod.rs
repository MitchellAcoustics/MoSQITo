//! ECMA-418-2:2022 (2nd Ed) §7 roughness (the Sottek Hearing Model's
//! roughness stage), implementing the standard with the corrections from
//! Wanty, Glesser & Casagrande Hirono, *"ECMA-418-2 roughness, a
//! challenging implementation"*, INTERNOISE 2024 — see `DEVIATIONS.md` for
//! the full deviations register (D-ecma-1 through D-ecma-7).

mod envelope_spectrum;
mod estimate_fund_mod_rate;
mod lowpass_filter;
mod noise_reduction;
mod non_linear_transform;
mod peak_picking;
mod refinement;
mod roughness_ecma;
mod von_hann_window;
mod weighting;

pub use estimate_fund_mod_rate::estimate_fund_mod_rate;
pub use lowpass_filter::lowpass_filter;
pub use noise_reduction::noise_reduction;
pub use non_linear_transform::non_linear_transform;
pub use peak_picking::peak_picking;
pub use refinement::refinement;
pub use roughness_ecma::roughness_ecma;
pub use von_hann_window::von_hann_window;
pub use weighting::{
    f_max, high_mod_rate_weighting, low_mod_rate_weighting, q2_high, q2_low, r_max,
};
