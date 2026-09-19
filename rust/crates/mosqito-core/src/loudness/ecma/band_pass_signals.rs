//! ECMA-418-2 (2nd Ed, 2022) §5.1.3: outer/middle ear filtering followed by
//! the 53-band complex gammatone auditory filter bank. Matches the (live,
//! non-dead-code) half of `_band_pass_signals.py`.

use rayon::prelude::*;

use super::auditory_filters_centre_freq::auditory_filters_centre_freq;
use super::gammatone::gammatone;
use super::tables::EAR_FILTER_SOS;
use crate::dsp::{lfilter_complex, sosfilt};

/// Filters `sig` through the outer/middle ear model, then through all 53
/// gammatone auditory filters (run in parallel with rayon, since each band
/// is independent), returning one full-length band-pass signal per band.
pub fn band_pass_signals(sig: &[f64], fs: f64) -> Vec<Vec<f64>> {
    let signal_filtered = sosfilt(&EAR_FILTER_SOS, sig);
    let centre_freq = auditory_filters_centre_freq();

    (0..53)
        .into_par_iter()
        .map(|band| {
            let (bm, am) = gammatone(centre_freq[band], fs);
            lfilter_complex(&bm, &am, &signal_filtered)
                .iter()
                .map(|c| 2.0 * c.re)
                .collect()
        })
        .collect()
}
