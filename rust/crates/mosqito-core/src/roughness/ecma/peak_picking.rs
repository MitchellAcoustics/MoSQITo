//! ECMA-418-2 (2nd Ed, 2022) §7.1.5.1: finds and refines the modulation-rate
//! peaks of one time/band block's noise-reduced power spectrum. Matches
//! `_peak_picking.py`.

use crate::dsp::find_peaks_with_prominence;

use super::refinement::refinement;

/// Finds up to 10 modulation-rate peaks (by prominence) in `phi_e_l_z`,
/// refines each, and returns `(f_p, amplitudes)`.
pub fn peak_picking(phi_e_l_z: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // Search window starts at k=2 (Eq. 72's own convention).
    let search = &phi_e_l_z[2..];
    let mut peaks = find_peaks_with_prominence(search);
    for p in &mut peaks {
        p.index += 2;
    }

    if peaks.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Eq. 72: drop maxima below 5% of the tallest one found.
    let max_amp = peaks
        .iter()
        .map(|p| phi_e_l_z[p.index])
        .fold(f64::NEG_INFINITY, f64::max);
    peaks.retain(|p| phi_e_l_z[p.index] > 0.05 * max_amp);

    // Keep only the 10 highest-prominence peaks (ascending prominence order
    // is irrelevant to the result, so a straightforward partial sort by
    // prominence suffices).
    if peaks.len() > 10 {
        peaks.sort_by(|a, b| a.prominence.total_cmp(&b.prominence));
        let keep = peaks.split_off(peaks.len() - 10);
        peaks = keep;
    }

    let mut f_p = Vec::with_capacity(peaks.len());
    let mut a = Vec::with_capacity(peaks.len());
    for p in &peaks {
        let (mod_rate, amp) = refinement(p.index, phi_e_l_z);
        f_p.push(mod_rate);
        a.push(amp);
    }

    (f_p, a)
}
