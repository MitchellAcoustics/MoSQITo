//! Specific-loudness slope attachment, per ISO 532-1:2017 Annex A (the
//! critical-band-rate loudness pattern stage of the Zwicker stationary
//! method) — converting the 21 core loudness values from [`main_loudness`]
//! into the 240-point specific loudness pattern (0.1–24.0 Bark, 0.1 Bark
//! steps) and the single total loudness value.
//!
//! `_calc_slopes.py`'s vectorised form is a masked-array rewrite of an
//! originally scalar, sequential algorithm; the file's own comments
//! (`_calc_slopes.py:184-198`) give that original scalar routine for one
//! slope segment directly. This module reproduces the *sequence of steps*
//! that vectorised code actually performs, not just the closed-form segment
//! formula: within a band, `_calc_slopes.py` steps through the fine
//! (0.1 Bark) grid one position at a time, and checks for a segment switch
//! at each position independently — so it processes **at most one switch per
//! grid position**, even when a segment is so short that its own endpoint
//! has *already* been passed by the time it's reached. The next switch, if
//! needed, only happens on the following position. This is not a cosmetic
//! detail: it changes the value filled at the switch position itself
//! (confirmed by comparing against the installed `mosqito` package — an
//! initial version of this module used a closed-form "jump directly to the
//! next boundary" shortcut, which is mathematically equivalent everywhere
//! *except* at exactly these switch positions, where it does not agree).
//! [`calc_slopes`] mirrors the grid walk directly instead. Validated across
//! 637 cases (`golden_zwst.json`), including both spectra derived from
//! [`main_loudness`] and synthetic core-loudness patterns chosen to exercise
//! rising, falling and mixed slope sequences directly — all but one match
//! exactly. The exception is a perfectly linear, integer-valued synthetic
//! `nm = 0..20`, for which MoSQITo's own `_calc_slopes.py` produces 9
//! *negative* specific-loudness values (confirmed directly against the
//! installed `mosqito` package, independent of the golden file) — physically
//! impossible under ISO 532-1's model, and a pattern [`main_loudness`] cannot
//! itself produce (its output is continuous and non-negative by
//! construction). That looks like a genuine instability in the original
//! vectorised Python for a degenerate input outside `calc_slopes`'s real
//! domain, not a target this port reproduces; see the golden test for
//! details.
//!
//! [`main_loudness`]: crate::loudness::zwst::main_loudness

use super::main_loudness::main_loudness;

const N_SPECIFIC_LEN: usize = 240;

/// Upper limits of the 21 critical bands, in Bark.
const ZUP: [f64; 21] = [
    0.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6, 12.3, 13.8, 15.2, 16.7, 18.1, 19.3, 20.6,
    21.8, 22.7, 23.6, 24.0,
];

/// Range boundaries of specific loudness for selecting the upper-slope
/// steepness, descending from 21.5 to 0.
const RNS: [f64; 18] = [
    21.5, 18.0, 15.1, 11.5, 9.0, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.30, 0.22, 0.15, 0.10,
    0.035, 0.0,
];

/// Steepness of the upper slopes (loudness decrease per Bark) for each `RNS`
/// range, as a function of critical band number (columns 0–7). Column 7 is
/// reused for every band number beyond 7 — see [`usl_at`].
const USL: [[f64; 8]; 18] = [
    [13.0, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5],
    [9.0, 7.5, 6.0, 5.1, 4.5, 4.5, 4.5, 4.5],
    [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9],
    [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
    [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7],
    [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
    [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7],
    [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
    [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1],
    [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
    [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
    [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
    [0.40, 0.33, 0.26, 0.24, 0.24, 0.22, 0.22, 0.22],
    [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
    [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
    [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
    [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
    [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
];

/// `USL[row]` at critical-band-number column `col`, extending the table's 8
/// columns by repeating the last one — `_calc_slopes.py:91`'s
/// `usl_reshaped`.
fn usl_at(row: usize, col: usize) -> f64 {
    USL[row][col.min(7)]
}

/// Index of the `RNS` row a loudness value falls into: the count of `RNS`
/// entries strictly greater than `value` (or `>=`, if `inclusive`), matching
/// `_get_rns_index`. `RNS` is sorted descending, so this is the row whose
/// range `value` falls into when scanning from the loudest end; clamped to
/// 17 (`RNS`'s last index) as `_get_rns_index.py:38` does.
fn rns_index(value: f64, inclusive: bool) -> usize {
    let count = RNS
        .iter()
        .filter(|&&r| {
            if inclusive {
                round8(value) <= round8(r)
            } else {
                round8(value) < round8(r)
            }
        })
        .count();
    count.min(17)
}

/// Rounds to 8 decimal places, matching `_calc_slopes.py`'s `dec_compare = 8`
/// — used wherever the original compares two loudness or Bark values for
/// equality, to absorb floating-point noise from upstream computation.
fn round8(x: f64) -> f64 {
    (x * 1e8).round() / 1e8
}

/// Converts a critical-band boundary (one of the fixed [`ZUP`] values) to its
/// index in the 240-point specific-loudness array, as an *exclusive* upper
/// bound — matching `zup_ea = (zup * 10).astype(int32)`
/// (`_calc_slopes.py:111`). `ZUP * 10` happens to be an exact integer in
/// `f64` for every entry, so truncation and rounding agree here.
///
/// This is *not* used for the dynamic within-band slope boundary (where a
/// segment's analytically-computed endpoint crosses into a new grid
/// position): `_calc_slopes.py` never computes an index for that endpoint
/// directly, it walks the fine grid one 0.1 Bark position at a time and
/// switches segments at the first position whose own grid value has reached
/// or passed it (`_calc_slopes.py:246-248`). [`calc_slopes`] mirrors that
/// walk directly rather than trying to shortcut it with a closed-form index,
/// because the two do not agree — concretely, for a segment endpoint of
/// `1.280691`, `round(1.280691 * 10) = 13`, but the crossing Python's own
/// grid walk finds is at position `12` (grid value `1.3`), one earlier — and
/// getting this wrong changes the value filled at the switch position
/// itself, not just which position it lands on. Found by comparing against
/// the installed `mosqito` package.
fn band_boundary_index_exclusive(z: f64) -> usize {
    (z * 10.0).trunc() as usize
}

/// Attaches slopes to the core loudness pattern and integrates total
/// loudness, per ISO 532-1:2017 Annex A.
///
/// `nm` is [`main_loudness`]'s 21-value output. Returns `(N, N_specific)`:
/// `N` is the single overall loudness value in sone, rounded exactly as ISO
/// 532-1 specifies (3 decimal places up to 16 sone, 2 decimal places above);
/// `N_specific` is the 240-point specific loudness pattern in sone/Bark.
pub fn calc_slopes(nm: &[f64; 21]) -> (f64, [f64; N_SPECIFIC_LEN]) {
    let mut n_specific = [0.0f64; N_SPECIFIC_LEN];
    let mut n_total = 0.0f64;

    // Baseline pre-fill, matching `_calc_slopes.py`'s own initial vectorised
    // fill (`N_specific[zup_ea[i-1]:zup_ea[i]] = nm[i]` for bands 0..19). The
    // main loop below relies on this being in place already: when a decaying
    // slope reaches this band's own target loudness before reaching the
    // band's edge, it stops writing for the rest of the band — mirroring
    // Python's early exit once every element has "caught up" — and leaves
    // these pre-filled values as the (already correct) answer. Band 20 is
    // never covered by this pre-fill in Python either (its loop stops at
    // 19); harmless, since `nm[20]` is always 0.0 (see [`main_loudness`])
    // and 0.0 is also `n_specific`'s own default.
    {
        let mut prev = 0usize;
        for (i, &nm_i) in nm.iter().enumerate().take(20) {
            let end = band_boundary_index_exclusive(ZUP[i]).min(N_SPECIFIC_LEN);
            for v in n_specific.iter_mut().take(end).skip(prev) {
                *v = nm_i;
            }
            prev = end;
        }
    }

    let mut z1 = 0.0f64;
    let mut n1 = 0.0f64;

    for i in 0..21 {
        let zup_i = ZUP[i];
        let zup_ea_i = band_boundary_index_exclusive(zup_i).min(N_SPECIFIC_LEN);
        let zup_ea_prev = if i == 0 {
            0
        } else {
            band_boundary_index_exclusive(ZUP[i - 1])
        };
        let nm_i = nm[i];
        // The USL column is the *previous* band's number; at i=0 this is
        // never actually read (n1 starts at 0, so band 0 can only take the
        // rising branch below), matching Python's `usl_reshaped[idx, i-1]`
        // being computed but unused there too.
        let usl_col = i.saturating_sub(1);

        if round8(n1) <= round8(nm_i) {
            // Rising (or flat): a single rectangle at the band's own level,
            // already present via the baseline pre-fill above.
            n_total += nm_i * (zup_i - z1);
            z1 = zup_i;
            n1 = nm_i;
            continue;
        }

        // Falling: set up the first decaying segment, exactly as a mid-band
        // switch does below.
        let mut row = rns_index(n1, false);
        let mut target = RNS[row].max(nm_i);
        let mut usl_val = usl_at(row, usl_col);
        let mut seg_z1 = z1;
        let mut seg_n1 = n1;
        let mut seg_z2 = (seg_n1 - target) / usl_val + seg_z1;
        let mut clipped = seg_z2 >= zup_i;
        if clipped {
            seg_z2 = zup_i;
        }
        let mut seg_n2 = if clipped {
            seg_n1 - (seg_z2 - seg_z1) * usl_val
        } else {
            target
        };
        n_total += (seg_z2 - seg_z1) * (seg_n1 + seg_n2) / 2.0;

        // Step the fine grid one position (0.1 Bark) at a time. At each
        // position, check whether the *currently active* segment's endpoint
        // has already been reached; if so, switch to exactly one new
        // segment — never more than one per position, even when that new
        // segment's own endpoint is *also* already behind this grid
        // position (that gets caught on the *next* position instead). This
        // one-switch-per-position pacing is not cosmetic: it changes the
        // value this fills at the switch position itself, matching
        // `_calc_slopes.py`'s own structure (one `mask_z_bigger_z2` check
        // per inner-loop position), verified against the installed
        // `mosqito` package.
        let mut reached_target = false;
        // `k` doubles as the fine-grid Bark position (via `z`) and the
        // `n_specific` index, with branches in between that don't write to
        // `n_specific` at all (the "reached target" break) — a plain
        // iterator over `n_specific` would have to reconstruct `k` anyway.
        #[allow(clippy::needless_range_loop)]
        for k in zup_ea_prev..zup_ea_i {
            let z = (k + 1) as f64 * 0.1;

            if round8(seg_z2) <= round8(z) {
                if round8(seg_n2) <= round8(nm_i) {
                    // This segment's endpoint already reached the band's own
                    // target: the remaining width is a flat rectangle at
                    // nm_i (matching `_calc_slopes.py`'s `N += n2 * dz` for
                    // this case), and the baseline nm_i fill already covers
                    // the grid positions themselves.
                    n_total += nm_i * (zup_i - seg_z2);
                    reached_target = true;
                    break;
                }
                row = rns_index(seg_n2, true);
                target = RNS[row].max(nm_i);
                usl_val = usl_at(row, usl_col);
                seg_z1 = seg_z2;
                seg_n1 = seg_n2;
                seg_z2 = (seg_n1 - target) / usl_val + seg_z1;
                clipped = seg_z2 >= zup_i;
                if clipped {
                    seg_z2 = zup_i;
                }
                seg_n2 = if clipped {
                    seg_n1 - (seg_z2 - seg_z1) * usl_val
                } else {
                    target
                };
                n_total += (seg_z2 - seg_z1) * (seg_n1 + seg_n2) / 2.0;
            }

            n_specific[k] = seg_n1 - (z - seg_z1) * usl_val;
        }

        if reached_target {
            z1 = zup_i;
            n1 = nm_i;
        } else {
            z1 = seg_z2;
            n1 = seg_n2;
        }
    }

    // ISO 532-1's fixed-precision rounding of the total loudness value.
    n_total = n_total.max(0.0);
    n_total = if n_total <= 16.0 {
        (n_total * 1000.0 + 0.5).floor() / 1000.0
    } else {
        (n_total * 100.0 + 0.5).floor() / 100.0
    };

    (n_total, n_specific)
}

/// Convenience: core loudness followed by slope attachment, matching what
/// `loudness_zwst` does with a single third-octave spectrum in dB.
pub fn loudness_from_spectrum(
    spec_third: &[f64],
    field_type: super::main_loudness::FieldType,
) -> (f64, [f64; N_SPECIFIC_LEN]) {
    let nm = main_loudness(spec_third, field_type);
    calc_slopes(&nm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loudness::zwst::FieldType;

    #[test]
    fn silence_gives_zero_loudness() {
        let (n, n_spec) = calc_slopes(&[0.0; 21]);
        assert_eq!(n, 0.0);
        assert!(n_spec.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn total_loudness_is_never_negative() {
        let nm: [f64; 21] = std::array::from_fn(|i| if i % 3 == 0 { 5.0 } else { 0.0 });
        let (n, n_spec) = calc_slopes(&nm);
        assert!(n >= 0.0);
        assert!(n_spec.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn a_pure_tone_spectrum_gives_a_plausible_loudness() {
        // Sanity check ahead of the golden-vector test: a moderate flat
        // spectrum should land in a physically reasonable sone range, not
        // some wildly wrong magnitude from a boundary-arithmetic slip.
        let spec = [60.0; 28];
        let (n, _) = loudness_from_spectrum(&spec, FieldType::Free);
        assert!(
            n > 1.0 && n < 100.0,
            "N = {n} sone is not plausible for a 60 dB flat spectrum"
        );
    }
}
