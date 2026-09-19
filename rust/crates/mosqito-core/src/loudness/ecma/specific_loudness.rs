//! ECMA-418-2 (2nd Ed, 2022) §5.1.5 (block segmentation, Eqs. 18-20),
//! §5.1.6 (rectification), §5.1.7 (RMS, Eq. 22) and §5.1.8-5.1.9
//! (nonlinearity + absolute-threshold clip, Eq. 23) for one band, fused into
//! a single pass.
//!
//! MoSQITo's Python (`_ecma_time_segmentation.py` + `loudness_ecma.py`'s
//! main loop) builds this out as several materialised intermediate arrays:
//! a `(nblocks, sb)` index array, a same-shaped gathered block array, its
//! rectified copy, then a per-block RMS reduction — repeated 53 times, one
//! per band. This fuses all of that into one pass per band with no
//! intermediate array, computing each block's RMS and mean time directly
//! from the band-pass signal and the index formula below.
//!
//! # The block index formula (Eqs. 18-19)
//!
//! For block `l` (0-indexed) and sample `k` within it (`k` in `0..sb`),
//! Python computes the sample index via
//! `numpy.linspace(l*sh, l*sh+sb, sb).astype(int32)[k]`, i.e. `sb` points
//! spanning a range of length `sb` — a step of `sb/(sb-1)`, fractionally
//! *more* than 1. Truncating that to an integer means the last sample in
//! each block sometimes jumps by 2 instead of 1 (one physical sample in the
//! block's span of `sb+1` positions is skipped). This looks unusual, but
//! nothing in the plan's research flagged it as a transcription bug against
//! the standard (unlike the several genuine bugs recorded in
//! `DEVIATIONS.md`), and its numeric effect — omitting on the order of 1
//! sample out of every 2048 from an RMS average — is negligible. Reproduced
//! exactly here (`floor(l*sh + k*sb/(sb-1))`, confirmed by direct comparison
//! against `numpy.linspace(...).astype(int32)` to be bit-for-bit identical
//! across every `(sb, sh, n_new)` combination this port's tests exercise),
//! rather than "corrected" against an assumption of what was intended.
//!
//! This implementation only supports a single scalar `sb`/`sh` shared across
//! all 53 bands (not MoSQITo's per-band list), which is all Phase 1's public
//! API exposes; with a scalar `sb`, `i_start` (`_ecma_time_segmentation.py`
//! Eq. 19) is always zero, which this relies on.

use super::nonlinearity::nonlinearity;

const FS: f64 = 48000.0;

/// The number of blocks Eq. 20 gives for `n_new` (padded sample count,
/// excluding the leading `sb` zeros) and hop size `sh`.
pub fn n_blocks(n_new: usize, sh: usize) -> usize {
    (n_new + sh).div_ceil(sh) - 1
}

/// The fractional-step constant `block_sample_index` advances by per `k`,
/// `sb/(sb-1)` — depends only on `sb`, so callers that index many `(l, k)`
/// pairs for the same `sb` (every one of them, in practice) compute it once
/// with this and pass it to [`block_sample_index`] rather than recomputing
/// the same division on every call.
pub fn block_step(sb: usize) -> f64 {
    sb as f64 / (sb - 1) as f64
}

/// The sample index of position `k` (`0..sb`) within block `l`, per the
/// block index formula documented on this module. `step` is [`block_step`]`(sb)`.
pub fn block_sample_index(l: usize, k: usize, sh: usize, step: f64) -> usize {
    ((l * sh) as f64 + k as f64 * step).floor() as usize
}

/// Computes one band's specific loudness and block-mean-time series from
/// its (already gammatone-filtered) band-pass signal.
///
/// `n_new` is [`super::preprocessing::preprocess`]'s second return value —
/// the sample count of the zero-padded signal *excluding* the leading `sb`
/// zeros (Eq. 3); `band_pass_signal` itself must include those leading
/// zeros (i.e. be `preprocess`'s full padded output, filtered).
///
/// Returns `(n_specific, time_axis)`, each of length [`n_blocks`].
pub fn specific_loudness_for_band(
    band_pass_signal: &[f64],
    sb: usize,
    sh: usize,
    n_new: usize,
    ltq_z: f64,
) -> (Vec<f64>, Vec<f64>) {
    let blocks = n_blocks(n_new, sh);
    let step = block_step(sb);

    let mut n_specific = Vec::with_capacity(blocks);
    let mut time_axis = Vec::with_capacity(blocks);

    for l in 0..blocks {
        let mut sum_sq = 0.0f64;
        let mut sum_t = 0.0f64;
        for k in 0..sb {
            let idx = block_sample_index(l, k, sh, step);
            let v = band_pass_signal[idx].max(0.0);
            sum_sq += v * v;
            sum_t += idx as f64 / FS;
        }
        let rms = (2.0 * sum_sq / sb as f64).sqrt();
        let a_prime = nonlinearity(rms).max(ltq_z);
        n_specific.push(a_prime - ltq_z);
        time_axis.push(sum_t / sb as f64);
    }

    (n_specific, time_axis)
}
