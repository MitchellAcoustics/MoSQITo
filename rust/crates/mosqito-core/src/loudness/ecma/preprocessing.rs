//! ECMA-418-2 (2nd Ed, 2022) §5.1.2: windowing and zero-padding. Matches
//! `_preprocessing.py`, except that it does not mutate the caller's signal
//! (Python's `signal[:240] *= w_fadein` does — see `DEVIATIONS.md`).

const N_FADEIN: usize = 240;

/// Applies a raised-cosine fade-in to the first 5 ms (240 samples) and
/// zero-pads the start (by `sb`) and end (out to a whole number of `sh`-size
/// hops past the signal, per Eq. 3) of `signal`.
///
/// Returns `(padded_signal, n_new)`: `n_new` is the total sample count after
/// the trailing zero padding, matching Eq. 3 exactly (the *count*, not
/// `padded_signal.len()` — the leading `sb` zeros are not part of it).
///
/// # Panics
/// Panics if `signal` has fewer than 240 samples (as does MoSQITo's Python,
/// via a numpy broadcasting error — no real conformance signal is this
/// short).
pub fn preprocess(signal: &[f64], sb: usize, sh: usize) -> (Vec<f64>, usize) {
    assert!(
        signal.len() >= N_FADEIN,
        "signal must have at least {N_FADEIN} samples"
    );

    let mut sig = signal.to_vec();
    for (i, s) in sig.iter_mut().take(N_FADEIN).enumerate() {
        let w = 0.5 - 0.5 * (std::f64::consts::PI * i as f64 / N_FADEIN as f64).cos();
        *s *= w;
    }

    let n_samples = sig.len();
    let n_zeros_start = sb;
    let n_new = sh * ((n_samples + sh + sb).div_ceil(sh) - 1);
    let n_zeros_end = n_new - n_samples;

    let mut padded = vec![0.0f64; n_zeros_start];
    padded.extend_from_slice(&sig);
    padded.resize(padded.len() + n_zeros_end, 0.0);

    (padded, n_new)
}
