//! ECMA-418-2 (2nd Ed, 2022) §7.1.3's Von Hann window — normalised
//! differently from `numpy.hanning`, so implemented directly rather than
//! reused from elsewhere. Matches `_von_hann_window.py`.

pub fn von_hann_window(n: usize) -> Vec<f64> {
    let norm = 0.375f64.sqrt();
    (0..n)
        .map(|i| {
            let arg = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            (0.5 - 0.5 * arg.cos()) / norm
        })
        .collect()
}
