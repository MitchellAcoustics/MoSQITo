//! Time-domain block segmentation.

use ndarray::Array2;

/// Segments a signal into overlapping (or non-overlapping) blocks, matching
/// `mosqito.utils.time_segmentation` for `is_ecma=false`.
///
/// `noverlap` is, despite the name, a **hop size**: consecutive blocks start
/// `noverlap` samples apart, and block `l` covers samples
/// `[l*noverlap, l*noverlap + nperseg)`. It defaults to `nperseg / 2`
/// (50% overlap); passing `Some(0)` is treated as `nperseg` (no overlap),
/// matching `time_segmentation.py:39-40`.
///
/// Returns `(blocks, time)`: `blocks` is (`nperseg`, `nseg`), one column per
/// block; `time` holds each block's mean sample time, in seconds.
///
/// # Panics
/// Panics if `nperseg` is zero or exceeds `sig.len()`.
pub fn time_segmentation(
    sig: &[f64],
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
) -> (Array2<f64>, Vec<f64>) {
    assert!(nperseg > 0, "nperseg must be positive");
    assert!(sig.len() >= nperseg, "signal shorter than one block");

    let hop = match noverlap {
        None => nperseg / 2,
        Some(0) => nperseg,
        Some(h) => h,
    };

    let mut starts = Vec::new();
    let mut l = 0usize;
    while l * hop + nperseg <= sig.len() {
        starts.push(l * hop);
        l += 1;
    }

    let nseg = starts.len();
    let mut blocks = Array2::<f64>::zeros((nperseg, nseg));
    let mut time = Vec::with_capacity(nseg);
    for (col, &start) in starts.iter().enumerate() {
        let block = &sig[start..start + nperseg];
        // A column of a (row-major, ndarray's default) Array2 is not
        // contiguous, so this assigns from a view rather than requiring a
        // mutable slice.
        blocks
            .column_mut(col)
            .assign(&ndarray::ArrayView1::from(block));
        let mean_index = start as f64 + (nperseg as f64 - 1.0) / 2.0;
        time.push(mean_index / fs);
    }

    (blocks, time)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn non_overlapping_blocks_tile_the_signal() {
        let sig: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (blocks, _time) = time_segmentation(&sig, 1.0, 5, Some(0));
        assert_eq!(blocks.shape(), &[5, 4]);
        assert_eq!(blocks.column(0).to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            blocks.column(3).to_vec(),
            vec![15.0, 16.0, 17.0, 18.0, 19.0]
        );
    }

    #[test]
    fn default_hop_is_half_the_block_length() {
        let sig: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let (blocks, _time) = time_segmentation(&sig, 1.0, 4, None);
        // hop = 4/2 = 2; blocks start at 0, 2, 4 (4+4=8<=10), 6 (6+4=10<=10).
        assert_eq!(blocks.shape(), &[4, 4]);
        assert_eq!(blocks.column(1).to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn time_axis_is_each_blocks_mean_sample_time() {
        let sig: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let (_blocks, time) = time_segmentation(&sig, 2.0, 4, Some(4));
        // Block 0 covers samples 0..4, mean index 1.5, at fs=2 -> 0.75 s.
        assert_relative_eq!(time[0], 0.75, epsilon = 1e-12);
    }

    #[test]
    fn matches_the_reference_block_count_from_mosqitos_own_test() {
        // tests/utils/test_time_segmentation.py: 1 s at 48 kHz, sb=8192,
        // sh=2048, is_ecma=False -> 20 blocks.
        let sig = vec![0.0; 48000];
        let (blocks, _) = time_segmentation(&sig, 48000.0, 8192, Some(2048));
        assert_eq!(blocks.ncols(), 20);
    }
}
