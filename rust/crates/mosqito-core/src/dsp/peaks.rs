//! Peak detection with topographic prominence, matching
//! `scipy.signal.find_peaks(..., prominence=...)`.

/// A detected local maximum and its topographic prominence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    /// Index of the peak in the input signal.
    pub index: usize,
    /// Topographic prominence: the peak's height above the higher of the two
    /// lowest points reached before meeting a taller peak on either side.
    pub prominence: f64,
}

/// Finds local maxima and their prominences.
///
/// A sample is a local maximum when it is strictly greater than its
/// neighbours; for a plateau of equal values, SciPy reports the middle sample
/// (rounding down), which this reproduces. ECMA-418-2 §7.1.5.1 selects
/// modulation-rate candidates by prominence, so the tie-breaking matters.
pub fn find_peaks_with_prominence(x: &[f64]) -> Vec<Peak> {
    local_maxima(x)
        .into_iter()
        .map(|index| Peak {
            index,
            prominence: prominence(x, index),
        })
        .collect()
}

/// Indices of local maxima, taking the middle sample of each flat plateau.
fn local_maxima(x: &[f64]) -> Vec<usize> {
    let n = x.len();
    let mut peaks = Vec::new();
    if n < 3 {
        return peaks;
    }

    let mut i = 1;
    while i < n - 1 {
        if x[i - 1] < x[i] {
            // Walk to the end of a possible plateau.
            let mut j = i;
            while j + 1 < n && x[j + 1] == x[i] {
                j += 1;
            }
            if j + 1 < n && x[j + 1] < x[i] {
                peaks.push((i + j) / 2);
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    peaks
}

/// Topographic prominence of the peak at `index`.
///
/// Walk outwards in each direction until reaching a sample taller than the
/// peak (or the signal's end), tracking the lowest value seen. The prominence
/// is the peak's height above the higher of the two minima.
fn prominence(x: &[f64], index: usize) -> f64 {
    let height = x[index];

    let mut left_min = height;
    let mut i = index;
    while i > 0 {
        i -= 1;
        if x[i] > height {
            break;
        }
        left_min = left_min.min(x[i]);
    }

    let mut right_min = height;
    let mut j = index;
    while j + 1 < x.len() {
        j += 1;
        if x[j] > height {
            break;
        }
        right_min = right_min.min(x[j]);
    }

    height - left_min.max(right_min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn finds_simple_interior_maxima() {
        let x = [0.0, 1.0, 0.0, 2.0, 0.0];
        let idx: Vec<usize> = find_peaks_with_prominence(&x)
            .iter()
            .map(|p| p.index)
            .collect();
        assert_eq!(idx, vec![1, 3]);
    }

    #[test]
    fn ignores_endpoints() {
        let x = [5.0, 1.0, 0.0, 1.0, 5.0];
        assert!(find_peaks_with_prominence(&x).is_empty());
    }

    #[test]
    fn reports_the_middle_of_a_plateau() {
        let x = [0.0, 1.0, 1.0, 1.0, 0.0];
        let idx: Vec<usize> = find_peaks_with_prominence(&x)
            .iter()
            .map(|p| p.index)
            .collect();
        assert_eq!(idx, vec![2]);

        // Even-length plateau rounds down, as SciPy does.
        let x = [0.0, 1.0, 1.0, 0.0];
        let idx: Vec<usize> = find_peaks_with_prominence(&x)
            .iter()
            .map(|p| p.index)
            .collect();
        assert_eq!(idx, vec![1]);
    }

    #[test]
    fn rising_or_falling_plateaus_are_not_peaks() {
        let x = [0.0, 1.0, 1.0, 2.0, 0.0];
        let idx: Vec<usize> = find_peaks_with_prominence(&x)
            .iter()
            .map(|p| p.index)
            .collect();
        assert_eq!(idx, vec![3], "a step up is not a local maximum");
    }

    #[test]
    fn prominence_of_an_isolated_peak_reaches_the_signal_floor() {
        let x = [0.0, 1.0, 5.0, 1.0, 0.0];
        let peaks = find_peaks_with_prominence(&x);
        assert_eq!(peaks.len(), 1);
        assert_relative_eq!(peaks[0].prominence, 5.0, epsilon = 1e-15);
    }

    #[test]
    fn prominence_is_measured_against_the_higher_saddle() {
        // The small peak at index 3 is bounded by the taller peaks either side;
        // its prominence is measured from the higher of the two valleys.
        let x = [0.0, 10.0, 2.0, 6.0, 1.0, 10.0, 0.0];
        let peaks = find_peaks_with_prominence(&x);
        let small = peaks
            .iter()
            .find(|p| p.index == 3)
            .expect("middle peak found");
        // Left valley is 2.0, right valley is 1.0; the higher is 2.0.
        assert_relative_eq!(small.prominence, 4.0, epsilon = 1e-15);
    }

    #[test]
    fn tallest_peak_uses_the_global_minimum_on_its_shallower_side() {
        let x = [3.0, 1.0, 9.0, 2.0, 4.0];
        let peaks = find_peaks_with_prominence(&x);
        let tallest = peaks
            .iter()
            .find(|p| p.index == 2)
            .expect("tallest peak found");
        // Nothing exceeds 9.0, so both walks run to the ends: minima 1.0 and
        // 2.0, the higher being 2.0.
        assert_relative_eq!(tallest.prominence, 7.0, epsilon = 1e-15);
    }

    #[test]
    fn handles_short_inputs() {
        assert!(find_peaks_with_prominence(&[]).is_empty());
        assert!(find_peaks_with_prominence(&[1.0]).is_empty());
        assert!(find_peaks_with_prominence(&[1.0, 2.0]).is_empty());
    }
}
