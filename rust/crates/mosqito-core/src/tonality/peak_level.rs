//! Port of `_peak_level`: corrects a peak's level by summing neighbouring
//! bins that decrease monotonically from the peak within a 10 dB window —
//! compensating for spectral resolution spreading a tone's energy across
//! more than one FFT bin.

/// Corrected SPL at `peak_index`, matching `_peak_level(freqs, spec,
/// peak_index)`. `freqs` is unused by the algorithm itself (Python takes it
/// too, for signature symmetry with its other tonality helpers) but kept
/// here to mirror the call site.
pub fn peak_level(_freqs: &[f64], spec: &[f64], peak_index: usize) -> f64 {
    let li = spec[peak_index];
    let mut l = li;

    // Right side.
    let mut temp = peak_index + 1;
    if temp != spec.len() {
        let mut ltemp = li;
        while ltemp - spec[temp].abs() > 0.0 {
            if li - spec[temp] < 10.0 {
                ltemp = spec[temp];
                l = 10.0 * (10f64.powf(l / 10.0) + 10f64.powf(spec[temp] / 10.0)).log10();
                temp += 1;
                if temp == spec.len() {
                    temp -= 1;
                    ltemp = -1.0;
                }
            } else {
                ltemp = -1.0;
            }
        }
    }

    // Left side.
    if peak_index != 0 {
        let mut temp: isize = peak_index as isize - 1;
        let mut ltemp = li;
        while ltemp - spec[temp as usize].abs() > 0.0 {
            let t = temp as usize;
            if li - spec[t] < 10.0 {
                ltemp = spec[t];
                l = 10.0 * (10f64.powf(l / 10.0) + 10f64.powf(spec[t] / 10.0)).log10();
                temp -= 1;
                if temp < 0 {
                    temp += 1;
                    ltemp = -1.0;
                }
            } else {
                ltemp = -1.0;
            }
        }
    }

    l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_isolated_peak_is_unchanged() {
        let freqs = vec![0.0; 5];
        let spec = vec![10.0, 10.0, 50.0, 10.0, 10.0];
        let l = peak_level(&freqs, &spec, 2);
        assert!((l - 50.0).abs() < 1e-9);
    }

    #[test]
    fn a_peak_at_the_signal_boundary_does_not_panic() {
        let freqs = vec![0.0; 3];
        let spec = vec![50.0, 40.0, 30.0];
        let _ = peak_level(&freqs, &spec, 0);
        let _ = peak_level(&freqs, &spec, 2);
    }
}
