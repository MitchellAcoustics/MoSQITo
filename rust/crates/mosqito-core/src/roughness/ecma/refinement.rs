//! ECMA-418-2 (2nd Ed, 2022) §7.1.5.1 peak refinement, Eqs. 73-81.
//!
//! Implements the two corrections from Wanty, Glesser & Casagrande Hirono,
//! *"ECMA-418-2 roughness, a challenging implementation"*, INTERNOISE 2024
//! (D-ecma-2, D-ecma-3 in `DEVIATIONS.md`): the closed-form solution of the
//! quadratic peak fit (Eq. 73-76) in place of solving the standard's matrix
//! form, and the corrected bias-adjustment formula (Eq. 78, which the
//! standard as published gets wrong). Matches `_refinement.py`.

/// Table from ECMA-418-2 Table 10 (the `E(theta)` bias values), Eq. 78-81.
const E: [f64; 34] = [
    0.0, 0.0457, 0.0907, 0.1346, 0.1765, 0.2157, 0.2515, 0.2828, 0.3084, 0.3269, 0.3364, 0.3348,
    0.3188, 0.2844, 0.2259, 0.1351, 0.0000, -0.1351, -0.2259, -0.2844, -0.3188, -0.3348, -0.3364,
    -0.3269, -0.3084, -0.2828, -0.2515, -0.2157, -0.1765, -0.1346, -0.0907, -0.0457, 0.000, 0.000,
];

/// Eq. 78 (corrected form) bias-correction term.
fn rho(f: f64, delta_f: f64) -> f64 {
    let mut b = [0.0f64; 34];
    for (theta, bv) in b.iter_mut().enumerate() {
        *bv = ((f / delta_f).floor() + theta as f64 / 32.0) * delta_f - (f + E[theta]);
    }

    // Eq. 80: index of the smallest |B|.
    let theta_min = b
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, c)| a.abs().total_cmp(&c.abs()))
        .map(|(i, _)| i)
        .expect("non-empty");

    // Eq. 81.
    let theta_corr = if theta_min > 0 && b[theta_min] * b[theta_min - 1] < 0.0 {
        theta_min
    } else {
        theta_min + 1
    };

    E[theta_corr]
        - (E[theta_corr] - E[theta_corr - 1]) * b[theta_corr] / (b[theta_corr] - b[theta_corr - 1])
}

/// Refines a peak at index `kpi` in `phi_e_l_z`, per §7.1.5.1.
///
/// Returns `(mod_rate, amp)`: the refined modulation rate [Hz] and the
/// peak's summed amplitude.
///
/// `kpi` is always in `2..=254` for every peak `find_peaks` can return from
/// `peak_picking` (a search window starting at index 2, and `find_peaks`
/// never reports an array boundary as a peak) — so the `kpi == 0` branch
/// below can't be reached from this crate's own call path, but is kept for
/// defensiveness and fidelity to the source, matching `_refinement.py:66-71`
/// exactly (Python's `elif kpi == 255` is equally unreachable there, for the
/// same reason).
pub fn refinement(kpi: usize, phi_e_l_z: &[f64]) -> (f64, f64) {
    let n = phi_e_l_z.len();
    let amp = if kpi == 0 {
        phi_e_l_z[kpi] + phi_e_l_z[kpi + 1]
    } else if kpi == n - 1 {
        phi_e_l_z[kpi - 1] + phi_e_l_z[kpi]
    } else {
        phi_e_l_z[kpi - 1] + phi_e_l_z[kpi] + phi_e_l_z[kpi + 1]
    };

    let delta_f = 1500.0 / 512.0;
    let k = kpi as f64;
    let f = (k
        - (phi_e_l_z[kpi + 1] - phi_e_l_z[kpi - 1])
            / (2.0 * phi_e_l_z[kpi - 1] + 2.0 * phi_e_l_z[kpi + 1] - 4.0 * phi_e_l_z[kpi]))
        * delta_f;

    let mod_rate = f + rho(f, delta_f);
    (mod_rate, amp)
}
