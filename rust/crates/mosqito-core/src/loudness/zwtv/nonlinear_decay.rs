//! Nonlinear temporal decay of the hearing system, per ISO 532-1's
//! Zwicker-method time-varying loudness. Matches `_nonlinear_decay.py`'s
//! `_nl_loudness`.
//!
//! This is the one genuinely sequential stage in the whole `loudness_zwtv`
//! pipeline: state (`uo`, `u2`) carries from one upsampled time step to the
//! next, so the outer loop over time cannot be parallelized. It *is*
//! embarrassingly parallel across the 21 critical bands at each step — the
//! original Python instead processes all 21 bands as masked-numpy operations
//! per step, which is exactly backwards for performance (a few numpy calls
//! per step dominated by Python-loop overhead, for `ntime * 24` steps). Here
//! that inner dimension is a plain scalar loop instead.

use ndarray::Array2;

const NL_ITER: usize = 24;
const SAMPLE_RATE: f64 = 2000.0;
const T_SHORT: f64 = 0.005;
const T_LONG: f64 = 0.015;
const T_VAR: f64 = 0.075;

/// The six filter coefficients `_nonlinear_decay.py:32-47` derives from the
/// three time constants, closed-form (not iterative), computed once.
fn nl_coefficients() -> [f64; 6] {
    let delta_t = 1.0 / (SAMPLE_RATE * NL_ITER as f64);
    let p = (T_VAR + T_LONG) / (T_VAR * T_SHORT);
    let q = 1.0 / (T_SHORT * T_VAR);
    let disc = (p * p / 4.0 - q).sqrt();
    let lambda_1 = -p / 2.0 + disc;
    let lambda_2 = -p / 2.0 - disc;
    let den = T_VAR * (lambda_1 - lambda_2);
    let e1 = (lambda_1 * delta_t).exp();
    let e2 = (lambda_2 * delta_t).exp();
    [
        (e1 - e2) / den,
        ((T_VAR * lambda_2 + 1.0) * e1 - (T_VAR * lambda_1 + 1.0) * e2) / den,
        ((T_VAR * lambda_1 + 1.0) * e1 - (T_VAR * lambda_2 + 1.0) * e2) / den,
        (T_VAR * lambda_1 + 1.0) * (T_VAR * lambda_2 + 1.0) * (e1 - e2) / den,
        (-delta_t / T_LONG).exp(),
        (-delta_t / T_VAR).exp(),
    ]
}

/// One band's state transition for one upsampled time step, mirroring
/// `_nonlinear_decay.py:89-125`'s four mask blocks exactly (including their
/// order, since a later block's condition reads the value an earlier block
/// in the same step may have just written).
fn nl_step(uo_prev: f64, u2_prev: f64, ui_cur: f64, b: &[f64; 6]) -> (f64, f64) {
    // Defaults to `ui_cur`: `uo_mat` starts as a full copy of the upsampled
    // input, so an entry no branch below touches keeps that value.
    let mut uo_cur = ui_cur;
    if uo_prev > u2_prev {
        let uo2 = uo_prev * b[2] - u2_prev * b[3];
        if uo2 >= ui_cur {
            uo_cur = uo2;
        }
    } else {
        let uo2 = uo_prev * b[4];
        if uo2 >= ui_cur {
            uo_cur = uo2;
        }
    }

    // Unconditional in the original: `u2_mat[:, col] = uo_mat[:, col]`.
    let mut u2_cur = uo_cur;

    let u22 = uo_prev * b[0] - u2_prev * b[1];
    if ui_cur < uo_prev && uo_prev > u2_prev && u22 <= uo_cur {
        u2_cur = u22;
    }

    let u2_alt = (u2_prev - ui_cur) * b[5] + ui_cur;
    let near_equal = (ui_cur - uo_prev).abs() < 1e-5 && uo_cur <= u2_prev;
    if ui_cur >= uo_prev && !near_equal {
        u2_cur = u2_alt;
    }

    (uo_cur, u2_cur)
}

/// Applies the nonlinear decay to `core_loudness` (21 bands x `ntime`
/// frames), returning a same-shaped array.
///
/// # Wraparound at the first time step (documented, not "fixed")
///
/// The very first upsampled step's "previous" state is read from index
/// `col - 1 = -1`, i.e. Python's (and this port's) negative-index
/// wraparound to the *last* upsampled column of `uo` (still holding its
/// initial value — the last frame's core loudness divided by 24, an
/// artifact of the ramp-to-zero the last frame's own delta uses — since the
/// sequential loop hasn't reached it yet) and a `u2` of exactly `0.0` (its
/// zero-initialized default, likewise untouched). Concretely: the decay
/// filter's state at time zero is seeded from the *end* of the signal
/// rather than silence.
///
/// `mosqito-rs` reproduces this exactly rather than seeding from silence,
/// for two reasons: first, ISO 532-1 does not publish an initial-condition
/// prescription to check this against, so "silence" is an assumption, not a
/// standards fact; second, its effect is bounded to roughly the recurrence's
/// slowest time constant (`t_var` = 75 ms, a handful of 2 kHz frames) at the
/// very start of the signal, which the ISO 532-1 Annex B.4/B.5 conformance
/// gate already tolerates (±2 ms realignment, ≤1% of samples allowed
/// outside tolerance) — confirmed by that gate passing in
/// `tests/conformance_loudness_zwtv.rs`. See `DEVIATIONS.md`.
pub fn nl_loudness(core_loudness: &Array2<f64>) -> Array2<f64> {
    let nm_wide = core_loudness.nrows();
    let ntime = core_loudness.ncols();
    let b = nl_coefficients();
    let total_cols = ntime * NL_ITER;

    let mut output = Array2::<f64>::zeros((nm_wide, ntime));
    if ntime == 0 {
        return output;
    }

    let mut uo_prev = vec![0.0f64; nm_wide];
    let u2_prev_init = vec![0.0f64; nm_wide];
    for row in 0..nm_wide {
        let last = core_loudness[[row, ntime - 1]];
        let delta_last = -last / NL_ITER as f64;
        uo_prev[row] = last + (NL_ITER - 1) as f64 * delta_last;
    }
    let mut u2_prev = u2_prev_init;

    let mut uo_cur = vec![0.0f64; nm_wide];
    let mut u2_cur = vec![0.0f64; nm_wide];

    let mut cur_val = vec![0.0f64; nm_wide];
    let mut delta = vec![0.0f64; nm_wide];

    for col in 0..total_cols {
        let t = col / NL_ITER;
        let i_in = col % NL_ITER;

        // `cur_val`/`delta` depend only on `t`, not `i_in`: recompute them
        // once per frame rather than on every one of the `NL_ITER` upsampled
        // steps within it.
        if i_in == 0 {
            for row in 0..nm_wide {
                let c = core_loudness[[row, t]];
                let next_val = if t + 1 < ntime {
                    core_loudness[[row, t + 1]]
                } else {
                    0.0
                };
                cur_val[row] = c;
                delta[row] = (next_val - c) / NL_ITER as f64;
            }
        }

        for row in 0..nm_wide {
            let ui = cur_val[row] + i_in as f64 * delta[row];

            let (o, u2) = nl_step(uo_prev[row], u2_prev[row], ui, &b);
            uo_cur[row] = o;
            u2_cur[row] = u2;
        }

        if i_in == 0 {
            for row in 0..nm_wide {
                output[[row, t]] = uo_cur[row];
            }
        }

        std::mem::swap(&mut uo_prev, &mut uo_cur);
        std::mem::swap(&mut u2_prev, &mut u2_cur);
    }

    output
}
