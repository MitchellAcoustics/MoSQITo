use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

const NL_ITER: usize = 24;

fn coeffs() -> [f64; 6] {
    let (t_short, t_long, t_var) = (0.005f64, 0.015f64, 0.075f64);
    let delta_t = 1.0 / (2000.0 * NL_ITER as f64);
    let p = (t_var + t_long) / (t_var * t_short);
    let q = 1.0 / (t_short * t_var);
    let l1 = -p / 2.0 + (p * p / 4.0 - q).sqrt();
    let l2 = -p / 2.0 - (p * p / 4.0 - q).sqrt();
    let den = t_var * (l1 - l2);
    let (e1, e2) = ((l1 * delta_t).exp(), (l2 * delta_t).exp());
    [
        (e1 - e2) / den,
        ((t_var * l2 + 1.0) * e1 - (t_var * l1 + 1.0) * e2) / den,
        ((t_var * l1 + 1.0) * e1 - (t_var * l2 + 1.0) * e2) / den,
        (t_var * l1 + 1.0) * (t_var * l2 + 1.0) * (e1 - e2) / den,
        (-delta_t / t_long).exp(),
        (-delta_t / t_var).exp(),
    ]
}

/// Direct transcription of the per-column mask logic in _nonlinear_decay.py,
/// including the col=-1 wraparound on the first step.
#[pyfunction]
fn nl_loudness<'py>(py: Python<'py>, core: PyReadonlyArray2<'py, f64>) -> Bound<'py, PyArray2<f64>> {
    let core = core.as_array();
    let (nb, nt) = (core.nrows(), core.ncols());
    let b = coeffs();
    let mut out = ndarray::Array2::<f64>::zeros((nb, nt));
    if nt == 0 { return out.into_pyarray(py); }
    let ncols = nt * NL_ITER;
    // ui_delta[:, k*24 + j] = core[:, k] + j * (core[:, k+1] - core[:, k]) / 24  (k+1 -> 0 at end)
    let ui = |row: usize, col: usize| -> f64 {
        let k = col / NL_ITER; let j = (col % NL_ITER) as f64;
        let c = core[[row, k]];
        let nx = if k + 1 < nt { core[[row, k + 1]] } else { 0.0 };
        c + j * ((nx - c) / NL_ITER as f64)
    };
    for row in 0..nb {
        // wraparound: uo_prev = uo_mat[:, -1] initial value = ui(last col); u2_prev = 0
        let mut uo_prev = ui(row, ncols - 1);
        let mut u2_prev = 0.0f64;
        for col in 0..ncols {
            let ui_c = ui(row, col);
            let mut uo_c = ui_c;
            let uo2 = uo_prev * b[2] - u2_prev * b[3];
            if uo_prev > u2_prev && uo2 >= ui_c { uo_c = uo2; }
            let uo2b = uo_prev * b[4];
            if uo_prev <= u2_prev && uo2b >= ui_c { uo_c = uo2b; }
            let mut u2_c = uo_c;
            let u22 = uo_prev * b[0] - u2_prev * b[1];
            if ui_c < uo_prev && uo_prev > u2_prev && u22 <= uo_c { u2_c = u22; }
            let u2_2 = (u2_prev - ui_c) * b[5] + ui_c;
            if ui_c >= uo_prev && !((ui_c - uo_prev).abs() < 1e-5 && uo_c <= u2_prev) { u2_c = u2_2; }
            if col % NL_ITER == 0 { out[[row, col / NL_ITER]] = uo_c; }
            uo_prev = uo_c; u2_prev = u2_c;
        }
    }
    out.into_pyarray(py)
}

#[pymodule]
fn mosqito_spike(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nl_loudness, m)?)?;
    Ok(())
}
