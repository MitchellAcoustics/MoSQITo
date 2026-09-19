//! ECMA-418-2 (2nd Ed, 2022) §5.1.8 Eq. 23: compressive nonlinearity of the
//! auditory system.

const P_0: f64 = 2e-5;
const C_N: f64 = 0.0211668;
const ALPHA: f64 = 1.50;

const V_I: [f64; 9] = [
    1.0, 0.6602, 0.0864, 0.6384, 0.0328, 0.4068, 0.2082, 0.3994, 0.6434,
];
const THRESH: [f64; 9] = [0.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0];

/// Applies Eq. 23 to a single rectified band-pass block's RMS value,
/// returning the specific loudness `a'` before the absolute-threshold clip.
pub fn nonlinearity(p: f64) -> f64 {
    let mut a_prime = 1.0;
    for i in 1..9 {
        let p_ti = P_0 * 10f64.powf(THRESH[i] / 20.0);
        a_prime *= (1.0 + (p / p_ti).powf(ALPHA)).powf((V_I[i] - V_I[i - 1]) / ALPHA);
    }
    a_prime * C_N * p / P_0
}
