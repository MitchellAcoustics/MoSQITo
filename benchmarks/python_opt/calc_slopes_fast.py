"""Scalar grid-walk port of _calc_slopes (from the Rust branch's calc_slopes.rs), runnable as pure Python or under numba."""
import time, sys
import numpy as np
try:
    import numba; njit = numba.njit(cache=False)
except ImportError:
    numba = None; njit = lambda f: f
from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import _calc_slopes
from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness

ZUP = np.array([0.9,1.8,2.8,3.5,4.4,5.4,6.6,7.9,9.2,10.6,12.3,13.8,15.2,16.7,18.1,19.3,20.6,21.8,22.7,23.6,24.0])
RNS = np.array([21.5,18,15.1,11.5,9,6.1,4.4,3.1,2.13,1.36,0.82,0.42,0.30,0.22,0.15,0.10,0.035,0])
_usl = np.array([(13,8.2,6.3,5.5,5.5,5.5,5.5,5.5),(9,7.5,6,5.1,4.5,4.5,4.5,4.5),(7.8,6.7,5.6,4.9,4.4,3.9,3.9,3.9),(6.2,5.4,4.6,4.0,3.5,3.2,3.2,3.2),(4.5,3.8,3.6,3.2,2.9,2.7,2.7,2.7),(3.7,3.0,2.8,2.35,2.2,2.2,2.2,2.2),(2.9,2.3,2.1,1.9,1.8,1.7,1.7,1.7),(2.4,1.7,1.5,1.35,1.3,1.3,1.3,1.3),(1.95,1.45,1.3,1.15,1.1,1.1,1.1,1.1),(1.5,1.2,0.94,0.86,0.82,0.82,0.82,0.82),(0.72,0.67,0.64,0.63,0.62,0.62,0.62,0.62),(0.59,0.53,0.51,0.50,0.42,0.42,0.42,0.42),(0.40,0.33,0.26,0.24,0.24,0.22,0.22,0.22),(0.27,0.21,0.20,0.18,0.17,0.17,0.17,0.17),(0.16,0.15,0.14,0.12,0.11,0.11,0.11,0.11),(0.12,0.11,0.10,0.08,0.08,0.08,0.08,0.08),(0.09,0.08,0.07,0.06,0.06,0.06,0.06,0.05),(0.06,0.05,0.03,0.02,0.02,0.02,0.02,0.02)])
USL = _usl
ZUP_EA = (ZUP * 10).astype(np.int64)

@njit
def _r8(x): return np.rint(x * 1e8) / 1e8

@njit
def _rns_index(v, inclusive, RNS):
    c = 0
    for r in RNS:
        if inclusive:
            if _r8(v) <= _r8(r): c += 1
        else:
            if _r8(v) < _r8(r): c += 1
    return min(c, 17)

@njit
def _one(nm, spec, ZUP, ZUP_EA, RNS, USL):
    for i in range(240): spec[i] = 0.0
    prev = 0
    for i in range(20):
        end = min(ZUP_EA[i], 240)
        for k in range(prev, end): spec[k] = nm[i]
        prev = end
    n_total = 0.0; z1 = 0.0; n1 = 0.0
    for i in range(21):
        zup_i = ZUP[i]; zea = min(ZUP_EA[i], 240); zprev = 0 if i == 0 else ZUP_EA[i - 1]
        nm_i = nm[i]; col = max(i - 1, 0)
        if _r8(n1) <= _r8(nm_i):
            n_total += nm_i * (zup_i - z1); z1 = zup_i; n1 = nm_i
            continue
        row = _rns_index(n1, False, RNS)
        target = max(RNS[row], nm_i); usl = USL[row, min(col, 7)]
        sz1 = z1; sn1 = n1
        sz2 = (sn1 - target) / usl + sz1
        clipped = sz2 >= zup_i
        if clipped: sz2 = zup_i
        sn2 = sn1 - (sz2 - sz1) * usl if clipped else target
        n_total += (sz2 - sz1) * (sn1 + sn2) / 2.0
        reached = False
        for k in range(zprev, zea):
            z = (k + 1) * 0.1
            if _r8(sz2) <= _r8(z):
                if _r8(sn2) <= _r8(nm_i):
                    n_total += nm_i * (zup_i - sz2); reached = True
                    break
                row = _rns_index(sn2, True, RNS)
                target = max(RNS[row], nm_i); usl = USL[row, min(col, 7)]
                sz1 = sz2; sn1 = sn2
                sz2 = (sn1 - target) / usl + sz1
                clipped = sz2 >= zup_i
                if clipped: sz2 = zup_i
                sn2 = sn1 - (sz2 - sz1) * usl if clipped else target
                n_total += (sz2 - sz1) * (sn1 + sn2) / 2.0
            spec[k] = sn1 - (z - sz1) * usl
        if reached: z1 = zup_i; n1 = nm_i
        else: z1 = sz2; n1 = sn2
    n_total = max(n_total, 0.0)
    if n_total <= 16.0: n_total = np.floor(n_total * 1000.0 + 0.5) / 1000.0
    else: n_total = np.floor(n_total * 100.0 + 0.5) / 100.0
    return n_total

@njit
def _all(nm, N, spec, ZUP, ZUP_EA, RNS, USL):
    for f in range(nm.shape[1]):
        col = np.ascontiguousarray(nm[:, f]); out = np.empty(240)
        N[f] = _one(col, out, ZUP, ZUP_EA, RNS, USL)
        for k in range(240): spec[k, f] = out[k]

def calc_slopes_fast(nm, python=False):
    nm = np.ascontiguousarray(nm, dtype=np.float64)
    N = np.empty(nm.shape[1]); spec = np.empty((240, nm.shape[1]))
    fn = _all.py_func if (python and numba) else _all
    fn(nm, N, spec, ZUP, ZUP_EA, RNS, USL)
    return N, spec

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for label, n_frames in [("random 3rd-octave spectra", 2000)]:
        sp = rng.uniform(10, 100, (28, n_frames)); sp[:11] = np.minimum(sp[:11], 119)
        sp += rng.uniform(-15, 15, (28, 1))
        sp[:11] = np.minimum(sp[:11], 119)
        nm = _main_loudness(sp, "free")
        t0 = time.perf_counter(); N0, S0 = _calc_slopes(nm.copy()); t_ref = time.perf_counter() - t0
        calc_slopes_fast(nm[:, :5])                      # JIT warm-up
        t0 = time.perf_counter(); N1, S1 = calc_slopes_fast(nm); t_nb = time.perf_counter() - t0
        t0 = time.perf_counter(); N2, S2 = calc_slopes_fast(nm[:, :200], python=True); t_py = (time.perf_counter() - t0) * n_frames / 200
        bad = np.sum(~np.isclose(S0, S1, rtol=0, atol=1e-9)) ; badN = np.sum(N0 != N1)
        print(f"{label}: {n_frames} frames  original {t_ref:.3f}s | numba {t_nb*1e3:.1f} ms ({t_ref/t_nb:.0f}x) | pure-python scalar (extrapolated) {t_py:.2f}s ({t_ref/t_py:.1f}x)")
        print(f"   mismatching specific-loudness cells (atol 1e-9): {bad} of {S0.size}; mismatching N: {badN} of {len(N0)}; max|dS|={np.max(np.abs(S0-S1)):.2e}")
        print(f"   pure-python-scalar vs numba identical on first 200 frames: {np.array_equal(S2, S1[:, :200])}")
