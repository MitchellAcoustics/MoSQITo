import time, numpy as np, numba
from nl_loudness_py import coeffs, upsample, nl_python_scalar
from mosqito.sq_metrics.loudness.loudness_zwtv._nonlinear_decay import _nl_loudness

@numba.njit(cache=False)
def _kernel(ui, b, out):
    nb, ncol = ui.shape
    for r in range(nb):
        uo_prev = ui[r, ncol-1]; u2_prev = 0.0
        for col in range(ncol):
            u = ui[r, col]
            if uo_prev > u2_prev: c = uo_prev*b[2] - u2_prev*b[3]
            else: c = uo_prev*b[4]
            uo = c if c >= u else u
            u2 = uo
            if u < uo_prev and uo_prev > u2_prev:
                u22 = uo_prev*b[0] - u2_prev*b[1]
                if u22 <= uo: u2 = u22
            if u >= uo_prev and not (abs(u-uo_prev) < 1e-5 and uo <= u2_prev):
                u2 = (u2_prev - u)*b[5] + u
            if col % 24 == 0: out[r, col//24] = uo
            uo_prev = uo; u2_prev = u2

def nl_loudness_numba(core):
    ui = upsample(core); out = np.empty_like(core)
    _kernel(ui, np.array(coeffs()), out); return out

rng = np.random.default_rng(1)
core = np.abs(rng.standard_normal((21, 2000))) * rng.uniform(0, 5, (21, 1)); core[:, ::3] *= 0.01
t0=time.perf_counter(); nl_loudness_numba(core); print(f"numba first call (incl. JIT compile): {time.perf_counter()-t0:.2f} s")
ref=_nl_loudness(core.copy())
for name, fn in [("original", _nl_loudness), ("python_scalar", nl_python_scalar), ("numba", nl_loudness_numba)]:
    ts=[]
    for _ in range(3):
        t0=time.perf_counter(); out=fn(core.copy()); ts.append(time.perf_counter()-t0)
    print(f"{name:14s} {min(ts)*1e3:8.1f} ms   max|diff|={np.max(np.abs(out-ref)):.1e}")
