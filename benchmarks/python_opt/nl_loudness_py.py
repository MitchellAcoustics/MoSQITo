"""Pure-numpy rewrites of _nl_loudness, timed against the original."""
import time, sys
import numpy as np
from math import sqrt, exp
from mosqito.sq_metrics.loudness.loudness_zwtv._nonlinear_decay import _nl_loudness

def coeffs():
    nl_iter=24; sr=2000; t_short=0.005; t_long=0.015; t_var=0.075
    dt=1/(sr*nl_iter); P=(t_var+t_long)/(t_var*t_short); Q=1/(t_short*t_var)
    l1=-P/2+sqrt(P*P/4-Q); l2=-P/2-sqrt(P*P/4-Q); den=t_var*(l1-l2); e1=exp(l1*dt); e2=exp(l2*dt)
    return [(e1-e2)/den, ((t_var*l2+1)*e1-(t_var*l1+1)*e2)/den, ((t_var*l1+1)*e1-(t_var*l2+1)*e2)/den,
            (t_var*l1+1)*(t_var*l2+1)*(e1-e2)/den, exp(-dt/t_long), exp(-dt/t_var)]

def upsample(core, nl_iter=24):
    # identical construction to the original (iterative accumulation)
    delta = np.roll(core, -1, axis=1); delta[:, -1] = 0
    delta = (delta - core) / nl_iter
    ui = np.empty((core.shape[0], core.shape[1], nl_iter))
    ui[:, :, 0] = core
    for i in range(1, nl_iter):
        ui[:, :, i] = ui[:, :, i-1] + delta
    return ui.reshape(core.shape[0], -1)

def nl_numpy_v1(core):
    """Same per-column structure, but np.where instead of masked fancy assignment,
    fewer temporaries, state kept in two vectors instead of full matrices."""
    b0,b1,b2,b3,b4,b5 = coeffs()
    ui = upsample(core)
    nb, ncol = ui.shape
    out = np.empty_like(core)
    uo_prev = ui[:, -1].copy(); u2_prev = np.zeros(nb)
    for col in range(ncol):
        u = ui[:, col]
        gt = uo_prev > u2_prev
        cand = np.where(gt, uo_prev*b2 - u2_prev*b3, uo_prev*b4)
        uo = np.where(cand >= u, cand, u)
        u22 = uo_prev*b0 - u2_prev*b1
        u2 = np.where((u < uo_prev) & gt & (u22 <= uo), u22, uo)
        u2_2 = (u2_prev - u)*b5 + u
        m = (u >= uo_prev) & ~((np.abs(u - uo_prev) < 1e-5) & (uo <= u2_prev))
        u2 = np.where(m, u2_2, u2)
        if col % 24 == 0: out[:, col//24] = uo
        uo_prev = uo; u2_prev = u2
    return out

def nl_python_scalar(core):
    """Band-by-band scalar Python loop (no numpy in the inner loop)."""
    b0,b1,b2,b3,b4,b5 = coeffs()
    ui = upsample(core)
    nb, ncol = ui.shape
    out = np.empty_like(core)
    for r in range(nb):
        row = ui[r].tolist()
        uo_prev = row[-1]; u2_prev = 0.0
        o = []
        for col, u in enumerate(row):
            if uo_prev > u2_prev:
                c = uo_prev*b2 - u2_prev*b3
            else:
                c = uo_prev*b4
            uo = c if c >= u else u
            u2 = uo
            if u < uo_prev and uo_prev > u2_prev:
                u22 = uo_prev*b0 - u2_prev*b1
                if u22 <= uo: u2 = u22
            if u >= uo_prev and not (abs(u-uo_prev) < 1e-5 and uo <= u2_prev):
                u2 = (u2_prev - u)*b5 + u
            if col % 24 == 0: o.append(uo)
            uo_prev = uo; u2_prev = u2
        out[r] = o
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    for nt in [400, 2000]:
        core = np.abs(rng.standard_normal((21, nt))) * rng.uniform(0, 5, (21, 1)); core[:, ::3] *= 0.01
        t0=time.perf_counter(); ref=_nl_loudness(core.copy()); t_ref=time.perf_counter()-t0
        for name, fn in [("numpy_v1", nl_numpy_v1), ("python_scalar", nl_python_scalar)]:
            t0=time.perf_counter(); out=fn(core.copy()); t=time.perf_counter()-t0
            print(f"nt={nt}: {name:14s} {t*1e3:7.0f} ms vs original {t_ref*1e3:7.0f} ms -> {t_ref/t:4.1f}x  max|diff|={np.max(np.abs(out-ref)):.1e}")
