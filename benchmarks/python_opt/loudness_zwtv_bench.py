import sys, time, numpy as np
import mosqito
from mosqito.sq_metrics import loudness_zwtv, sharpness_din_tv
ZT = sys.modules["mosqito.sq_metrics.loudness.loudness_zwtv.loudness_zwtv"]
import nl_loudness_py, nl_loudness_numba, calc_slopes_fast
try:
    import mosqito_spike; HAVE_RUST = True
except ImportError: HAVE_RUST = False

orig_nl, orig_slopes = ZT._nl_loudness, ZT._calc_slopes
def nl_numba_fn(core): return nl_loudness_numba.nl_loudness_numba(np.ascontiguousarray(core))
def slopes_numba(nm):
    N, S = calc_slopes_fast.calc_slopes_fast(nm); return N, S
def slopes_py(nm): return calc_slopes_fast.calc_slopes_fast(nm, python=True)
# NB: loudness_zwtv uses `loudness, spec_loudness = _calc_slopes(core_loudness)`
variants = {
  "original": (orig_nl, orig_slopes),
  "python refactor (scalar loops)": (nl_loudness_py.nl_python_scalar, slopes_py),
  "+ numba": (nl_numba_fn, slopes_numba),
}
if HAVE_RUST: variants["rust nl_loudness only"] = (lambda c: mosqito_spike.nl_loudness(np.ascontiguousarray(c)), orig_slopes)
if HAVE_RUST: variants["rust nl_loudness + numba slopes"] = (lambda c: mosqito_spike.nl_loudness(np.ascontiguousarray(c)), slopes_numba)
FS = 48000; rng = np.random.default_rng(0)
nl_loudness_numba.nl_loudness_numba(np.ones((21, 4))); calc_slopes_fast.calc_slopes_fast(np.ones((21, 4)))
for dur in [1.0, 5.0]:
    sig = 0.02 * rng.standard_normal(int(FS * dur))
    ref = None
    print(f"\n--- loudness_zwtv, {dur:.0f} s ---")
    for name, (nl, sl) in variants.items():
        ZT._nl_loudness, ZT._calc_slopes = nl, sl
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); out = loudness_zwtv(sig.copy(), FS); ts.append(time.perf_counter() - t0)
        if ref is None: ref = out; t_ref = min(ts)
        dN = np.max(np.abs(out[0] - ref[0])); dS = np.max(np.abs(out[1] - ref[1]))
        print(f"{name:34s} {min(ts):6.2f} s  ({t_ref/min(ts):5.1f}x)   max|dN|={dN:.1e} max|dN_spec|={dS:.1e}")
ZT._nl_loudness, ZT._calc_slopes = orig_nl, orig_slopes

# stage floor with numba variant
ST = {}
for nm_ in ["_third_octave_levels", "_main_loudness", "_nl_loudness", "_calc_slopes", "_temporal_weighting"]:
    f = getattr(ZT, nm_)
    def w(*a, _f=f, _n=nm_, **k):
        t0 = time.perf_counter(); r = _f(*a, **k); ST[_n] = ST.get(_n, 0) + time.perf_counter() - t0; return r
    setattr(ZT, nm_, w)
ZT._nl_loudness_w = ZT._nl_loudness
orig_w = ZT._nl_loudness
ZT._nl_loudness = lambda c: (lambda t0: (ST.__setitem__("_nl_loudness", ST.get("_nl_loudness", 0) + 0), nl_numba_fn(c))[1])(0)
ZT._calc_slopes = slopes_numba
sig = 0.02 * rng.standard_normal(FS * 5)
t0 = time.perf_counter(); loudness_zwtv(sig, FS); tot = time.perf_counter() - t0
print(f"\nfloor with numba stages (5 s, total {tot:.2f} s):", ", ".join(f"{k} {v:.2f}s" for k, v in ST.items()))
