import time, functools, numpy as np, sys, mosqito
from scipy.signal import butter
from mosqito.sq_metrics import loudness_zwst, sharpness_din_st
from mosqito.sound_level_meter import noct_spectrum
M = sys.modules["mosqito.sound_level_meter.noct_spectrum._n_oct_time_filter"]
FS=48000; sig = 0.02*np.random.default_rng(0).standard_normal(FS)
cases=[("noct_spectrum", noct_spectrum, (sig, FS, 24, 12500)), ("loudness_zwst", loudness_zwst, (sig, FS)), ("sharpness_din_st", sharpness_din_st, (sig, FS))]
def bench(fn, *a):
    ts=[]
    for _ in range(7):
        t0=time.perf_counter(); r=fn(*a); ts.append(time.perf_counter()-t0)
    return min(ts), np.asarray(r[0] if isinstance(r, tuple) else r)
base={n:bench(f,*a) for n,f,a in cases}
orig=M.butter
M.butter=functools.lru_cache(maxsize=None)(orig)
new={n:bench(f,*a) for n,f,a in cases}
for n in base: print(f"{n:18s} {base[n][0]*1e3:6.1f} ms -> {new[n][0]*1e3:6.1f} ms ({base[n][0]/new[n][0]:.2f}x)  identical={np.array_equal(base[n][1],new[n][1])}")
# what is left?
import cProfile, pstats, io
pr=cProfile.Profile(); pr.enable(); noct_spectrum(sig,FS,24,12500); pr.disable()
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats("tottime").print_stats(5); print("\n".join(s.getvalue().splitlines()[6:13]))
