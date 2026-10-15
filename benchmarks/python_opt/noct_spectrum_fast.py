import time, functools, numpy as np, sys, mosqito
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import butter
from mosqito.sound_level_meter import noct_spectrum
from mosqito.sound_level_meter.noct_spectrum._center_freq import _center_freq
from mosqito.sound_level_meter.noct_spectrum._filter_bandwidth import _filter_bandwidth
M = sys.modules["mosqito.sound_level_meter.noct_spectrum._n_oct_time_filter"]
_butter = functools.lru_cache(maxsize=None)(butter)
def noct_fast(sig, fs, fmin, fmax, n=3, G=10, fr=1000, workers=4):
    if sig.ndim == 1: sig = sig[:, None]
    fc_vec, fpref = _center_freq(fmin=fmin, fmax=fmax, n=n, G=G, fr=fr)
    alpha_vec, _, _ = _filter_bandwidth(fc_vec, n=n)
    M.butter = _butter
    with ThreadPoolExecutor(workers) as ex:
        spec = list(ex.map(lambda p: M._n_oct_time_filter(sig, fs, p[0], p[1]), zip(fc_vec, alpha_vec)))
    return np.squeeze(np.array(spec)), fpref
FS = 48000
for dur in [1.0, 10.0]:
    sig = 0.02 * np.random.default_rng(0).standard_normal(int(FS * dur))
    M.butter = butter
    ts = []; 
    for _ in range(5): t0 = time.perf_counter(); ref = noct_spectrum(sig, FS, 24, 12500); ts.append(time.perf_counter() - t0)
    t_ref = min(ts)
    out = {}
    for w in (1, 4):
        ts = []
        for _ in range(5): t0 = time.perf_counter(); r = noct_fast(sig, FS, 24, 12500, workers=w); ts.append(time.perf_counter() - t0)
        out[w] = min(ts)
    print(f"noct_spectrum {dur:4.0f}s: original {t_ref*1e3:7.1f} ms -> design cache {out[1]*1e3:7.1f} ms ({t_ref/out[1]:.1f}x), + 4 threads {out[4]*1e3:7.1f} ms ({t_ref/out[4]:.1f}x), identical={np.array_equal(ref[0], r[0])}")
