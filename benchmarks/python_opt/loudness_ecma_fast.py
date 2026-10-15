"""Pure-numpy refactor of loudness_ecma's block stage: cumsum-based block RMS, no block array."""
import time, tracemalloc, sys
import numpy as np, scipy.signal as sps
import mosqito
from mosqito.sq_metrics import loudness_ecma
from mosqito.sq_metrics.loudness.loudness_ecma._preprocessing import _preprocessing
from mosqito.sq_metrics.loudness.loudness_ecma._ear_filter_design import _ear_filter_design
from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import _auditory_filters_centre_freq
from mosqito.sq_metrics.loudness.loudness_ecma._gammatone import _gammatone
from mosqito.sq_metrics.loudness.loudness_ecma._nonlinearity import _nonlinearity
from mosqito.sq_metrics.loudness.loudness_ecma._loudness_ecma_data import ltq_z

def loudness_ecma_fast(signal, fs, sb=2048, sh=1024, workers=1, method="gather"):
    """method='gather': per-band block gather (same samples and summation as the original, exact parity).
    method='cumsum': running-sum differences (faster, but loses precision after a loud passage)."""
    from concurrent.futures import ThreadPoolExecutor
    signal, n_new = _preprocessing(signal, sb, sh)
    sig_ear = sps.sosfilt(_ear_filter_design(), signal)
    cf = _auditory_filters_centre_freq()
    L_last = int(np.ceil((n_new + sh) / sh) - 1)
    starts = np.arange(L_last) * sh
    def band(z):
        b, a = _gammatone(cf[z], k=5, fs=48000)
        bp = 2.0 * sps.lfilter(b, a, sig_ear).real
        if method == "cumsum":
            sq = np.square(np.clip(bp, 0, None))
            S = np.concatenate(([0.0], np.cumsum(sq)))
            rms = np.sqrt(2 * (S[starts + sb - 1] - S[starts] + sq[starts + sb]) / sb)   # = the block the original gathers
        else:
            idx = np.linspace(starts, starts + sb, sb).astype(np.int32).T               # the original's exact gather
            rms = np.sqrt(2 * np.mean(np.clip(bp[idx], 0, None) ** 2, axis=1))
        a_prime = _nonlinearity(rms)
        a_prime[a_prime < ltq_z[z]] = ltq_z[z]
        return a_prime - ltq_z[z]
    if workers == 1: rows = [band(z) for z in range(53)]
    else:
        with ThreadPoolExecutor(workers) as ex: rows = list(ex.map(band, range(53)))
    N_spec = np.array(rows)
    N_time = np.sum(N_spec, axis=0) * 0.5
    e = 1 / np.log10(2)
    return np.mean(N_time ** e) ** (1 / e), N_time, N_spec

FS = 48000
def best(fn, sig, reps=3):
    ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); r=fn(sig.copy(), FS); ts.append(time.perf_counter()-t0)
    return min(ts), r
def peak(fn, sig):
    tracemalloc.start(); fn(sig.copy(), FS); _,m=tracemalloc.get_traced_memory(); tracemalloc.stop(); return m/1e6
def main():
    for dur in [1.0, 5.0, 20.0]:
        t = np.arange(int(FS*dur))/FS
        sig = 0.05*(1+np.sin(2*np.pi*70*t))*np.sin(2*np.pi*1000*t)
        t_ref, ref = best(loudness_ecma, sig); t_new, out = best(loudness_ecma_fast, sig); t_thr, out2 = best(lambda x, f: loudness_ecma_fast(x, f, workers=4), sig); t_cs, _ = best(lambda x, f: loudness_ecma_fast(x, f, method='cumsum'), sig)
        rel = max(np.max(np.abs(np.asarray(ref[2])-out[2])/(np.abs(np.asarray(ref[2]))+1e-12)), abs(ref[0]-out[0])/abs(ref[0]))
        print(f"loudness_ecma {dur:4.0f}s: original {t_ref:6.2f}s {peak(loudness_ecma,sig):6.0f} MB -> gather {t_new:5.2f}s (cumsum {t_cs:5.2f}s) {peak(loudness_ecma_fast,sig):4.0f} MB  ({t_ref/t_new:.1f}x), 4 threads {t_thr:5.2f}s ({t_ref/t_thr:.1f}x)  max rel diff {rel:.1e}")


def dynamic_range_check():
    """Loud 1 s burst followed by 19 s at -100 dB: where a running-sum difference loses precision."""
    rng = np.random.default_rng(0); t = np.arange(20 * FS) / FS
    sig = 0.05 * np.sin(2 * np.pi * 1000 * t); sig[FS:] *= 1e-5
    ref = loudness_ecma(sig.copy(), FS)
    for m in ("gather", "cumsum"):
        out = loudness_ecma_fast(sig.copy(), FS, method=m)
        rel = np.max(np.abs(np.asarray(ref[2]) - out[2]) / (np.abs(np.asarray(ref[2])) + 1e-12))
        print(f"dynamic-range signal, method={m:7s}: max relative error in specific loudness {rel:.1e}; N {out[0]:.6g} vs {ref[0]:.6g}")


if __name__ == "__main__":
    dynamic_range_check()
    main()
