"""Per-band, threaded refactor of roughness_ecma. Same maths and same SciPy calls, applied to one band at a time."""
import sys, time, tracemalloc
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import sosfilt, lfilter, hilbert, decimate
from scipy.fft import fft
import mosqito
from mosqito.sq_metrics import roughness_ecma
from mosqito.sq_metrics.loudness.loudness_ecma._preprocessing import _preprocessing
from mosqito.sq_metrics.loudness.loudness_ecma._ear_filter_design import _ear_filter_design
from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import _auditory_filters_centre_freq
from mosqito.sq_metrics.loudness.loudness_ecma._gammatone import _gammatone
from mosqito.sq_metrics.loudness.loudness_ecma._nonlinearity import _nonlinearity
from mosqito.sq_metrics.loudness.loudness_ecma._loudness_ecma_data import ltq_z
from mosqito.sq_metrics.roughness.roughness_ecma._weighting import _f_max, _r_max, _Q2_high, _Q2_low, _high_mod_rate_weighting, _low_mod_rate_weighting
from mosqito.sq_metrics.roughness.roughness_ecma._estimate_fund_mod_rate import _estimate_fund_mod_rate
from mosqito.sq_metrics.roughness.roughness_ecma._peak_picking import _peak_picking
from mosqito.sq_metrics.roughness.roughness_ecma._von_hann_window import _von_hann_window
from mosqito.sq_metrics.roughness.roughness_ecma._noise_reduction import _noise_reduction
from mosqito.sq_metrics.roughness.roughness_ecma._interpolation_50 import _interpolation_50
from mosqito.sq_metrics.roughness.roughness_ecma._non_linear_transform import _non_linear_transform
from mosqito.sq_metrics.roughness.roughness_ecma._lowpass_filter import _lowpass_filter

STAGES = {}
def _tic(): return time.perf_counter()
def _acc(k, t0): STAGES[k] = STAGES.get(k, 0.0) + time.perf_counter() - t0

def front_end(signal, workers):
    """Everything up to envelopes_downsampled (L, 53, 512), N_specific (L, 53), time_axis."""
    sb, sh = 16384, 4096
    t0 = _tic()
    signal, n_new = _preprocessing(signal, sb, sh)
    sig_ear = sosfilt(_ear_filter_design(), signal)
    L = np.arange(int(np.ceil((n_new + sh) / sh) - 1))
    idx = np.linspace(L * sh, L * sh + sb, sb).astype(np.int32).T          # (L, sb) – same gather as the original
    time_vec = np.linspace(0, (len(sig_ear) - 1) / 48000, len(sig_ear))
    time_axis = np.mean(time_vec[idx], axis=1)                              # band 0 only: all bands share it
    cf = _auditory_filters_centre_freq()
    _acc("prep", t0)

    def band(z):
        b, a = _gammatone(cf[z], k=5, fs=48000)
        bp = 2.0 * lfilter(b, a, sig_ear).real
        blocks = bp[idx]
        rms = np.sqrt(2 * np.mean(np.clip(blocks, 0, None) ** 2, axis=1))
        a_prime = _nonlinearity(rms)
        a_prime[a_prime < ltq_z[z]] = ltq_z[z]
        env = np.abs(hilbert(blocks, axis=-1))
        env = decimate(decimate(env, 8, axis=-1), 4, axis=-1)
        return a_prime - ltq_z[z], env

    t0 = _tic()
    if workers == 1: res = [band(z) for z in range(53)]
    else:
        with ThreadPoolExecutor(workers) as ex: res = list(ex.map(band, range(53)))
    _acc("per-band front end", t0)
    N_specific = np.stack([r[0] for r in res], axis=1)                       # (L, 53)
    env_ds = np.stack([r[1] for r in res], axis=1)                           # (L, 53, 512)
    return env_ds, N_specific, time_axis, len(signal)

def roughness_ecma_fast(signal, fs, workers=4):
    STAGES.clear()
    signal = np.array(signal, dtype=float)
    duration = len(signal) / fs
    env_ds, N_specific, time_axis, _ = front_end(signal, workers)
    bark_axis = np.linspace(0.5, 26.5, num=53, endpoint=True)
    CBF = 53; sbb = 512; L = N_specific.shape[0]
    center_freq = _auditory_filters_centre_freq()
    t0 = _tic()
    N_specific_max = N_specific.max(axis=1)
    hann = _von_hann_window(sbb)
    phi_E0 = np.sum(np.power(env_ds * hann, 2), axis=2)
    den = N_specific_max[:, None] * phi_E0
    dft = (abs(fft(env_ds * hann, axis=2)[:, :, :sbb // 2]) / 2 * np.sqrt(2)) ** 2
    scaling = np.zeros((L, CBF)); scaling[den != 0] = np.power(N_specific[den != 0], 2) / den[den != 0]
    Phi_E = _noise_reduction(scaling[:, :, None] * dft)
    _acc("spectra + noise reduction", t0)
    t0 = _tic()
    fmax = _f_max(center_freq); rmax = _r_max(center_freq); q2h = _Q2_high(center_freq); q2l = _Q2_low(center_freq)
    amplitude = np.zeros((L, CBF))
    for l in range(L):
        for z in range(CBF):
            f_p, Ai = _peak_picking(Phi_E[l, z, :])
            if len(f_p) == 0: continue
            Ai_t = np.array([_high_mod_rate_weighting(f_p[i], Ai[i], fmax[z], rmax[z], q2h[z]) for i in range(len(f_p))])
            mod_rate, A_hat = _estimate_fund_mod_rate(f_p, Ai_t)
            amplitude[l, z] = _low_mod_rate_weighting(mod_rate, A_hat, fmax[z], q2l[z])
    _acc("peak picking + weighting", t0)
    amplitude[amplitude < 0.074376] = 0
    t0 = _tic()
    a50, t_50 = _interpolation_50(amplitude, time_axis, duration)
    R_spec_t = _lowpass_filter(_non_linear_transform(np.clip(a50, 0, None)))
    R_spec = np.mean(R_spec_t[10:, :], axis=0); R_time = 0.5 * np.sum(R_spec_t, axis=1); R = np.percentile(R_time, 90)
    _acc("tail", t0)
    return R, R_time, R_spec, bark_axis, t_50

if __name__ == "__main__":
    FS = 48000
    durs = [float(d) for d in sys.argv[1:]] or [1.0, 5.0]
    for dur in durs:
        t = np.arange(int(FS * dur)) / FS
        sig = 0.05 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)
        ts_ref=[]
        for _ in range(3):
            t0 = time.perf_counter(); ref = roughness_ecma(sig.copy(), FS); ts_ref.append(time.perf_counter() - t0)
        t_ref = min(ts_ref)
        tracemalloc.start(); roughness_ecma(sig.copy(), FS); m_ref = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
        row = {}
        for w in (1, 4):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); out = roughness_ecma_fast(sig.copy(), FS, workers=w); ts.append(time.perf_counter() - t0)
            row[w] = (min(ts), dict(STAGES))
        tracemalloc.start(); roughness_ecma_fast(sig.copy(), FS, workers=1); m_new = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
        runs = '/'.join('%.1f' % x for x in ts_ref)
        d = max(np.max(np.abs(ref[1] - out[1])), np.max(np.abs(ref[2] - out[2])), abs(ref[0] - out[0]))
        print(f"\n{dur:.0f} s: original {t_ref:6.2f}s (runs {runs}) {m_ref:5.0f} MB | refactor 1 thread {row[1][0]:5.2f}s ({t_ref/row[1][0]:.1f}x) {m_new:4.0f} MB | 4 threads {row[4][0]:5.2f}s ({t_ref/row[4][0]:.1f}x) | max|diff| {d:.1e}  R {ref[0]:.5f} vs {out[0]:.5f}")
        print("   stages (4 threads): " + ", ".join(f"{k} {v:.2f}s" for k, v in row[4][1].items()))
