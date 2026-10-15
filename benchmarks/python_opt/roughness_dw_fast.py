"""Batched rewrite of _roughness_dw_main_calc (Daniel & Weber): same math, no per-component Python loops."""
import math, time, sys
import numpy as np
import scipy.fft as sfft
import numpy.fft as nfft_mod
from mosqito.utils.LTQ import LTQ
from mosqito.utils import freq2bark, db2amp, amp2db, bark2freq
from mosqito.sq_metrics.roughness.roughness_dw._ear_filter_coeff import _ear_filter_coeff
from mosqito.sq_metrics.roughness.roughness_dw._roughness_dw_main_calc import _roughness_dw_main_calc
from mosqito.sq_metrics.roughness.roughness_dw._gzi_weighting import _gzi_weighting
from mosqito.sq_metrics.roughness.roughness_dw._H_weighting import _H_weighting

def main_calc_fast(spec, freq_axis, fs, gzi, hWeight, workers=1, backend='numpy'):
    F = nfft_mod if backend == 'numpy' else sfft
    kw = {} if backend == 'numpy' else {'workers': workers}
    n_in = len(spec)
    spec = np.concatenate((spec, spec[len(spec)::-1]))
    n = len(spec)
    bark_axis = freq2bark(freq_axis)
    nZ = np.arange(1, n // 2 + 1)
    a0 = np.zeros(n); a0[nZ - 1] = db2amp(_ear_filter_coeff(bark_axis), ref=1)
    spec = a0 * spec
    module = np.abs(spec[: n // 2])
    spec_dB = amp2db(module, ref=2e-5)
    threshold = LTQ(bark_axis, reference="roughness")
    aud = np.where(spec_dB > threshold)[0]
    n_aud = len(aud)
    n_ch = 47
    lev = spec_dB[aud]; b = bark_axis[aud]
    s1 = -27.0
    s2 = np.minimum(-24 - 230 / freq_axis[aud] + 0.2 * lev, 0)
    zi = np.arange(1, n_ch + 1) / 2
    zb = bark2freq(zi) * n / fs
    minExcitDB = np.interp(zb, nZ, threshold)
    ch_low = np.floor(2 * b) - 1
    ch_high = np.ceil(2 * b) - 1
    j = np.arange(n_ch)[None, :]                      # (1, 47)
    centre = (j + 1) * 0.5
    sl_low = s1 * (b[:, None] - centre) + lev[:, None]            # (n_aud, 47)
    sl_high = s2[:, None] * (centre - b[:, None]) + lev[:, None]
    slopes = np.zeros((n_aud, n_ch))
    m_low = (j <= ch_low[:, None]) & (sl_low > minExcitDB[None, :])
    slopes[m_low] = db2amp(sl_low[m_low], ref=0.00002)
    m_high = (j >= ch_high[:, None]) & (sl_high > minExcitDB[None, :])
    slopes[m_high] = db2amp(sl_high[m_high], ref=0.00002)
    # amplitude of each component in each channel (rows = channels)
    i = np.arange(n_ch)[:, None]                      # (47, 1)
    up = np.take_along_axis(slopes.T, np.broadcast_to(np.minimum(i + 1, n_ch - 1), (n_ch, n_aud)), axis=0)
    dn = np.take_along_axis(slopes.T, np.broadcast_to(i - 1, (n_ch, n_aud)), axis=0)
    mod_a = module[aud][None, :]
    ampl = np.where((ch_low[None, :] == i) | (ch_high[None, :] == i), 1.0,
                    np.where(ch_high[None, :] > i, up / mod_a, dn / mod_a))
    exc = np.zeros((n_ch, n), dtype=complex)
    exc[:, aud] = ampl * spec[aud][None, :]
    te = np.abs(n * np.real(F.ifft(exc, axis=1, **kw)))
    h0 = te.mean(axis=1)
    env = F.fft(te - h0[:, None], axis=1, **kw) * hWeight
    hBP = 2 * np.real(F.ifft(env, axis=1, **kw))
    rms = np.sqrt(np.mean(hBP ** 2, axis=1))
    mod_depth = np.where(h0 > 0, np.minimum(rms / np.where(h0 > 0, h0, 1), 1.0), 0.0)
    nz = (hBP != 0).all(axis=1)
    c = hBP - hBP.mean(axis=1, keepdims=True)
    ss = np.sum(c * c, axis=1)
    ki = np.zeros(47)
    num = np.sum(c[:45] * c[2:47], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / np.sqrt(ss[:45] * ss[2:47])
    ok = nz[:45] & nz[2:47]
    ki[:45] = np.where(ok, np.clip(r, -1, 1), 0.0)
    R_spec = np.empty(47)
    R_spec[0] = gzi[0] * (mod_depth[0] * ki[0]) ** 2
    R_spec[1] = gzi[1] * (mod_depth[1] * ki[1]) ** 2
    R_spec[2:45] = gzi[2:45] * (mod_depth[2:45] * ki[2:45] * ki[0:43]) ** 2
    R_spec[45] = gzi[45] * (mod_depth[45] * ki[43]) ** 2
    R_spec[46] = gzi[46] * (mod_depth[46] * ki[44]) ** 2
    return 0.25 * R_spec.sum(), R_spec, zi

if __name__ == "__main__":
    from mosqito.utils.time_segmentation import time_segmentation
    from mosqito.sound_level_meter.comp_spectrum import comp_spectrum
    FS = 48000; nperseg = int(0.2 * FS)
    rng = np.random.default_rng(0)
    t = np.arange(FS) / FS
    sigs = {"AM tone": 0.05 * (1 + np.sin(2*np.pi*70*t)) * np.sin(2*np.pi*1000*t),
            "noise": 0.02 * rng.standard_normal(FS),
            "2 tones": 0.03*np.sin(2*np.pi*500*t) + 0.03*np.sin(2*np.pi*600*t)}
    gzi = _gzi_weighting(np.arange(1, 48) / 2); hW = _H_weighting(nperseg, FS)
    freq_axis = np.arange(1, nperseg // 2 + 1) * (FS / nperseg)
    for name, sig in sigs.items():
        blocks, _ = time_segmentation(sig, FS, nperseg=nperseg, noverlap=nperseg // 2)
        spec, _ = comp_spectrum(blocks, FS, nfft="default", window="blackman", db=False)
        col = spec[:, 0].copy()
        t0 = time.perf_counter(); R0, Rs0, _ = _roughness_dw_main_calc(col.copy(), freq_axis, FS, gzi, hW); t_ref = time.perf_counter() - t0
        for w in ('numpy', 'scipy'):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); R1, Rs1, _ = main_calc_fast(col.copy(), freq_axis, FS, gzi, hW, backend=w); ts.append(time.perf_counter() - t0)
            print(f"{name:8s} fft={w}: original {t_ref*1e3:6.1f} ms -> {min(ts)*1e3:6.1f} ms ({t_ref/min(ts):.1f}x)  R {R0:.6f} vs {R1:.6f}  max|dRspec|={np.max(np.abs(Rs0-Rs1)):.1e}")
