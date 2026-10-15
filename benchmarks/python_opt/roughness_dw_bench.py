import time, numpy as np
from concurrent.futures import ThreadPoolExecutor
from mosqito.sq_metrics import roughness_dw
from mosqito.utils.time_segmentation import time_segmentation
from mosqito.sound_level_meter.comp_spectrum import comp_spectrum
from roughness_dw_fast import main_calc_fast, _gzi_weighting, _H_weighting

def roughness_dw_fast(signal, fs, overlap=0.5, workers=4):
    nperseg = int(0.2 * fs); noverlap = int(overlap * nperseg)
    sig, time_ax = time_segmentation(signal, fs, nperseg=nperseg, noverlap=noverlap, is_ecma=False)
    nseg = 1 if sig.ndim == 1 else sig.shape[1]
    spec, _ = comp_spectrum(sig, fs, nfft="default", window="blackman", db=False)
    freq_axis = np.arange(1, nperseg // 2 + 1) * (fs / nperseg)
    hW = _H_weighting(nperseg, fs); gzi = _gzi_weighting(np.arange(1, 48) / 2)
    if spec.ndim == 1:
        return main_calc_fast(spec, freq_axis, fs, gzi, hW)[0], None, None, time_ax
    def one(i): return main_calc_fast(spec[:, i].copy(), freq_axis, fs, gzi, hW)
    with ThreadPoolExecutor(workers) as ex: res = list(ex.map(one, range(nseg)))
    R = np.array([r[0] for r in res]); Rs = np.stack([r[1] for r in res], axis=1)
    return R, Rs, res[0][2], time_ax

FS = 48000; rng = np.random.default_rng(0)
for dur in [1.0, 5.0]:
    t = np.arange(int(FS*dur))/FS
    for name, sig in [("AM tone", 0.05*(1+np.sin(2*np.pi*70*t))*np.sin(2*np.pi*1000*t)), ("noise", 0.02*rng.standard_normal(len(t)))]:
        t0=time.perf_counter(); ref=roughness_dw(sig.copy(), FS); t_ref=time.perf_counter()-t0
        row=[]
        for w in (1,4):
            ts=[]
            for _ in range(2):
                t0=time.perf_counter(); out=roughness_dw_fast(sig.copy(), FS, workers=w); ts.append(time.perf_counter()-t0)
            row.append(min(ts))
        print(f"{dur:3.0f}s {name:8s} original {t_ref:6.2f}s -> batched {row[0]:5.2f}s ({t_ref/row[0]:4.1f}x), +4 threads {row[1]:5.2f}s ({t_ref/row[1]:4.1f}x)  max|dR|={np.max(np.abs(ref[0]-out[0])):.1e} max|dRspec|={np.max(np.abs(ref[1]-out[1])):.1e}")
