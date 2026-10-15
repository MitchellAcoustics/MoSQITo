import inspect, numpy as np, sys, mosqito
import roughness_dw_fast as D
import importlib; OM = importlib.import_module("mosqito.sq_metrics.roughness.roughness_dw._roughness_dw_main_calc")
src = inspect.getsource(OM)
src = src.replace("return R, R_spec, zi", "return R, R_spec, zi, dict(slopes=slopes, hBP=hBP, mod_depth=mod_depth, ki=ki, aud=audible_index, ch_low=ch_low, ch_high=ch_high)")
ns = dict(vars(OM)); exec(compile(src, "orig_dbg", "exec"), ns)
orig = ns["_roughness_dw_main_calc"]
from mosqito.utils.time_segmentation import time_segmentation
from mosqito.sound_level_meter.comp_spectrum import comp_spectrum
FS=48000; nperseg=int(0.2*FS); t=np.arange(FS)/FS
sig = 0.03*np.sin(2*np.pi*500*t)+0.03*np.sin(2*np.pi*600*t)
blocks,_ = time_segmentation(sig, FS, nperseg=nperseg, noverlap=nperseg//2)
spec,_ = comp_spectrum(blocks, FS, nfft="default", window="blackman", db=False)
col = spec[:,0].copy()
gzi = D._gzi_weighting(np.arange(1,48)/2); hW = D._H_weighting(nperseg, FS); fa = np.arange(1,nperseg//2+1)*(FS/nperseg)
R0,Rs0,_,dbg = orig(col.copy(), fa, FS, gzi, hW)
print("n_aud", len(dbg["aud"]), "ch_low unique", np.unique(dbg["ch_low"]), "ch_high unique", np.unique(dbg["ch_high"]))
print("orig mod_depth", np.round(dbg["mod_depth"][:30],4))
print("orig ki       ", np.round(dbg["ki"][:30],4))
nz = (dbg["hBP"]!=0).all(axis=1); print("rows all nonzero:", nz.astype(int))

fsrc = inspect.getsource(D.main_calc_fast).replace("return 0.25 * R_spec.sum(), R_spec, zi", "return 0.25 * R_spec.sum(), R_spec, zi, dict(slopes=slopes, hBP=hBP, mod_depth=mod_depth, ki=ki, ampl=ampl)")
ns2 = dict(vars(D)); exec(compile(fsrc, "fast_dbg", "exec"), ns2)
for backend in ("numpy", "scipy"):
    R1,Rs1,_,d2 = ns2["main_calc_fast"](col.copy(), fa, FS, gzi, hW, backend=backend)
    print(f"\n=== batched version with {backend}.fft: R = {R1:.6f} (original {R0:.6f}) ===")
    for k in ["hBP","mod_depth","ki"]:
        a,b = dbg[k], d2[k]; i = np.unravel_index(np.argmax(np.abs(a-b)), a.shape)
        print(f"{k:10s} max|diff| {np.max(np.abs(a-b)):.3e} at {i}  orig {a[i]:.6g}  fast {b[i]:.6g}")
    print("ki diffs, channels 15-21:", np.round(np.abs(dbg['ki']-d2['ki'])[15:22],5))
print("\nChannels 20 and 21 carry only rounding noise (hBP ~1e-17, mod_depth ~1e-16) yet pass the 'all samples nonzero' test,")
print("so their correlation with channels 18 and 19 is computed from noise and depends on the exact FFT routine.")
