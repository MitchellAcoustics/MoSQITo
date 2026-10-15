import time, numpy as np
import roughness_ecma_fast as RF
from roughness_ecma_fast import front_end, fft, _von_hann_window, _noise_reduction, _peak_picking
FS=48000
def phi_for(sig):
    env_ds, Ns, ta, _ = front_end(np.array(sig, float), 4)
    sbb=512; L=Ns.shape[0]; hann=_von_hann_window(sbb)
    phi_E0=np.sum(np.power(env_ds*hann,2),axis=2); den=Ns.max(axis=1)[:,None]*phi_E0
    dft=(abs(fft(env_ds*hann,axis=2)[:,:,:sbb//2])/2*np.sqrt(2))**2
    sc=np.zeros((L,53)); sc[den!=0]=np.power(Ns[den!=0],2)/den[den!=0]
    return _noise_reduction(sc[:,:,None]*dft)
def stage(Phi, skip):
    t0=time.perf_counter(); n_calls=0; n_peaks=0
    for l in range(Phi.shape[0]):
        for z in range(53):
            row=Phi[l,z]
            if skip and not row.any(): continue
            n_calls+=1
            f_p,Ai=_peak_picking(row); n_peaks+=len(f_p)
    return time.perf_counter()-t0, n_calls, n_peaks
t=np.arange(FS*5)/FS
for name,sig in [("AM tone",0.05*(1+np.sin(2*np.pi*70*t))*np.sin(2*np.pi*1000*t)),("white noise",0.02*np.random.default_rng(0).standard_normal(len(t))),("AM noise",0.02*(1+0.8*np.sin(2*np.pi*70*t))*np.random.default_rng(1).standard_normal(len(t)))]:
    Phi=phi_for(sig); tot=Phi.shape[0]*53
    a=stage(Phi,False); b=stage(Phi,True)
    print(f"{name:11s} (l,z) pairs {tot}, nonzero spectra {b[1]} ({100*b[1]/tot:.0f}%), pairs with peaks≥1 -> peak_picking all {a[0]:.2f}s, skipping empty {b[0]:.2f}s")
