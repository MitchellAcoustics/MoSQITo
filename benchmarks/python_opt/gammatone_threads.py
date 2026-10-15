import time, numpy as np, scipy.signal as sps
from concurrent.futures import ThreadPoolExecutor
from mosqito.sq_metrics.loudness.loudness_ecma._ear_filter_design import _ear_filter_design
from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import _auditory_filters_centre_freq
from mosqito.sq_metrics.loudness.loudness_ecma._gammatone import _gammatone
FS=48000; dur=5.0
t=np.arange(int(FS*dur))/FS; sig=0.05*(1+np.sin(2*np.pi*70*t))*np.sin(2*np.pi*1000*t)
x=sps.sosfilt(_ear_filter_design(), sig); cf=_auditory_filters_centre_freq()
coef=[_gammatone(f,k=5,fs=48000) for f in cf]
def band(z):
    b,a=coef[z]; bp=2.0*sps.lfilter(b,a,x).real
    return float(np.square(np.clip(bp,0,None)).sum())
def run(workers):
    t0=time.perf_counter()
    if workers==1: r=[band(z) for z in range(53)]
    else:
        with ThreadPoolExecutor(workers) as ex: r=list(ex.map(band, range(53)))
    return time.perf_counter()-t0, r
run(1)
r1=None
for w in [1,2,4,8]:
    ts=[]; 
    for _ in range(3):
        t_, r = run(w); ts.append(t_)
    if r1 is None: r1=r
    print(f"workers={w}: {min(ts):.3f} s  identical={r==r1}")
import os; print("cpus", os.cpu_count())
