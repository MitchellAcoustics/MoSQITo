"""Parity and timing check for the nl_loudness spike against MoSQITo's Python.

Build and install the spike first (needs a Rust toolchain and maturin):

    cd benchmarks/spike_nl_loudness && maturin build --release -o dist && pip install dist/*.whl

then run this from the repository root:

    PYTHONPATH=. python benchmarks/spike_nl_loudness/check.py
"""
import sys
import time

import numpy as np

import mosqito  # noqa: F401  (registers the submodules below)
import mosqito_spike
from mosqito.sq_metrics.loudness.loudness_zwtv._nonlinear_decay import _nl_loudness

ZT = sys.modules["mosqito.sq_metrics.loudness.loudness_zwtv.loudness_zwtv"]
rng = np.random.default_rng(1)

for nt in [2, 7, 400, 2000]:
    core = np.abs(rng.standard_normal((21, nt))) * rng.uniform(0, 5, (21, 1))
    core[:, ::3] *= 0.01
    a = _nl_loudness(core.copy())
    b = mosqito_spike.nl_loudness(np.ascontiguousarray(core))
    print(f"nt={nt:5d}  max|python - rust| = {np.max(np.abs(a - b)):.3e}")

FS = 48000
for dur in [1.0, 5.0]:
    sig = 0.02 * rng.standard_normal(int(FS * dur))
    t0 = time.perf_counter()
    N1, Ns1, _, _ = ZT.loudness_zwtv(sig, FS)
    t_py = time.perf_counter() - t0
    orig = ZT._nl_loudness
    ZT._nl_loudness = lambda c: mosqito_spike.nl_loudness(np.ascontiguousarray(c))
    t0 = time.perf_counter()
    N2, Ns2, _, _ = ZT.loudness_zwtv(sig, FS)
    t_rs = time.perf_counter() - t0
    ZT._nl_loudness = orig
    print(
        f"{dur:.0f} s signal: loudness_zwtv {t_py:.2f} s (python) -> {t_rs:.2f} s (rust stage), "
        f"{t_py / t_rs:.0f}x; N identical={np.array_equal(N1, N2)}, "
        f"max|N_spec diff|={np.max(np.abs(Ns1 - Ns2)):.1e}"
    )
