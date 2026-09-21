# Targeted Rust acceleration of MoSQITo

A plan for putting a *small* amount of Rust inside the existing `mosqito`
Python package, where profiling shows it pays, instead of porting the package.
Everything here is measured on the current `master` (Python 3.11, numpy 2.x,
scipy 1.17, 4 vCPU container), with the scripts in `benchmarks/`.

## 1. Summary

* Two shared helpers explain most of the pure-Python slowness:
  * `_nl_loudness` (`loudness_zwtv/_nonlinear_decay.py`), a per-sample
    recurrence written as 48 000 masked-numpy steps per second of audio. It is
    **91 %** of `loudness_zwtv` and `sharpness_din_tv`.
  * the ECMA-418-2 front end shared by `loudness_ecma` and `roughness_ecma`
    (`_band_pass_signals` + `_ecma_time_segmentation` + blockwise envelope /
    RMS), which materialises a `(53, n_blocks, 16384)` array with 4× overlap
    redundancy. It costs **2.3 GB** of peak allocation and ~65 % of the run
    time of a 5 s `roughness_ecma`.
* A 60-line Rust spike of `_nl_loudness` (`benchmarks/spike_nl_loudness/`) is
  bit-faithful to 4e-15 and makes the stage **110× faster**, which makes
  `loudness_zwtv` end to end **10–12× faster** (1.48 s → 0.15 s per second of
  audio; 7.8 s → 0.64 s for 5 s).
* Five Rust functions, roughly 900 lines in total, cover every hot spot worth
  touching. Everything else (`roughness_dw`, tonality, SII, `noct_spectrum`)
  is already numpy/scipy-bound or runs in tens of milliseconds and is *not* a
  Rust candidate.
* Filter *design* stays in SciPy (it is cheap and done once per call); Rust
  does the per-sample and per-block work, using `rustfft`/`realfft` for FFTs
  and `sci-rs` for SciPy-compatible `sosfilt`/`sosfiltfilt`.
* The Python implementation stays as the reference and the fallback: the
  extension is optional at install time and switchable at run time, and every
  accelerated helper is differential-tested against its Python body.

## 2. Baseline measurements

Wall time of each public entry point (`benchmarks/profile_hotspots.py`).
Signals: pink-ish noise for the Zwicker family, a 1 kHz tone amplitude
modulated at 70 Hz for the ECMA and roughness metrics.

| Metric | 1 s signal | 5 s signal | Where the time goes |
| --- | ---: | ---: | --- |
| `loudness_zwtv` | 1.55 s | 8.1 s | `_nl_loudness` 91 %, `_calc_slopes` 8 % |
| `sharpness_din_tv` | 1.43 s | — | same pipeline as `loudness_zwtv` |
| `roughness_ecma` | 2.3–4.1 s | 9.2–14.3 s | `hilbert` 43 %, `decimate` 16 %, `_ecma_time_segmentation` 10 %, `_peak_picking` + `_estimate_fund_mod_rate` 14 %, gammatone bank 6 % |
| `loudness_ecma` | 0.22–0.44 s | 1.1–2.5 s | `_ecma_time_segmentation` 40 %, gammatone bank 40 % |
| `roughness_dw` | 0.24 s | 1.3 s | numpy FFT 67 % (already C) |
| `loudness_zwst`, `sharpness_din_st`, `noct_spectrum` | 0.05 s | 0.17 s | SciPy `butter` *design* 80 % at 1 s (38 designs per call) |
| `loudness_zwst_perseg` | 0.08 s | — | as above |
| `tnr_ecma_st`, `pr_ecma_st` | 0.02 s | — | `_LTH`, `_spectrum_smoothing` |
| `sii_ansi` | 0.014 s | — | nothing to gain |

Peak memory (tracemalloc) for the ECMA metrics:

| Metric | 1 s | 5 s |
| --- | ---: | ---: |
| `roughness_ecma` | 592 MB | 2 312 MB |
| `loudness_ecma` | 150 MB | 723 MB |

The wide wall-time ranges for the ECMA metrics come from page-faulting those
allocations. Memory, not arithmetic, is what makes `roughness_ecma` slow.

Stage breakdown of `roughness_ecma` on 5 s (total 9.2 s on the fast run):

| Stage | Time | Notes |
| --- | ---: | --- |
| `scipy.signal.hilbert` on `(53, 60, 16384)` | 4.13 s | complex FFT of every overlapping block |
| `scipy.signal.decimate` ×2 (8 then 4) | 1.53 s | cheby1 + `sosfiltfilt` with odd-extension padding |
| `_ecma_time_segmentation` | 0.88 s | fancy-indexed gather into the big block array |
| `_peak_picking` (`find_peaks` + refinement) | 0.76 s | 53 × 60 Python calls |
| `_estimate_fund_mod_rate` | 0.58 s | Python loops over ≤10 peaks |
| `_band_pass_signals` (53 complex `lfilter`) | 0.44 s | already C, but serial |
| `_loudness_from_bandpass` (clip + RMS) | 0.23 s | on the big block array |
| everything else | < 0.05 s | |

## 3. Lessons taken from the `claude/psychoacoustics-rust-port-*` and `claude/phase-2-rust-port` branches

Reused directly:

* `crates/mosqito-core/src/loudness/zwtv/nonlinear_decay.rs`: the exact
  mask-by-mask transcription of `_nl_loudness`, including the `col - 1 = -1`
  wraparound on the first step. The spike in this branch is the same logic.
* `crates/mosqito-core/src/loudness/zwst/calc_slopes.rs` (318 lines): the
  finding that `_calc_slopes.py`'s vectorised code processes *at most one
  slope switch per 0.1-Bark grid position*, so a scalar grid walk reproduces
  it and a closed-form jump does not. It was validated on 637 golden cases.
* `crates/mosqito-core/src/dsp/peaks.rs`: `find_peaks` with SciPy prominence
  semantics, needed for `_peak_picking`.
* The ECMA block index formula `linspace(l*sh+i0, l*sh+i0+sb, sb).astype(int32)`
  skips one sample per block. Reproduce it, do not "fix" it, so results stay
  identical to the Python.
* `rust/tools/gen_golden*.py`: the pattern of capturing Python outputs on
  fixed inputs to JSON and asserting the Rust against them.

Deliberately not carried over:

* The separate workspace, the standalone `mosqito-core` crate, the mirror
  `mosqito_rs` Python package and the crates.io/PyPI release machinery. This
  plan puts one small `cdylib` inside `mosqito` itself.
* Re-implementing SciPy filter design (`butter`, `cheby1`, `zpk2sos`),
  `resample`, `pchip`, percentiles and windows. Those stay in SciPy/numpy;
  they are not where the time goes.
* Whole-metric ports and the "standard wins over Python" policy with its
  `DEVIATIONS.md`. Here the Python output is the oracle, because the goal is
  to make the *existing* package fast, not to re-derive it. Standards questions
  found on the way get filed as separate issues.

## 4. Ranked candidates

Ordered by benefit per line of Rust. "Users" lists the public metrics that go
through the helper.

### P1 — `_nl_loudness` (nonlinear temporal decay, ISO 532-1)

* Users: `loudness_zwtv`, `sharpness_din_tv`.
* Rust: ~60 lines, no crates beyond `pyo3`/`numpy`/`ndarray`. Plain nested
  loop over 21 bands × (n_frames × 24) steps. Done: `benchmarks/spike_nl_loudness/src/lib.rs`.
* Measured: stage 1 291 ms → 11.8 ms (110×); `loudness_zwtv` 1.48 s → 0.15 s
  (10×) at 1 s, 7.81 s → 0.64 s (12×) at 5 s. Total loudness `N` identical,
  specific loudness within 4e-15 (the spike computes the upsampled input as
  `c + j·δ`; accumulating `+= δ` like the Python makes it bit-identical).
* Note: the wraparound seeding of the first step from the last frame is kept
  on purpose (parity).

### P2 — `_calc_slopes` (+ `_main_loudness`) (specific loudness pattern, ISO 532-1)

* Users: `loudness_zwst`, `loudness_zwst_freq`, `loudness_zwst_perseg`,
  `loudness_zwtv`, all five `sharpness_din_*`.
* Rust: ~300 lines, lifted from the branch's `calc_slopes.rs` and reduced to a
  per-frame scalar walk over the 240-point grid; `_main_loudness` is ~80
  lines of table lookups and can ride along or stay in numpy.
* Expected: after P1 this is ~60 % of what is left of `loudness_zwtv`
  (0.12 s/s → a few ms), so `loudness_zwtv` goes from ~10× to ~20× overall.
  For the stationary metrics it is a few milliseconds per call, which matters
  for `_perseg` and batch use.
* Rounding: `_calc_slopes.py` rounds to 8 decimals before comparisons
  (`dec_compare`); the Rust must do the same to stay identical.

### P3 — ECMA-418-2 front end: gammatone bank + rectified block RMS

* Users: `loudness_ecma`, `roughness_ecma`.
* Rust: ~150 lines.
  * `gammatone_bank(signal, b[53,6] complex, a[53,6] complex) -> (53, n)`:
    direct-form II transposed IIR with complex coefficients on a real input,
    real part ×2 returned; `rayon` over the 53 bands. Coefficients keep coming
    from `_gammatone` (Python). No crate offers complex-coefficient `lfilter`,
    and it is 20 lines.
  * `block_rms_rectified(bands, sb, sh, i_start, n_blocks) -> (53, n_blocks)`:
    for each block, gather with the exact ECMA index formula, clip at 0,
    `sqrt(2·mean(x²))`. Never materialises the block array.
* Expected: `loudness_ecma` 1.1–2.5 s → ~0.15 s at 5 s (7–15×), peak memory
  723 MB → tens of MB. Removes ~15 % of `roughness_ecma` time. The 5.1.2 ear
  filter (`sosfilt`, 1 ms) and `_nonlinearity` stay in Python.

### P4 — `roughness_ecma` envelope power spectrum per block

* Users: `roughness_ecma` only, but it is the single most expensive metric.
* Rust: ~250 lines. One function
  `envelope_spectra(bands, sb, sh, i_start, n_blocks, sos_q8, sos_q4, hann) -> (n_blocks, 53, 256)`
  doing, per band and block: gather (same formula as P3), analytic-signal
  envelope via `realfft`/`rustfft` (16384-point FFT, zero negative bins,
  inverse FFT, modulus), `sosfiltfilt` by 8 then 4 with the cheby1 SOS
  designed once in SciPy and passed in, keep every q-th sample, ECMA von Hann
  window, 512-point FFT, one-sided scaled power. `rayon` over bands.
  `sci-rs::signal::filter::sosfiltfilt_dyn` mirrors SciPy's odd-extension
  `padlen` and `sosfilt_zi` handling, so it should match `scipy.signal.decimate`
  once given the same SOS. Verify it on the golden vectors before relying on
  it; the branch's hand-rolled `dsp/filter.rs::sosfiltfilt` is the fallback.
  (`sci-rs` 0.4.1 has Butterworth design but its Chebyshev I design is a
  `todo!()`, which is why the SOS is designed in SciPy.)
* Expected: removes `hilbert` + `decimate` + segmentation + clip, ≈ 65 % of
  `roughness_ecma` time and ~95 % of its memory. With P3, roughly 5× on time
  and 50× on memory at 5 s. FFT rounding differs from pocketfft, so parity
  here is ~1e-9 relative rather than bit-exact.

### P5 — `roughness_ecma` peak picking and fundamental modulation rate

* Users: `roughness_ecma`.
* Rust: ~200 lines: `find_peaks` with prominence (from the branch's
  `peaks.rs`), Eq. 72 threshold, top-10 by prominence, the `_refinement`
  quadratic fit with the corrected `_rho` bias, and `_estimate_fund_mod_rate`
  (integer-ratio harmonic complex search). Takes `Phi_E (n_blocks, 53, 256)`
  and the per-band `fmax/rmax/q2` vectors, returns `amplitude (n_blocks, 53)`.
  The weighting functions in `_weighting.py` are three lines each and move
  with it.
* Expected: the remaining ~15 % of `roughness_ecma`; with P3+P4 the metric
  lands near 10× overall. Lowest priority because it is the most logic for
  the least time.

### Not Rust candidates (and what to do instead)

* `loudness_zwst` / `sharpness_din_st` / `noct_spectrum` at ~50 ms: 40 ms is
  SciPy *designing* 38 Butterworth band-pass filters per call. An
  `functools.lru_cache` on `_n_oct_time_filter`'s design keyed by
  `(fs, fc, alpha, N)` is a two-line Python change worth ~4×. Do this first.
* `roughness_dw`: 67 % is numpy's FFT (1 270 calls of 9 600 points per second
  of audio). Batching the 47 channels into one `(47, n)` `ifft`/`fft` call in
  numpy is the fix; Rust would only replace one C FFT with another.
* Tonality (`tnr_*`, `pr_*`), `sii_ansi`, `noct_synthesis`, generators: tens of
  milliseconds; not worth a compiled dependency.
* `time_segmentation` (utils): a Python `while` loop, but it appends views;
  negligible.

## 5. Integration design

### Layout

```
mosqito/
  _accel.py            # dispatch: Rust if importable, else the Python bodies
  _rs/                 # the Rust crate, one cdylib, no workspace
    Cargo.toml
    src/lib.rs         # pymodule registration only
    src/nl_loudness.rs
    src/calc_slopes.rs
    src/ecma_front_end.rs
    src/envelope_spectrum.rs
    src/peak_picking.rs
```

Each accelerated Python helper keeps its body and gains a two-line prologue:

```python
from mosqito._accel import rs
def _nl_loudness(core_loudness):
    if rs is not None:
        return rs.nl_loudness(np.ascontiguousarray(core_loudness, dtype=np.float64))
    ...  # existing Python
```

`mosqito/_accel.py` imports `mosqito._rs` inside a `try`, and honours
`MOSQITO_PURE_PYTHON=1` to force the fallback (used by the differential tests
and by anyone who wants to reproduce old numbers).

### Build

* `setuptools-rust` in the existing `setup.py`:
  `rust_extensions=[RustExtension("mosqito._rs", "mosqito/_rs/Cargo.toml", binding=Binding.PyO3, py_limited_api="auto", optional=True)]`.
  `optional=True` means `pip install .` still succeeds without a Rust
  toolchain and yields the pure-Python package. This keeps `setup.py`,
  `requirements.txt` and `MANIFEST.in` as they are (add `recursive-include
  mosqito/_rs *.rs Cargo.toml Cargo.lock`).
* Wheels: `cibuildwheel` with `setuptools-rust` (abi3 wheels, one per
  platform, CPython ≥ 3.9). `python_requires` moves from `>= 3.5` to
  `>= 3.9`; numpy ≥ 1.22 already implies ≥ 3.8.
* Crates (all current on crates.io at the time of writing):
  `pyo3 0.29` + `numpy 0.29` (bindings, `abi3-py39`), `ndarray 0.17`,
  `rayon 1.12`, `realfft 3.5` / `rustfft 6.4`, `num-complex 0.4`,
  `sci-rs 0.4.1` (`sosfilt_dyn`, `sosfiltfilt_dyn`). No hand-rolled FFT,
  no hand-rolled filter design.
* Threads: `rayon` only over the 53 ECMA bands (P3/P4). Release the GIL
  (`py.allow_threads`) around every Rust call so batch users can thread.

### Validation

* Oracle: the Python helpers, run with `MOSQITO_PURE_PYTHON=1`.
* Per helper, a differential pytest that generates inputs from the existing
  test signals (`tests/input/*.wav`, `am_sine_generator`,
  `sine_wave_generator`) and asserts `rust == python` with:
  `atol=0, rtol=1e-13` for P1/P2 (pure arithmetic transcriptions; aim for
  bit-identical), `rtol=1e-9` for P3–P5 (different FFT / summation order).
* The existing standards tests (ISO 532-1 Annex B compliance,
  DIN 45692, Daniel & Weber curve, ECMA reference points) run twice in CI,
  once per mode, via a matrix on the environment variable.
* `benchmarks/profile_hotspots.py` before/after each phase, numbers recorded
  in the PR.

## 6. Phased delivery

| Phase | Deliverable | Rust lines (approx.) | Headline gain |
| --- | --- | ---: | --- |
| 0 | Build plumbing (`setuptools-rust`, `_accel.py`, CI matrix, differential-test harness), `lru_cache` for the Butterworth designs | ~20 | `loudness_zwst` family ~4× for free |
| 1 | `nl_loudness` (spike promoted, `+= δ` accumulation for bit-parity) | 60 | `loudness_zwtv`, `sharpness_din_tv` 10–12× |
| 2 | `calc_slopes` (+ `main_loudness`) | 300 | `loudness_zwtv` ~20× overall; every Zwicker/DIN metric a few ms faster |
| 3 | `gammatone_bank`, `block_rms_rectified` | 150 | `loudness_ecma` 7–15×, memory ÷30 |
| 4 | `envelope_spectra` | 250 | `roughness_ecma` ~5×, memory ÷50 |
| 5 | `peak_picking` + `fund_mod_rate` | 200 | `roughness_ecma` ~10× |

Phases 1 and 2 are independent of 3–5 and can ship first; each phase is its
own PR with its differential test and benchmark numbers. Stopping after
phase 2 already removes the two worst user-facing waits (time-varying
loudness and sharpness); stopping after phase 4 fixes the memory blow-up that
makes long-signal `roughness_ecma` impractical.

## 7. Reproducing the numbers

```bash
pip install numpy scipy matplotlib pyuff
PYTHONPATH=. python benchmarks/profile_hotspots.py 1 5          # wall time per stage
PYTHONPATH=. python benchmarks/profile_hotspots.py 1 5 --memory # peak allocation (slower)

# P1 spike: needs cargo and maturin
cd benchmarks/spike_nl_loudness && maturin build --release -o dist && pip install dist/*.whl && cd ../..
PYTHONPATH=. python benchmarks/spike_nl_loudness/check.py
```
