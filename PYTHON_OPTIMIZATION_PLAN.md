# How far can Python-only optimisation go?

A companion to `RUST_ACCELERATION_PLAN.md`. Same hot spots, same machine,
same inputs, but the question is: what do we get from refactoring the Python
itself (numpy, scipy, the standard library), and what does an optional JIT
(numba) add on top? Every prototype lives in `benchmarks/python_opt/`, runs
against the unmodified package, and checks its output against the original.

## 1. Answer

Pure Python gets most of the way for the two ECMA metrics and for
`roughness_dw`, and only part of the way for the Zwicker time-varying chain.
The gap is almost entirely two sequential kernels.

| Metric (signal) | Original | Pure Python refactor | + numba (optional) | Parity with original |
| --- | --- | --- | --- | --- |
| `loudness_zwtv`, `sharpness_din_tv` (1 s / 5 s) | 1× | **4.7× / 5.5×** | **30× / 25×** | N identical, specific loudness ≤ 7e-15 |
| `loudness_ecma` (1 s / 5 s / 20 s) | 1× | 1.9× / 1.6× / 3.8× single thread; **3.7× / 5.3× / 11.6×** with 4 threads; memory ÷45 | n/a | ≤ 3e-14 relative |
| `roughness_ecma` (1 s / 5 s) | 1× | 1.3× / 2.4× single thread; **2.3× / 5.7×** with 4 threads; memory ÷20 | peak picking not measured | ≤ 2e-13 |
| `roughness_dw`, tonal signal (1 s / 5 s) | 1× | **2.0× / 3.7×** with 4 threads | n/a | ≤ 2e-15 |
| `roughness_dw`, broadband noise (1 s / 5 s) | 1× | **28× / 33×** with 4 threads (11× single thread) | n/a | ≤ 6e-17 |
| `noct_spectrum`, `loudness_zwst`, `sharpness_din_st` | 1× | 1.1–1.7× from a filter-design cache; 3.2× at 10 s with threads | n/a | identical |

How this compares with Rust, using only what was measured rather than the
estimates in the Rust plan:

* The one Rust stage that was built, the nonlinear decay, is 110× faster than
  the Python stage. A numba version of the same loop is 145× faster
  (14 ms against 2 100–2 500 ms) and a scalar-Python version is 7–10× faster.
  Numba and Rust are equivalent here; plain Python is an order of magnitude
  behind.
* End to end, Rust for that stage alone gives 11–15× on `loudness_zwtv`. Pure
  Python gives about 5×. Adding numba for both sequential kernels gives
  25–30×.
* For `loudness_ecma`, `roughness_ecma` and `roughness_dw` there is no
  sequential kernel. Their cost is memory traffic, Python-level loops and
  unused cores, and those are fixable in Python. The Rust plan's estimates
  for them (7–15× and about 10×) were not measured, and the Python results
  above already reach a good share of them on 4 cores.

Timings on this container vary a lot between runs (a run of the same code
differed by up to 2.5×, and the first call of a memory-hungry function is the
slowest). Ratios below come from best-of-three runs measured in one
session. Treat them as ±30 %. All thread results are on 4 vCPUs.

## 2. The techniques, in order of payoff

1. **Stop materialising the overlapping block array.** `loudness_ecma` and
   `roughness_ecma` build a `(53, blocks, block_size)` array that repeats each
   sample two to four times, then clip, square and filter it in place. Doing
   the same work one band at a time keeps the identical maths and cuts peak
   memory 20–50×. On a 20 s signal the original allocates 2.9 GB.
2. **Use the other cores.** The 53 ECMA bands, the 47 channels of
   `roughness_dw` and the 48 n-octave filters are independent. SciPy's
   `lfilter`, `hilbert` and `decimate` and numpy's FFT all release the GIL, so
   a plain `ThreadPoolExecutor` scales: the 53-band gammatone bank went from
   0.72 s to 0.16 s on 4 cores with identical output, and no new dependency
   is needed.
3. **Replace per-element Python loops with batched array calls.**
   `_roughness_dw_main_calc` loops over 47 channels, and inside that over
   every audible spectral line. On broadband input that is thousands of
   Python-level iterations per 200 ms segment. Building the whole
   `(channel, line)` matrix and doing one batched FFT removed that cost
   (4.9 s per second of noise down to 0.17 s together with threads).
4. **Write sequential recurrences as scalar loops, not masked numpy.** The
   nonlinear-decay loop updates 21 values per step with 14 numpy mask
   operations. At width 21 the numpy call overhead is far larger than the
   arithmetic. A plain Python loop with `if` statements is 7–10× faster than
   the numpy version and bit-identical. Rewriting the same masks with
   `np.where` only gave 1.5–2×.
5. **Cache work that repeats.** `scipy.signal.butter` redesigns the same 38
   band-pass filters on every call to `noct_spectrum`, `loudness_zwst` and
   `sharpness_din_st`. An `lru_cache` is two lines and gives 1.1–1.7×. This is
   smaller than the 4× the Rust plan assumed and should not be oversold.
6. **A JIT for what is left.** Only the two sequential kernels need it, and
   only if 5× is not enough.

## 3. Results by component

### Nonlinear temporal decay (`_nl_loudness`)

`benchmarks/python_opt/nl_loudness_py.py`, `nl_loudness_numba.py`.

| Variant | Time, 2 000 frames | Speedup | Max difference |
| --- | ---: | ---: | ---: |
| original | 2 100–2 500 ms | 1× | — |
| `np.where` per step | 1 130–1 190 ms | 2× | 0 |
| scalar Python loop | 220–260 ms | 7–10× | 0 |
| numba, single compile of about 0.7 s | 14–15 ms | 145–170× | 0 |
| Rust spike (earlier measurement) | 12 ms | about 110× | 4e-15 |

### Specific loudness slopes (`_calc_slopes`)

`calc_slopes_fast.py`: a per-frame scalar port of the grid walk found in the
Rust branch. It reproduces the original on all 480 000 output cells and all
2 000 totals of a random-spectrum test. It is 3–5× faster as plain Python
and 29–58× under numba. `_calc_slopes` is already vectorised across frames
and spends a quarter of its time in `_get_rns_index` building an 18-row tile;
a `searchsorted` there would help the plain-Python path but was not built.

### Whole `loudness_zwtv`

`loudness_zwtv_bench.py` patches the two kernels in and runs the real entry
point.

| Variant | 1 s | 5 s |
| --- | ---: | ---: |
| original | 2.35 s | 13.7 s |
| scalar Python kernels | 0.50 s (4.7×) | 2.48 s (5.5×) |
| numba kernels | 0.08 s (30×) | 0.55 s (25×) |
| Rust decay stage only | 0.21 s (11×) | 0.91 s (15×) |

With numba in, the floor is the third-octave filter bank at about 0.2 s of
0.5 s for 5 s of audio, then the main loudness table lookups. Both are
SciPy-bound and independent across bands, so threads are the next lever.

### `loudness_ecma`

`loudness_ecma_fast.py`. Per band: filter, gather the blocks the original
gathers, rectify and take the RMS, then free everything. Two block-RMS
methods:

* `gather` (default): the same samples and the same summation as the original,
  differences ≤ 3e-14.
* `cumsum`: running-sum differences, up to 2× faster than `gather` on longer signals (no gain at 1 s), but
  precision depends on the dynamic range. A test with a loud second followed
  by 19 s at −100 dB stayed at 1.6e-11 because the quiet part sits below the
  hearing threshold, but it is not guaranteed in general.

| Duration | Original | gather, 1 thread | gather, 4 threads | cumsum, 1 thread | Peak memory |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 s | 0.34 s | 0.18 s | 0.09 s | 0.17 s | 150 → 3 MB |
| 5 s | 1.69 s | 1.03 s | 0.32 s | 0.61 s | 723 → 16 MB |
| 20 s | 18.9 s | 5.0 s | 1.63 s | 2.5 s | 2 866 → 62 MB |

The ECMA block index `linspace(l·sh, l·sh+sb, sb).astype(int32)` turns out to
select `sb − 1` consecutive samples plus the sample `sb` positions after the
block start. This was verified for every block of three block sizes, so the
cumulative-sum variant reproduces the original's quirk exactly.

### `roughness_ecma`

`roughness_ecma_fast.py`. The same per-band restructure, plus the envelope
step (`hilbert`, then two `decimate` calls) applied per band. Only the final
`(blocks, 53, 512)` envelope array is kept.

| Duration | Original | 1 thread | 4 threads | Peak memory |
| --- | ---: | ---: | ---: | --- |
| 1 s | 2.2 s | 1.66 s | 0.95 s | 592 → 29 MB |
| 5 s | 13.4 s | 5.56 s | 2.34 s | 2 312 → 111 MB |

Differences stay at 1.6e-13. At 5 s with 4 threads the remaining time is
2.4 s in the per-band front end (FFT-bound `hilbert` and IIR-bound
`decimate`) and 0.5 s in peak picking and weighting, which is Python-bound
and not helped by threads. Skipping empty spectra does not help, since 68 %
(an amplitude-modulated tone) to 98 % (noise) of the spectra are non-empty.
Numba on the peak-picking loop is the obvious next step but was not built.

### `roughness_dw`

`roughness_dw_fast.py`, `roughness_dw_bench.py`. Batches the 47 channels and
the audible lines into array operations, one batched FFT, then threads across
the 200 ms segments.

| Signal | Original | Batched, 1 thread | Batched, 4 threads |
| --- | ---: | ---: | ---: |
| tone, 1 s | 0.44 s | 0.43 s | 0.22 s (2.0×) |
| tone, 5 s | 2.24 s | 1.83 s | 0.61 s (3.7×) |
| white noise, 1 s | 4.86 s | 0.44 s (11×) | 0.17 s (28×) |
| white noise, 5 s | 27.7 s | 2.56 s (11×) | 0.83 s (33×) |

The Rust plan called `roughness_dw` "numpy-FFT-bound" from a profile of a
pure tone, which has only a handful of audible lines. On broadband input the
Python loops dominate and the plan was wrong about it.

### Filter-design cache and threads for `noct_spectrum`

`noct_spectrum_fast.py`, `design_cache.py`. The cache alone gave 1.1–1.7×
depending on the run. With threads across bands, 10 s of audio went from
445 ms to 141 ms (3.2×) with identical output.

## 4. Where pure Python stops

* **Sequential recurrences.** The nonlinear decay and the slopes walk cannot
  be batched across time. Scalar Python is the best available (7–10× and
  3–5×); numba or Rust gets 30–150×.
* **FFT- and IIR-bound stages.** `hilbert` and `decimate` already run in C. A
  Rust version would save memory traffic and fuse the passes, but the
  arithmetic would be about the same. The Python route cannot go below
  roughly 2 s for a 5 s `roughness_ecma` on 4 cores without changing those
  calls.
* **Peak picking.** 3 000 small `find_peaks` calls per 5 s plus Python
  weighting loops. Threads do not help because it holds the GIL.
* **Cores.** All thread gains depend on core count. With one core the
  threaded rows revert to the single-thread column.

## 5. Findings along the way

* **`roughness_dw` is sensitive to FFT rounding noise.** In a two-tone test,
  channels 20 and 21 contain only rounding noise at about 1e-17, yet they pass
  the `hBP[i].all() != 0` test, so the algorithm correlates noise with the
  neighbouring channels. Swapping `numpy.fft` for `scipy.fft` changed the
  roughness by 0.13 % (0.145031 → 0.145216) because the noise differs.
  `benchmarks/python_opt/roughness_dw_rounding_noise.py` shows it. The batched
  version therefore keeps numpy's FFT to match the original exactly. A
  tolerance in that test would make the metric stable but would change
  results, so it is worth deciding deliberately rather than as a side effect
  of optimisation.
* `loudness_ecma._preprocessing` writes the fade-in into the caller's array
  (`signal[:240] *= ...`). All prototypes pass a copy. The Rust branch's
  deviation register records the same behaviour.
* The `loudness_ecma` default block sizes (2048 and 1024) differ from the
  `roughness_ecma` ones (16384 and 4096). An early prototype of mine used the
  wrong ones, which is how this was noticed.

## 6. Recommendation

1. **Do the pure-Python phase first.** It needs no new dependency, keeps
   outputs identical to about 1e-13, and covers the ECMA metrics,
   `roughness_dw` and the memory problem. Order: per-band restructure of
   `loudness_ecma` and `roughness_ecma`; the batched `roughness_dw`; scalar
   loops for the two Zwicker kernels; the filter-design cache; an opt-in
   `workers` argument (or `MOSQITO_WORKERS`) for threads, default 1 so users
   who already run MoSQITo in their own thread pool are not oversubscribed.
2. **Then choose between numba and Rust for the two Zwicker kernels and peak
   picking.** They are small, about 250 lines in either language, and give
   the 25–30× on `loudness_zwtv`. Numba matched the Rust spike on speed and
   needs no build step, but adds a heavy dependency, a roughly 0.7 s compile
   on first call (cacheable) and limits supported numpy versions. Rust gives
   no JIT latency and releases the GIL for threaded use, but costs a build
   and wheel pipeline. If the package must stay pure Python, scalar loops
   still deliver 5× on `loudness_zwtv`.
3. **Leave the rest of the Rust plan unbuilt** unless the pure-Python result
   on `roughness_ecma` (about 5.7× on 4 cores) is not enough. Fusing
   `hilbert` and `decimate` in Rust is the only remaining large item and its
   gain over this baseline is unmeasured.

## 7. Reproducing

```bash
pip install numpy scipy matplotlib pyuff numba
cd benchmarks/python_opt
export PYTHONPATH=../..:.
python nl_loudness_py.py            # decay: np.where vs scalar loop
python nl_loudness_numba.py         # decay: numba
python calc_slopes_fast.py          # slopes: parity and speed
python loudness_zwtv_bench.py       # whole loudness_zwtv, all variants
python loudness_ecma_fast.py        # gather vs cumsum, threads, memory
python roughness_ecma_fast.py 1 5   # per-band roughness_ecma
python roughness_dw_bench.py        # batched roughness_dw
python roughness_dw_rounding_noise.py
python noct_spectrum_fast.py        # design cache and threads
python gammatone_threads.py         # GIL release check
python peak_picking_probe.py        # how many spectra are empty
```

`loudness_zwtv_bench.py` also compares against the Rust spike if
`benchmarks/spike_nl_loudness` has been built and installed.
