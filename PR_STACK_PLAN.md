# Plan: three stacked PRs to speed up MoSQITo (bug fix, pure Python, optional numba)

## Context

Profiling (see `PYTHON_OPTIMIZATION_PLAN.md` and the prototypes in
`benchmarks/python_opt/`, both on branch `claude/python-rust-optimization-plan-s2dzn4`)
showed that most of MoSQITo's slowness sits in a few shared helpers, and that
plain Python restructuring recovers much of it, with an optional numba JIT
covering the two sequential kernels. This plan turns that study into three
sequential PRs, each executable by an AI agent without further design decisions.

Order is deliberate: the bug fix lands first so that the "golden" reference
outputs used for all later parity checks are not contaminated by a numerically
fragile edge case.

| PR | Branch | Results change? |
| --- | --- | --- |
| 1. Fix `roughness_dw` noise channels, add parity harness | `fix/roughness-dw-noise-channels` | Yes, in edge cases (sparse or tonal spectra) |
| 2. Pure-Python performance refactor | `perf/python-refactor` | No, within stated tolerances |
| 3. Optional numba backend (`mosqito[performance]`) | `perf/numba-backend` | No, bit-identical to PR 2 |

The PRs are **stacked**: PR 1 branches from `master`, PR 2 from PR 1's branch,
PR 3 from PR 2's branch. Open each as a draft against its parent's branch (so the
diff shows only its own changes), titled `[1/3]`, `[2/3]`, `[3/3]`, with
`Stacked on #<parent>` on the first line of the description. Fixes to a lower PR
are committed there and merged up the stack (no rebasing). Merge bottom-up and
retarget the next PR to `master`; after a squash merge, run
`git rebase --onto origin/master <old parent tip> <child>` once.

The reference prototypes are **not** on `master`: run
`git fetch origin claude/python-rust-optimization-plan-s2dzn4` and read them with
`git show origin/claude/python-rust-optimization-plan-s2dzn4:<path>`. Treat them
as sketches to adapt, not code to copy blindly.

## Decisions already made

* Threading is exposed as a trailing keyword `workers=None` on `noct_spectrum`,
  `loudness_ecma`, `roughness_ecma` and `roughness_dw`, resolved against the
  `MOSQITO_WORKERS` environment variable, default 1. No threads unless asked.
* Parity is checked against **frozen golden files** produced by PR 1, not
  against retained legacy code.
* Block RMS in the ECMA metrics uses the exact per-band gather (same samples,
  same summation as the original), not the faster cumulative-sum variant.
* Out of scope for all three PRs: Rust, `roughness_ecma` peak picking and
  fundamental-rate search (unmeasured under numba, needs a SciPy `find_peaks`
  re-implementation; follow-up only if PR 3 profiling still shows it), and
  rfft/irfft shortcuts in `roughness_dw`.

## Conventions for every PR

* Test-driven: write the failing test first, commit it, then make it pass.
  One commit per component so a regression bisects cleanly.
* Run the existing suite from the repo root (tests use relative paths such as
  `tests/input/...`): `pytest tests -x -q`. Record the baseline result on the
  parent branch before changing anything and state it in the PR.
* No public signature changes other than the `workers` keyword in PR 2.
  Never delete a private helper that `validations/` or `docs/` imports
  (`grep -r` first; `_ecma_time_segmentation` and `_band_pass_signals` are
  used by `validations/sq_metrics/loudness_ecma/`). Leave unused helpers in
  place and say so in the PR.
* Syntax must stay valid on Python 3.8 (`setup.py` still claims `>= 3.5`,
  which is already stale; do not change it in PRs 1 and 2).
* Every PR description contains: summary; an explicit "numerical impact"
  statement; the baseline and final test results; for PRs 2 and 3 a
  before/after table of wall time and peak memory (best of three runs, state
  the CPU count); and the checklist from its "Acceptance" section.

---

## PR 1: fix `roughness_dw` noise channels, add the parity harness

### Problem

`mosqito/sq_metrics/roughness/roughness_dw/_roughness_dw_main_calc.py` skips
the cross-correlation of two channels only when a channel's envelope is
**exactly** zero (`hBP[i].all() != 0`). Channels that contain only floating
point rounding noise (about 1e-17, modulation depth about 1e-16) pass that
test, so their correlation with neighbouring channels is computed from noise
and feeds `ki`, hence the roughness. Demonstrated by
`benchmarks/python_opt/roughness_dw_rounding_noise.py`: a 500 Hz + 600 Hz
two-tone signal gives R = 0.145031 with `numpy.fft` and 0.145216 with
`scipy.fft` (0.13 % apart) because channels 20 and 21 hold only noise.

### Fix

Define a channel as **active** when `h0 > 0` and its modulation depth
`hBPrms / h0` exceeds a documented floor, `_ACTIVE_MOD_DEPTH_FLOOR = 1e-10`
(real depths are above 1e-6, noise is about 1e-16). Compute `ki[i]` only when
channels `i` and `i + 2` are both active, otherwise `ki[i] = 0`, which is what
the original does for all-zero rows. This also stops `corrcoef` ever seeing a
constant row. Keep the change to that condition, so PR 2 can batch it.

### Steps (commit order)

1. **Harness** under `tests/parity/` (register a `parity` marker in `pytest.ini`):
   * `signals.py`: deterministic, seeded signal factory. Cases: AM tone
     (1 kHz carrier, 70 Hz, 60 dB), white noise, two tones (500 + 600 Hz),
     a loud burst followed by 19 dB-quieter silence (dynamic range),
     a length that is not a multiple of any hop size, a 44.1 kHz signal
     (resample path), and the ISO 532-1 pink noise wav already in
     `tests/input/`. Keep each ≤ 1.5 s.
   * `generate_goldens.py`: runs the metrics and writes compressed
     `tests/parity/data/<metric>.npz` plus `manifest.json` (numpy, scipy, python
     versions, signal specs). Metrics: `loudness_zwst`, `loudness_zwst_perseg`,
     `loudness_zwtv`, `sharpness_din_st`, `sharpness_din_tv`, `loudness_ecma`,
     `roughness_ecma`, `roughness_dw`, `noct_spectrum`. Also kernel-level
     goldens for `_nl_loudness` and `_calc_slopes` on seeded random and
     realistic core-loudness inputs (including a 1-D `nm`).
   * `tolerances.py`: one table, used by every parity test. Set to survive
     ordinary NumPy/SciPy upgrades, which change last digits of FFT and filter
     results: `rtol=1e-7` with `atol = 1e-9 × max|golden|` (the
     `numpy.testing.assert_allclose` convention). ISO 532-1 totals are rounded
     to 0.001 sone (0.01 above 16 sone), so allow one rounding quantum on those
     totals; integer outputs compare exactly. A failure message prints the
     versions recorded in `manifest.json` next to the installed ones.
   * `strict_compare.py --baseline <sha>`: builds the baseline commit in a
     temporary `git worktree`, runs the same signals in the same environment
     (generator run with `PYTHONPATH=<worktree>`; assert `mosqito.__file__` is
     inside it) and compares with the branch under test at tight tolerances
     (exact for `_nl_kernel`, `_calc_slopes` totals and `noct_spectrum`;
     `rtol=1e-10` elsewhere). Run by the PR author, not part of the suite; it
     covers what the loose golden tolerances cannot.
   * `test_parity_*.py`: one per metric, loading goldens and asserting against
     `tolerances.py`. Total golden size budget: under 3 MB.
2. **Red test** `tests/sq_metrics/roughness/test_roughness_dw_noise_channels.py`:
   run `_roughness_dw_main_calc` on the two-tone spectrum twice, the second time
   with a wrapper around `numpy.fft.fft`/`ifft` that adds seeded noise of
   relative size 1e-16; assert the roughness agrees to `rtol=1e-9`. Mark it
   `xfail(strict=True)` in this commit so it documents the failure. Add a second
   test that a spectrum with only audible components in far-apart bands gives
   `ki == 0` for the empty channels.
3. **Fix** the condition as above, remove the `xfail`, document the floor in
   the function docstring and add a "Notes" paragraph to `roughness_dw`'s
   docstring.
4. **Input mutation** (separate commit, no numerical change):
   `loudness_ecma/_preprocessing.py` writes the fade-in into the caller's array
   (`signal[:240] *= w_fadein`). Copy first. Test: calling `loudness_ecma` and
   `roughness_ecma` leaves the input array unchanged.
5. **Generate goldens** from the fixed code (`python tests/parity/generate_goldens.py`),
   commit them, and enable the parity tests.
6. **Impact report** (in the PR description, not tested): run
   `strict_compare.py --baseline <master sha>` and the Daniel and Weber grid in
   `validations/sq_metrics/roughness_dw/` on old and new code; tabulate ΔR.
   Expect no change for broadband and amplitude-modulated signals and a change
   only for sparse or tonal spectra.
7. **Optional CI** (final commit; drop it if the maintainers run CI elsewhere,
   there is no `.github/` on `master`): `.github/workflows/tests.yml` running
   `pytest tests -q` on Python 3.10 to 3.12.

### Acceptance

* Existing suite passes, including `test_roughness_dw*` Daniel and Weber
  compliance checks, with the baseline recorded.
* The new noise-channel test passes and is stable across 50 random seeds.
* `strict_compare` against `master` shows differences only in `roughness_dw`
  for sparse-spectrum cases, each listed with before/after values and the
  reason; every other metric matches at the strict tolerances.
* `loudness_ecma` and `roughness_ecma` do not modify their input.
* Goldens, generator, manifest, tolerance table and `strict_compare.py`
  committed; regenerating the goldens from this branch reproduces the committed
  files within the tolerances.
* PR description states plainly that results change for sparse spectra.

---

## PR 2: pure-Python performance refactor

Requires PR 1. **Every golden must pass unchanged**; never regenerate them in
this PR.

### Design rule

Write the two sequential kernels as module-level functions that take and
return NumPy arrays and use only scalars, `for` loops, `if`, `min`/`max`,
`np.rint`, `np.floor`, `np.abs`: no dicts, no Python lists, no keyword-only
calls. They must work as plain Python and must be compilable by numba without
edits, so PR 3 only adds a decorator.

### Components (one commit each, tests first)

1. **`mosqito/utils/_parallel.py`**: `resolve_workers(workers)` (argument, then
   `MOSQITO_WORKERS`, then 1; integers ≥ 1; otherwise `ValueError`) and
   `parallel_map(fn, items, workers)` (plain list comprehension when
   `workers == 1`, otherwise `ThreadPoolExecutor.map` with a pool of at most
   `len(items)` threads, order preserved). Unit tests for resolution order,
   invalid values, ordering and the pool cap. The useful maximum is set by the
   algorithm, not by a constant: 53 independent bands in `loudness_ecma` and
   `roughness_ecma`, one task per band in `noct_spectrum` (about 28 for third
   octaves from 24 Hz to 12.5 kHz), one per 200 ms segment in `roughness_dw`
   (about 9 per second at the default overlap). Document these in
   `performance.rst`, together with the fact that gains stop at the machine's
   core count.
2. **`_nl_loudness`** (`loudness_zwtv/_nonlinear_decay.py`): move the recurrence into
   `_nl_kernel(ui, B, out)` as scalar loops. Keep the `ui_delta` construction
   identical (iterative accumulation) so output is bit-identical, and keep the
   first-step wraparound from the last column. Reference:
   `benchmarks/python_opt/nl_loudness_py.py`.
3. **`_calc_slopes`** (`loudness_zwst/_calc_slopes.py`): replace the masked
   array routine with a per-frame scalar walk `_calc_slopes_frame(nm, out)` plus a loop over
   frames; keep the 1-D `nm` behaviour and the rounding of the totals. Reference:
   `benchmarks/python_opt/calc_slopes_fast.py` (matches the original on 480 000
   cells). The original misbehaves on degenerate patterns (for example
   `nm = arange(21)` yields negative loudness); do not reproduce that, add a
   test documenting that the new output is non-negative there.
4. **ECMA front end**: new `mosqito/sq_metrics/loudness/loudness_ecma/_band_blocks.py`
   with `ecma_block_index(n_new, sb, sh)` (the original `linspace(...).astype(int32)`
   gather, computed once) and a per-band function that filters, gathers the
   blocks, rectifies and returns RMS. Use it in `loudness_ecma` (per band, via
   `parallel_map`) and `roughness_ecma` (also per band: `hilbert`, then the two
   `decimate` calls, keeping only the `(blocks, 53, 512)` envelope array).
   References: `loudness_ecma_fast.py`, `roughness_ecma_fast.py`. The scalar
   `sb`/`sh` assumption holds for `roughness_ecma`; `loudness_ecma` accepts the
   same scalars. Add the `workers` keyword.
5. **`roughness_dw`**: batch `_roughness_dw_main_calc` over the 47 channels and the
   audible lines (slope matrix, amplitude matrix, one batched
   `numpy.fft.ifft`/`fft`), keeping PR 1's active-channel rule. Process segments
   with `parallel_map`. Use `numpy.fft`, not `scipy.fft`. Reference:
   `roughness_dw_fast.py`. Add the `workers` keyword.
6. **`noct_spectrum`**: cache the band-pass design in
   `_n_oct_time_filter` via a small `_design_sos(N, w1, w2)` with
   `functools.lru_cache(maxsize=256)` returning a read-only array; run bands
   through `parallel_map`. Add the `workers` keyword. Apply the same cache in
   `_n_oct_freq_filter` if it calls `butter` with the same arguments.
7. **Benchmark script** `benchmarks/bench_metrics.py`: times the nine metrics on
   fixed seeded signals and prints wall time and peak memory.
8. **Docs**: docstring `Parameters` entry for `workers` on all four functions;
   new `docs/source/user_guide/performance.rst` (linked from
   `docs/source/user_guide/index.rst`) covering `workers`, `MOSQITO_WORKERS`,
   oversubscription advice, the memory reduction, and a measured speed table.

### Tests added before the code (red first)

* Parity suite from PR 1 (already present): must stay green after each commit.
* `workers=1` equals `workers=4` **exactly** for the four threaded functions.
* Memory budget, gating, using `tracemalloc` peak on fixed inputs:
  `loudness_ecma` 5 s under 100 MB (original 723 MB), `roughness_ecma` 1 s
  under 100 MB (original 592 MB). These fail before component 4 and pass after.
* Edge cases: signal shorter than one block, length not a multiple of the hop,
  44.1 kHz input, 1-D versus 2-D core loudness, `workers` invalid values.

### Acceptance

* Full existing suite and every golden pass at the `tolerances.py` values,
  unchanged. Any needed loosening is a stop-and-ask, not an edit.
* `strict_compare.py --baseline <PR 1 tip>` passes at the strict tolerances;
  paste its output in the PR.
* Measured on the same 4-core machine, best of three, against PR 1's tip
  (same performance as `master`; floors set below the study's measurements to
  allow for noise): `loudness_zwtv` at least 4×; `loudness_ecma` (20 s,
  `workers=1`) at least 3× and (`workers=4`) at least 8×; `roughness_ecma` (5 s)
  at least 2× single thread and 4× with 4 workers; `roughness_dw` on white noise
  at least 8× single thread; `noct_spectrum` no slower.
* `import mosqito` time and public signatures unchanged apart from `workers`.
* No new runtime dependency. Docs page added; Sphinx build shows no new
  warnings on touched pages.

---

## PR 3: optional numba backend, `mosqito[performance]`

Requires PR 2. Scope: add a decorator and packaging; do not rewrite
algorithms.

### Packaging (standard optional-extra pattern)

* `setup.py`: `extras_require={"performance": ["numba>=0.60"]}`; `numba` stays
  out of `requirements.txt` and `install_requires`. Restrict by Python version
  with a marker (`numba>=0.60; python_version >= "3.9"`). Confirm the lowest
  supported numba/numpy pairing in the test matrix and adjust the floor if a
  tested combination fails.
* `docs/source/user_guide/installation.rst`: add `pip install "mosqito[performance]"`
  with one line on what it does, and extend `performance.rst` with the numba
  section (first-call compile cost, `NUMBA_CACHE_DIR`, expected speedups).
  Mention the extra in `README.md`.

### Implementation

1. **`mosqito/utils/_jit.py`**:
   * `HAS_NUMBA`: true only if `numba` is importable and `MOSQITO_DISABLE_NUMBA`
     is not set to `1`.
   * `jit(**options)`: a decorator returning a **lazy** wrapper. `import numba`
     and compilation happen on first call, not at `import mosqito`, so import
     time does not regress for users who have numba installed. Compile with
     `numba.njit(cache=True, nogil=True, **options)`. Without numba the wrapper
     calls the original function. The wrapper always exposes `.py_func` (the
     plain function) and `.is_compiled`.
2. Apply `@jit()` to `_nl_kernel` and `_calc_slopes_frame`/the frame loop from
   PR 2. Coerce arguments at the call sites to C-contiguous `float64`.
   No `parallel=True` (avoids threading-layer conflicts; `nogil=True` already
   lets the `workers` thread pools overlap). If a kernel needs an edit to compile, fix
   it in the pure-Python source so both modes stay identical.

### Tests (red first)

* Kernel equality: for seeded random and realistic inputs, `kernel(...)` equals
  `kernel.py_func(...)` **exactly** (`atol=0`) for `_nl_kernel` and for the
  slopes totals and specific loudness. Skipped with a clear reason if numba is
  absent.
* Fallback: a subprocess test that sets `sys.modules["numba"] = None`, imports
  `mosqito`, runs `loudness_zwtv` on a short signal, and checks `HAS_NUMBA is False`.
  A second subprocess test with `MOSQITO_DISABLE_NUMBA=1`.
* Laziness: subprocess check that `"numba" not in sys.modules` after
  `import mosqito`.
* Whole-metric goldens for `loudness_zwtv`, `sharpness_din_tv`, `loudness_zwst`
  and `sharpness_din_st` pass **unchanged** in both modes.
* CI (extends the workflow from PR 1, or document equivalent local commands if
  none was added): matrix with and without `pip install .[performance]`.

### Acceptance

* All PR 2 acceptance criteria still hold with numba absent (no regression
  beyond 5 % against PR 2's tip).
* With numba: compiled and interpreted kernels bit-identical;
  `strict_compare.py --baseline <PR 2 tip>` passes in both modes; every golden
  passes unchanged; `loudness_zwtv` at least 15× faster than PR 1's tip at 1 s
  and 5 s, each kernel at least 50× faster than its PR 2 pure-Python version
  after warm-up (study: 25–30× and 145×/29×).
* `import mosqito` time with numba installed within 50 ms of PR 2
  (`python -X importtime -c "import mosqito"` before and after).
* First-call compile time measured and written in the docs; cached second
  process start measured too.
* `pip install mosqito` (no extra) works and never imports numba.

---

## Verification (for the person reviewing each PR)

```bash
git fetch origin && git checkout <pr-branch>
pip install -e . && pip install pytest
pytest tests -q                       # existing suite + parity + new tests
pytest tests/parity -q                # goldens only
python tests/parity/strict_compare.py --baseline <parent tip sha>   # PRs 2 and 3
pip install -e ".[performance]"       # PR 3: repeat both commands with numba
MOSQITO_DISABLE_NUMBA=1 pytest tests/parity -q
python benchmarks/bench_metrics.py    # PRs 2 and 3: before/after wall time and peak memory
```

## Risks and how the plan handles them

* **Timing noise** (runs differed up to 2.5× on the study machine): speed
  floors are set well below measured gains, timing is reported not gated, and
  memory (deterministic) is the gating performance test.
* **Golden drift with numpy/scipy versions**: tolerances are set for ordinary
  upgrades and the manifest records versions. If an upgrade still fails a
  golden, investigate before loosening, and regenerate only in a dedicated PR
  that records the old and new versions.
* **Thread oversubscription**: default is one worker; the docs say so.
* **Bug-fix scope creep in PR 1**: the fix is limited to the active-channel
  rule; any other discovered discrepancy is filed as a separate issue.
