# Deviations from MoSQITo's Python implementation

This is the register the plan requires: every place `mosqito-rs` deliberately
diverges from MoSQITo's current Python behaviour. Each entry records the
standard clause (where one applies), the MoSQITo file:line it diverges from,
what `mosqito-rs` does instead, why, and — where it matters — the measured
numeric impact.

The target is conformance to the published standards, not bit-parity with
MoSQITo's Python. Where the Python is unambiguously a bug relative to the
standard, `mosqito-rs` fixes it. Where the Python encodes a deliberate,
standards-sanctioned correction (the ECMA-418-2 roughness deviations from
Wanty, Glesser & Casagrande Hirono, INTERNOISE 2024), `mosqito-rs` follows the
same correction. Both Phase 1 and Phase 2 are complete: every metric they
scoped — including `roughness_dw`, speech intelligibility, the
`utils`/generator helpers, and tonality (TNR/PR) — has landed. Every entry
below is either **implemented** or **deferred** (identified during research
but deliberately out of either phase's scope).

---

## Implemented

### D-noct-1 — `center_freq` no longer truncates bands outside the nominal table

- **MoSQITo**: `mosqito/sound_level_meter/noct_spectrum/_center_freq.py:71-73`.
  For `n` of 1 or 3, band numbers are mapped onto the ANSI nominal ("preferred")
  frequency table by filtering: `ind = where((k >= -i_ref) & (k < len(freq) -
  i_ref)); f_nom = freq[k[ind] + i_ref]`. Because `f_exact` is *not* filtered
  the same way, a band number outside the table's range leaves `f_exact` and
  `f_nom` different lengths — a latent bug (mismatched array lengths propagate
  into `noct_spectrum`'s `spec`/`fpref` outputs), not something ANSI S1.1
  intends.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/slm/noct.rs`, `center_freq`.
  Falls back to the exact (unrounded) frequency for any band outside the
  table, so the two returned arrays are always the same length and always
  band-for-band aligned.
- **Why**: a caller receiving `spec` and `fpref` of different lengths has no
  correct way to use them together; this is an implementation accident, not a
  standards question.
- **Measured impact**: none, on every `fmin`/`fmax` pair MoSQITo's reference
  corpus and public API actually use (`(24, 12600)`, `(25, 20000)`, `(31.5,
  16000)`, etc. — all fully inside the table). The fallback path is exercised
  only by a dedicated unit test
  (`center_freq_falls_back_to_the_exact_frequency_below_the_table`), not by
  any conformance-gate input. Verified against the installed `mosqito`
  package by `rust/tools/gen_golden_noct.py`, which asserts the two
  implementations agree on every in-range case before writing the golden file.

### D-zwst-1 — `_calc_slopes.py:112` wraparound made explicit

- **MoSQITo**: `zup_ea` gets a trailing `0` appended and is then indexed with
  `i-1`, so the lookup for `i=0` relies on Python's negative-index
  wraparound to reach that trailing `0`.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/loudness/zwst/calc_slopes.rs`
  computes the same boundary directly (the pre-fill/grid-step algorithm
  never needs a `zup_ea[i-1]`-shaped lookup at all — see the module's doc
  comment for why the whole vectorised segment-jump approach was replaced by
  a position-by-position grid walk).
- **Why**: no behavioural change; purely a readability/robustness fix so the
  logic doesn't depend on wraparound.
- **Measured impact**: none — confirmed via 637 golden `calc_slopes` cases
  generated from real MoSQITo output (see `tests/golden_zwst.rs`), agreeing
  to ~1e-9 relative, and against the published ISO 532-1 Annex B2/B3
  reference values in `tests/conformance_iso532_1.rs`.

### D-zwst-2 — `conversion/amp2db.py:25` no longer mutates its input

- **MoSQITo**: `amp2db` replaces any exact-zero element of its input array
  *in place* (to avoid a `log10(0)` warning) before computing the result,
  silently mutating the caller's array as a side effect of what looks like a
  pure function.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/utils/conversion.rs`
  applies the same `0.0 -> 2e-12` substitution but only to a local copy,
  returning a new `Vec` and leaving the caller's slice untouched.
  `does_not_mutate_its_input` asserts this directly.
- **Why**: a pure-looking function silently mutating a caller's array is a
  footgun, and nothing in `loudness_zwst`'s pipeline (the only current
  caller) depends on the mutation being visible afterwards.
- **Measured impact**: none on any conformance or golden result — `amp2db`'s
  *output* is bit-identical either way; only the (unused, in every current
  call site) mutation of the input is dropped.

### D-din-1 — `sharpness_din_from_loudness.py:114/146` `UnboundLocalError` for 2-D `N`

- **MoSQITo**: `ind = where(N < 0.1)` is only computed when `N.ndim <= 1`
  (true for every value MoSQITo's own callers actually pass), but read
  unconditionally in the `else` branch at line 146 — a 2-D `N` (which no
  shipped entry point produces, but which a caller invoking
  `sharpness_din_from_loudness` directly with a pre-shaped array could) would
  raise `UnboundLocalError`, not a meaningful error message.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/sharpness/din.rs` has no
  such state — `sharpness_din_from_loudness` (scalar) and
  `sharpness_din_from_loudness_segmented` (per-segment, always masks `N <
  0.1`) are two separate functions with no shared mutable binding; the
  Python wrapper (`python/mosqito_rs/sharpness_din.py`) dispatches between
  them on `N`'s size, matching Python's actual `S.size == 1` dispatch rather
  than its `N.ndim` precondition for computing `ind`.
- **Why**: not a standards question, just a latent crash on an input shape
  none of MoSQITo's own code produces.
- **Measured impact**: none on any conformance or differential result — every
  real caller (`sharpness_din_st`, `_freq`, `_perseg`) passes `N` with
  `ndim <= 1`, the only shape MoSQITo's own code path actually exercises.

### D-din-2 — `sharpness_din_st.py:97-113` duplicated resample block dropped

- **MoSQITo**: resamples to 48 kHz itself (two copies of an identical
  `if fs < 48000: resample(...)` block, the second always a no-op since the
  first already updated `fs`) before calling `loudness_zwst`, which resamples
  to 48 kHz internally anyway if it still needs to.
- **`mosqito-rs`**: `sharpness_din_st` calls `loudness_zwst` directly; the
  single resample happens there.
- **Why**: the duplicate block is dead code (a copy-paste artifact), and the
  single live copy is itself redundant with `loudness_zwst`'s own resampling
  — there is exactly one resample either way, so this is a no-op
  simplification, not a behavioural change.
- **Measured impact**: none — `sharpness_din_st_matches_din_45692_broadband_noise_reference_values`/`_narrowband_noise_reference_values`
  (41 DIN 45692 signals, `tests/conformance_sharpness_din.rs`) and the
  differential test against installed `mosqito` both pass at ~1e-9 relative.

### D-tv-1 — `_nonlinear_decay.py:90-91` — investigated, kept as-is

- **MoSQITo**: the nonlinear-decay recurrence's very first upsampled time
  step reads its "previous" state at Python index `col - 1` with `col = 0`,
  which negative-index-wraps to the *last* upsampled column of `uo_mat` —
  still holding its initial value (the signal's *last* frame's core
  loudness, scaled down by the same ramp-to-zero its own last-frame delta
  uses) — and a `u2_mat` of exactly `0.0` (also untouched, hence its
  zero-initialized default). Concretely: the decay filter's state at time
  zero is seeded from the end of the signal, not from silence. Separately,
  the explicit pre-loop initialization of `u2_mat[:, 0]`
  (`_nonlinear_decay.py:68-69`) is provably dead code: the loop's first
  iteration (`col = 0`) unconditionally overwrites `u2_mat[:, 0]` at line
  102 (`u2_mat[:, col] = uo_mat[:, col]`) before anything downstream ever
  reads it, so that special-cased initialization has no effect on any
  output. This was flagged during the `noct_spectrum` step as "needs
  verification during the `loudness_zwtv` port"; both points above are that
  verification.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/loudness/zwtv/nonlinear_decay.rs`
  reproduces the wraparound exactly (seeding the sequential recurrence's
  initial `uo`/`u2` state from the same values Python's wraparound reaches),
  and simply omits the dead `u2` pre-initialization rather than translating
  code that can never affect a result.
- **Why not "fixed"**: unlike the other bugs in this register, ISO 532-1
  does not publish an initial-condition prescription for this recurrence to
  check the wraparound against, so "seed from silence instead" would be an
  assumption substituted for another assumption, not a standards-anchored
  correction. Its effect is also bounded: the recurrence's slowest time
  constant (`t_var` = 75 ms) is a handful of 2 kHz frames, so any distortion
  from the wraparound is confined to the first fraction of a second of
  output.
- **Measured impact**: none observable at the ISO 532-1 conformance
  tolerance. `nl_loudness_matches_mosqito_including_the_first_frame_wraparound`
  (`tests/golden_zwtv.rs`) checks exactly the short (`ntime` as low as 2)
  matrices where the wraparound dominates the output, against real
  `mosqito`, to ~1e-9 relative — the safest bit-for-bit signal this
  behaviour is reproduced correctly, independent of whether keeping it was
  the right call. And it is: all 20 ISO 532-1 Annex B.4 + B.5 reference
  signals (`tests/conformance_loudness_zwtv.rs`) pass the standard's own
  section 6.1 compliance procedure (±2 ms realignment, ≤1% of samples
  allowed outside the wider of ±5%/±0.1 sone) with this behaviour in place.

### D-ecma-loud-1 — `_preprocessing.py:28` no longer mutates its input

- **MoSQITo**: `signal[:240] *= w_fadein` applies the 5 ms raised-cosine
  fade-in *in place* on the caller's own array — the same
  mutate-a-caller's-array pattern as `conversion/amp2db.py` (D-zwst-2).
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/loudness/ecma/preprocessing.rs`
  copies `signal` before applying the fade-in.
- **Why**: same reasoning as D-zwst-2 — a pure-looking function silently
  mutating a caller's array is a footgun nothing in the pipeline needs.
- **Measured impact**: none on any output — confirmed by
  `golden_ecma_loudness.rs`'s end-to-end pipeline cases matching real
  `mosqito` to ~1e-6 relative.

### Noted, not changed — `_ecma_time_segmentation.py`'s block index formula

`_ecma_time_segmentation.py:64-65` builds each block's sample indices via
`numpy.linspace(l*sh, l*sh+sb, sb).astype(int32)` rather than the more
obvious `arange(l*sh, l*sh+sb)`: `sb` points spanning a range of length
`sb` is a step fractionally *greater* than 1 (`sb/(sb-1)`), so truncating to
an integer occasionally skips one physical sample near the end of each
block (a block of "`sb` samples" actually spans `sb+1` positions, with one
omitted). `mosqito-rs` reproduces this exactly
(`rust/crates/mosqito-core/src/loudness/ecma/specific_loudness.rs`),
verified bit-for-bit against `numpy.linspace(...).astype(int32)` across
several `(sb, sh, n_new)` combinations before ever writing the Rust version,
and further via the full-pipeline golden vectors in `golden_ecma_loudness.rs`.
This is **not** recorded as a "bug, kept for now" the way D-tv-1 is: nothing
in this port's research turned up a reason to believe it diverges from
ECMA-418-2 §5.1.5 Eqs. 18-20 (unlike `_nonlinear_decay.py`'s wraparound,
which is visibly an artifact of Python's negative-index semantics with no
plausible standards grounding). Noted here only so the behaviour is
documented rather than silently present.

### Scope: scalar `sb`/`sh` only

MoSQITo's `loudness_ecma`, `_band_pass_signals` and `_ecma_time_segmentation`
all accept either a single block/hop size or a 53-element list (one per
band, `_ecma_time_segmentation.py:40-48`), producing a potentially ragged
`N_specific`. `mosqito-rs` only implements the scalar case — the same
simplification already made for `noct_synthesis` (1-D only) and
`loudness_zwst_freq` (1-D only): every one of MoSQITo's own callers,
tests and validation scripts uses a scalar `sb`/`sh` for `loudness_ecma`.
This also means `N_specific` is a plain `(53, n_blocks)` array here, not a
list of 53 possibly-different-length arrays — a consequence of the scalar
restriction, not an independent behavioural choice.

`_ecma_time_segmentation` likewise returns `time_array` as a list of 53
per-band time axes even for a scalar `sb`/`sh` — identical across bands in
that case (same block layout for every band), but still 53 rows deep, and
MoSQITo's own example indexes it as `time_array[0]`. `mosqito-core`'s Rust
function computes the shared axis once (there is no reason to repeat 53
identical 1-D arrays internally), and `python/mosqito_rs/loudness_ecma.py`
broadcasts it to the documented `(53, Ntime)` shape before returning —
caught by a GitHub code-review bot on PR #2 (the differential test at the
time only compared `time_py[0]`, hiding the shape mismatch); fixed and the
test now compares the full array.

### Dead code not ported — `_band_pass_signals.py`'s `_rectified_band_pass_signals`

A second, unused function duplicating `_band_pass_signals` plus
rectification and `mosqito.utils.time_segmentation`'s `is_ecma=True` branch.
`loudness_ecma.py` calls only `_band_pass_signals` (via `_ecma_time_segmentation`
for the actual segmentation), never this one. Not reproduced.

### D-ecma-1 — §7.1.2 envelope downsampling method

ECMA-418-2 does not specify how to downsample the envelopes. Wanty et al.
(2024) choose decimation over Fourier resampling, to avoid assuming the
signal is periodic — sanctioned, followed as written. The paper's prose says
an "anti-aliasing FIR filter", but `roughness_ecma.py:133-134` calls
`scipy.signal.decimate` with its default IIR (Chebyshev type I) filter, not
FIR; resolved in favour of the code (matching what the published reference
values were actually produced with), via the `decimate` primitive already
golden-tested against SciPy in `dsp::filter`. Implemented in
`rust/crates/mosqito-core/src/roughness/ecma/envelope_spectrum.rs`, split as
`8*4` matching `roughness_ecma.py:133-134`'s own split (SciPy warns against a
single decimation by more than 13).

### D-ecma-2 — §7.1.5.1 Eqs. 73–76 replaced by a closed form

Wanty et al.'s analytic solution of the standard's quadratic peak-fit,
agreeing with the matrix solution to 1e-8 Hz and cheaper to compute. Taken
from MoSQITo's code (`_refinement.py:75`), not the paper's PDF — text
extraction from the PDF renders the numerator's minus sign as a plus, which
would silently invert the sub-bin correction:

```
f_p = (k_p - (Phi[k+1] - Phi[k-1]) / (2*Phi[k-1] + 2*Phi[k+1] - 4*Phi[k])) * delta_f
```

`delta_f = 1500/512`. Implemented in
`rust/crates/mosqito-core/src/roughness/ecma/refinement.rs`'s `refinement`.
Boundary cases `k=0` and `k=255` use truncated amplitude sums
(`_refinement.py:66-71`, ported verbatim) — provably unreachable from this
crate's own `peak_picking` (its search window starts at index 2 and
`find_peaks` never reports an array boundary as a peak, so `kpi` is always
in `2..=254`), kept anyway for defensiveness and fidelity to the source.

### D-ecma-3 — §7.1.5.1 Eq. 78 is wrong as published

Corrected bias adjustment (already in MoSQITo, wrong published form commented
out beside it, `_refinement.py:31-40`), implemented in `refinement.rs`'s
`rho`:

```
rho = E(theta_corr) - (E(theta_corr) - E(theta_corr-1)) * beta(theta_corr) / (beta(theta_corr) - beta(theta_corr-1))
```

`E` is the 34-float bias table from the standard's Table 10
(`_refinement.py:18-19`).

### D-ecma-4 — §7.1.7 Eq. 105 fitting curve is wrong

Uses the Sottek/Becker/Lobato (INTERNOISE 2020) form instead
(`_non_linear_transform.py:37`), implemented in
`rust/crates/mosqito-core/src/roughness/ecma/non_linear_transform.rs`:

```
E(l50) = 0.25 * tanh(1.75 * (B(l50) - 2.5)) + 0.7
```

### D-ecma-5 — calibration factor `c_R` re-derived: `0.045` → `0.03288`

A consequence of D-ecma-4 and, more directly, of fixing the unsanctioned
`_lowpass_filter.py` bug below: `c_R = 0.045` was fitted by MoSQITo *with
that bug in place*, so it could not be carried over once the bug was fixed.
Re-derivation used the fact that, with a fixed `tau`/rising-falling
classification, `R` is exactly linear in `c_R` (the classification compares
`R_hat` values scaled by the same positive `c_R`, so it — and everything
downstream — is scale-invariant in sign; only the final magnitude scales).
This meant the expensive part (the full pipeline up to `non_linear_transform`)
only needed running once per test point, at `c_R = 1`, then fit in closed
form: `c_R = sum(R1 * ref) / sum(R1^2)` against the ECMA-418-2 Annex C values
in `references.py:1132-1172`, over all 7 fc x 15 fmod points.
- **Old** (MoSQITo, with the buggy filter): `c_R = 0.045`. Measured
  (with the *corrected* filter substituted in, to isolate `c_R`'s own
  effect): only 55/105 points within ±0.1 asper of Zwicker & Fastl, mean
  38% relative error against ECMA Annex C, only 2/105 within 30% of it.
- **New** (`mosqito-rs`, with the corrected filter): `c_R = 0.03288`.
  104/105 points within ±0.1 asper of Zwicker & Fastl (max deviation 0.176,
  at fc=125/fmod=300 — the one exception); mean 2.3% relative error against
  ECMA Annex C, 104/105 within 30% of it (same one exception, at 46%).
  Gated directly in `tests/conformance_roughness_ecma.rs`.

### D-ecma-6 — amplitude floor 0.074376

Kept as-is; the paper notes the standard gives no origin for this threshold
(`roughness_ecma.py:182`). Implemented as `AMPLITUDE_FLOOR` in
`roughness_ecma.rs`.

### D-ecma-7 — edition

Targets ECMA-418-2 **2nd edition (December 2022)**. The 1st edition (2020)
had an error in Eqs. 13 & 14, corrected in the 2nd; MoSQITo 1.2 implements the
corrected version, and `mosqito-rs` does too (inherited via D-ecma-4's
gammatone/nonlinearity equations, already 2nd-edition in
`mosqito-core::loudness::ecma`).

### Bug fix (unsanctioned by the paper) — `roughness_ecma/_lowpass_filter.py`

Computes `R_spec` along `axis=-1` (across critical bands), discards it, then
computes `R_time_spec` from `R_hat[1,:]` and `tau[1,:]` — single rows
broadcast across every time frame, rather than `R_hat[1:,:]`/`tau[1:,:]`.
`R_rising` likewise diffs across bands. ECMA Eq. 109/110 specify a first-order
lowpass **over time**, τ = 0.0625 rising / 0.5 falling. This is an
unambiguous bug the INTERNOISE 2024 paper does not mention. **Fixed** in
`rust/crates/mosqito-core/src/roughness/ecma/lowpass_filter.rs`: a genuine
per-band recursive filter over time, with `tau` chosen fresh at each step
from that step's own `R_hat` vs. the filter's own previous output. Directly
coupled to D-ecma-5's `c_R` re-fit — see there for the measured effect on
conformance.

### D-ecma-8 — `_estimate_fund_mod_rate.py:59-61` `i_peak` indexing — investigated, kept as-is

- **MoSQITo**: `i_peak = np.argmax(Ai_tilde[I_max])` is an index *local* to
  the winning harmonic complex's own sub-array (position `0..len(I_max)`),
  but is then used to index the *global* `f_p` array directly (`f_p[i_peak]`)
  in Eq. 93's centre-of-gravity weighting, rather than `f_p[I_max[i_peak]]`.
  Confirmed by direct tracing (a constructed 6-peak example where `I_max =
  [0, 5]` and `argmax` picks local position 1): the published behaviour uses
  `f_p[1]` where the evidently-intended value is `f_p[5]`.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/roughness/ecma/estimate_fund_mod_rate.rs`
  reproduces the local-index behaviour exactly (`i_peak_local`, indexing
  `f_p` directly with it) — this is one of the few bugs in this register
  *not* fixed despite looking like a clear array-indexing slip rather than
  anything ECMA-418-2 Eq. 93 intends.
- **Why not fixed**: D-ecma-5's `c_R` re-fit was performed by calling
  MoSQITo's own unmodified `_estimate_fund_mod_rate` (via
  `tools/gen_reference_roughness_ecma_annex_c.py`'s companion fitting
  script), so it embeds this exact behaviour. Changing it here without
  re-fitting `c_R` again would silently invalidate that calibration; the
  effect is a secondary multiplicative correction (`w_peak`) on top of the
  harmonic-complex amplitude sum, not large enough on its own to justify
  redoing the fit for this one finding.
- **Measured impact**: `estimate_fund_mod_rate_matches_mosqito`
  (`tests/golden_roughness_ecma.rs`) checks this bit-for-bit against real
  `mosqito` on 20 synthetic multi-peak cases (including the specific
  multi-element-`I_max` case that exposed this), to ~1e-9 relative.

### `_noise_reduction.py:37`

The guard term is written `10e-10` (= 1e-9); almost certainly intended as
`1e-10`, but transcribed literally
(`rust/crates/mosqito-core/src/roughness/ecma/noise_reduction.rs`) rather
than silently corrected — no standards text was available to confirm intent,
and the term only guards a division by (near-)zero, so the practical effect
of either value is a matter of degree, not correctness. Verified bit-for-bit
against real `mosqito`'s `_noise_reduction` in `golden_roughness_ecma.rs`.

### Residual floating-point differences in the envelope/spectrum chain

`roughness_ecma`'s Hilbert-transform/decimate/FFT chain
(`envelope_spectrum.rs`) uses `rustfft`/`realfft` rather than NumPy/SciPy's
FFT and IIR filter implementations. Every algorithmic stage matches real
`mosqito` (or, for the two corrected stages, this port's own validated
reproduction) to ~1e-9 in isolation
(`golden_roughness_ecma.rs`), but chained through several FFT/IIR passes
per block, small floating-point differences accumulate — visible mainly as
a large *relative* error during a signal's near-silent attack transient
(where the true value is itself close to zero), while the standards-anchored
representative values (`R`, `R_specific`) agree to <1%. Not a deviation from
Python's *behaviour* (nothing here is a choice diverging from MoSQITo), so
not standards-relevant — recorded because `golden_roughness_ecma_pipeline.rs`
had to use a looser tolerance than this crate's other golden-vector tests,
and that's worth explaining rather than leaving as an unexplained number.

---

## Phase 2

### `am_noise_generator`'s RNG does not match NumPy's bit-for-bit

- **MoSQITo**: `mosqito/utils/am_noise_generator.py` draws its Gaussian noise
  carrier from `numpy.random.default_rng()`, freshly seeded from OS entropy
  on every call — the function is not reproducible run to run even in the
  original package.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/generators.rs`'s
  `am_noise_generator` takes an explicit `seed: u64` (via `rand`'s
  `StdRng`/`StandardNormal`), rather than reaching for system entropy
  directly. The Python wrapper (`python/mosqito_rs/generators.py`) keeps
  MoSQITo's own signature — `seed` is an added optional keyword, defaulting
  to a freshly drawn seed each call so the *default* behaviour still varies
  call to call the way MoSQITo's does.
- **Why**: matching NumPy's PCG64 bit-for-bit is a large, standards-irrelevant
  undertaking (no standard specifies this generator's carrier), and the
  function's own contract is statistical (a Gaussian carrier at a target
  RMS/modulation index), not a specific bit sequence. An explicit seed is
  strictly more useful than MoSQITo's un-seedable version.
- **Measured impact**: none on any statistical property (achieved RMS level,
  modulation index) — verified in `tests/test_generators.py`. The generated
  *samples* differ from any given MoSQITo run by construction.

### `roughness_dw`'s `hBP[i].all() != 0` guard — reproduced, not fixed

- **MoSQITo**: `roughness_dw/_roughness_dw_main_calc.py:172-173`:
  `if hBP[i].all() != 0 and hBP[i + 2].all() != 0:` gates which channel pairs
  get a cross-correlation. `.all() != 0` is equivalent to `.all()` itself (a
  redundant comparison of a bool to `0`), so this only excludes a channel pair
  when *every* sample in one row is exactly `0.0` — correct for a channel with
  literally no excitation (its `hBP` row is then exactly all-zero from
  `ifft` of an all-zero input), but a single sample landing on exactly zero
  anywhere else would silently — and incorrectly — exclude an otherwise-excited
  channel from its correlation.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/roughness/dw/main_calc.rs`
  reproduces the guard exactly (`h_bp[i].iter().all(|&v| v != 0.0)`).
- **Why**: no isolated reference exists for this guard alone — only the
  end-to-end `roughness_dw` output is validated (against the Zwicker & Fastl
  and Daniel & Weber curves), which were presumably fit/checked against
  MoSQITo's actual behaviour including this guard.
- **Measured impact**: none observed — golden-vector tests
  (`golden_roughness_dw.rs`) match real MoSQITo bit-for-bit on every case
  exercised, and the guard's failure mode (an exact-zero sample in an
  otherwise-excited channel) did not occur in any tested signal.

### `roughness_dw`'s negative-index wraparound at `ch_high == -1`

- **MoSQITo**: `_roughness_dw_main_calc.py`'s excitation-reconstruction loop
  indexes `slopes[j, i-1]` unguarded. For `i == 0`, this is `slopes[j, -1]`,
  which NumPy's negative indexing silently resolves to the *last* channel
  (46) rather than raising — but the slope *value* itself was computed with
  the literal channel index `-1` in its formula, not 46, so the value and the
  slot it lands in disagree. Only reachable when a component sits at exactly
  0 Bark (`ch_high[k] == ceil(2*0) - 1 == -1`).
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/roughness/dw/main_calc.rs`
  reproduces the same wraparound (indexes `N_CHANNEL - 1` when `i == 0` in
  that branch) for whatever input could reach it, though — unlike Python's
  silent wraparound — a *different* edge case one branch over (`slopes[j,
  i+1]` reaching index 47, which raises `IndexError` in Python) is clamped
  rather than made to panic.
- **Why**: 0 Bark is not reachable through this crate's public API (every
  `freq_axis`, from `comp_spectrum`, starts at `df > 0`, never exactly `0`
  Hz), so neither branch is exercised by any real caller; reproduced anyway
  in case a future caller of the lower-level `roughness_dw_main_calc` (public
  in `mosqito-core`, unlike Python's private `_roughness_dw_main_calc`)
  supplies a spectrum that does include it.
- **Measured impact**: none — not reachable through `roughness_dw`/
  `roughness_dw_freq`'s own signatures.

### `_main_sii.py`'s critical-band bandwidth adjustment — dead code, not ported

- **MoSQITo**: `_main_sii.py:108`: `if (method == "critical_bands") or
  (method == "equal_critical_bands"):` guards a per-band noise-spectrum
  bandwidth adjustment (`noise_spectrum -= 10*log10(upper - lower)`) — but
  the only method strings any caller can ever pass are `"critical"`/
  `"equally_critical"` (checked and rejected earlier in the same call
  chain), so this branch is unreachable dead code; the adjustment never
  fires for any input.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/speech_intelligibility/sii.rs`'s
  `main_sii` does not implement this branch at all.
- **Why**: ANSI S3.5 §4.3.2 was not consulted in enough depth during this
  port to confirm whether critical-band noise levels actually require this
  adjustment; neither of MoSQITo's own validated reference cases
  (`validations/sq_metrics/speech_intelligibility/validation_sii.py`) uses
  the critical-band procedures, so there is no corpus in this repository to
  test the alternative against either. Reproducing the dead code as written
  (i.e., omitting it) matches MoSQITo's actual, tested behaviour; flagged
  here rather than silently guessing at "correct" behaviour with nothing to
  validate it against.
- **Measured impact**: none observed — `main_sii`'s golden-vector and
  standards-conformance tests all pass; none of them exercise the critical or
  equally-critical procedures' bandwidth adjustment.

### SII's `threshold` as an explicit array — fixed, not reproduced

- **MoSQITo**: `_main_sii.py` dispatches on `threshold` with `elif threshold
  == "zwicker":`. For an array threshold, comparing an array to a string
  raises `ValueError: The truth value of an array with more than one element
  is ambiguous` — confirmed directly against the installed package, both
  through the public `sii_ansi*` wrappers and calling `_main_sii` itself.
  Real MoSQITo therefore cannot actually accept an explicit array threshold
  at all, despite documenting the parameter as `array_like or 'zwicker'`.
- **`mosqito-rs`**: `SiiThreshold::Custom` (`speech_intelligibility/sii.rs`)
  works as documented instead of reproducing the crash.
- **Why**: there is no ambiguity in the standard to preserve here — every
  non-crashing code path is unaffected by fixing this one, and reproducing a
  `ValueError` crash as the "correct" behaviour for a documented parameter
  would make that parameter unusable for no standards reason.
- **Measured impact**: `SiiThreshold::Custom` is unit-tested directly
  (`speech_intelligibility/sii.rs`'s `custom_threshold_*` tests) rather than
  differentially, since there is no working Python call to compare against.

### `freq_band_synthesis`'s axis-extension branches — initially skipped, now ported

- **MoSQITo**: `freq_band_synthesis` rebuilds its frequency axis as
  `arange(fmin.min(), fmax.max() + df, df)` and resamples the spectrum onto
  it with `numpy.interp` whenever the requested bands reach past either end
  of the input axis. The warning it prints says "Empty values will be filled
  with 0", but `numpy.interp` *clamps* to the edge value, so the extension
  repeats the spectrum's first/last value.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/slm/freq_band_synthesis.rs`
  reproduces both branches, edge-clamping included (message notwithstanding).
- **Why this is here**: an earlier version of this port documented these
  branches as unreachable and omitted them. That was wrong. `sii_ansi`'s
  octave procedure asks for bands up to 11360 Hz from a `comp_spectrum` axis
  that only reaches `fs/2`, so every `fs` below ~22.7 kHz takes the `fmax`
  branch — 16 kHz speech audio being the obvious case.
- **Measured impact**: at `fs = 16000`, octave procedure, the top band came
  out 4.0 dB low (51.003 vs. MoSQITo's 55.011 dB). That was invisible in the
  first tests written for it because SII's `k` term clamps to 0 or 1 for
  loud or quiet noise; sweeping the noise level until that band landed in
  the unclamped range showed SII itself differing by up to 7.3e-3. Now
  agrees to 2.2e-16 across all four band procedures and `fs` in {16, 20, 32,
  48} kHz. Caught in review; regression test:
  `freq_band_synthesis_matches_mosqito_beyond_the_spectrums_axis`.

### SII's custom threshold is length-checked rather than silently truncated

- **MoSQITo**: `_main_sii` crashes on an array `threshold` before this can
  matter (see the entry above), so it has no length check either.
- **`mosqito-rs`**: `main_sii` asserts a `SiiThreshold::Custom` array has one
  value per band. Without it, every `zip` downstream truncates to the
  shorter operand, silently dropping trailing bands from the SII sum and
  returning a plausible but too-low number instead of an error.
- **Measured impact**: none on any correct call; turns a silent wrong answer
  into an immediate, explicit failure. Caught in review.

### `_H_weighting.py`'s three curves share one truncated frequency range

- **MoSQITo**: `_H_weighting.py` computes the highest bin index to fill
  (`last`) from each curve's own top x-value for `H2` (358 Hz) and `H5`
  (502 Hz) — then *reuses* `H5`'s `last` for `H16`, `H21` and `H42` too,
  instead of recomputing it from their own top x-values (each 645 Hz). Every
  bin past 502 Hz's index in those three curves is left at zero, even though
  their tables define values out to 645 Hz.
- **`mosqito-rs`**: `rust/crates/mosqito-core/src/roughness/dw/h_weighting.rs`
  reproduces this exactly (see the module's own doc comment for the detailed
  trace).
- **Why**: no isolated reference for `_H_weighting` alone exists to confirm
  whether this is intentional or a transcription slip — only the end-to-end
  `roughness_dw` output is validated.
- **Measured impact**: `h_weighting_matches_mosqito`
  (`golden_roughness_dw.rs`) checks this bit-for-bit against real MoSQITo.

### Not a 100% gate — `roughness_dw` vs. the Zwicker & Fastl curve

`roughness_dw`'s own conformance gate (`conformance_roughness_dw.rs`, ±0.1
asper against the digitised Zwicker & Fastl curve) allows ~10% of its
84-point (fc, fmod) grid to fall outside tolerance, concentrated at
`fc=2000, fmod>=80`. This is not a port defect: running the installed
MoSQITo package directly on the same stimuli reproduces the same shortfall
against the same digitised curve, bit-for-bit with this port's own output
(`golden_roughness_dw.rs`). Daniel & Weber's algorithm — or specifically
MoSQITo's implementation of it — does not reach full compliance with the
Zwicker & Fastl curve at that carrier frequency; the gate reflects the
achievable pass rate rather than asserting an unreachable 100%, the same
principle `roughness_ecma`'s own ≤1-point exception budget applies.

### Residual floating-point differences in `roughness_dw`'s FFT chain

Like `roughness_ecma` (see above), `roughness_dw`'s pipeline chains several
FFT/IFFT passes (`comp_spectrum`'s own FFT, plus per-channel
excitation/envelope FFTs in `_roughness_dw_main_calc`) through `rustfft`
rather than NumPy's FFT. Every stage matches real MoSQITo bit-for-bit in
isolation and end-to-end on a captured snapshot
(`golden_roughness_dw.rs`, ~1e-6 relative), but comparing two *live* FFT
implementations directly (`tests/test_roughness_dw.py`, Rust vs. an
installed `mosqito` in the same process) shows ~1e-4 relative agreement
instead — the accumulated rounding difference is larger for `roughness_dw`'s
9600-point transforms than for the smaller cases the golden-vector snapshots
use. Not a behavioural deviation; recorded because the differential test
needed a looser tolerance than this crate's other differential checks.

## Phase 2 — tonality (TNR/PR)

`tone_to_noise_ecma`/`prominence_ratio_ecma` share one `tonality` module
(critical bands, threshold of hearing, spectrum smoothing, candidate
screening, within-band tie-breaking), with `tnr`/`pr` as thin orchestration
on top, mirroring `sharpness::din`'s four-weightings-share-one-integral
shape. TNR/PR has no standards-anchored numeric reference anywhere in
MoSQITo's own repository — only a single orphan wav
(`validations/sq_metrics/tonality_tnr_pr/white_noise_tone_at_442_Hz.wav`)
with no validation script, confirmed by directory listing. Validated instead
via extensive golden vectors captured directly from MoSQITo's private
functions at every stage (`_critical_band`/`_lower_critical_band`/
`_upper_critical_band`, `_LTH`, `_getFrequencies`, `_spectrum_smoothing`,
`_screening_for_tones`, `_tnr_main_calc`/`_pr_main_calc` for both a single
spectrum and multiple segments sharing one frequency axis) and the full
`tnr_ecma_*`/`pr_ecma_*` entry points on a two-tone-plus-noise stimulus, plus
a differential pytest cross-check through the public Python API
(`tests/test_tonality.py`).

- **`tnr_ecma_perseg.py:127` / `pr_ecma_perseg.py:129`** — the
  pre-segmented-input (2-D signal) branch references an undefined `sig` (a
  real `NameError`, confirmed by direct read and by triggering it), never
  exercised by any test in MoSQITo. Not ported: this port's
  `tnr_ecma_perseg`/`pr_ecma_perseg` implement the 1-D-signal branch only,
  the same scope-narrowing already applied to `loudness_zwst_freq`/
  `noct_synthesis`'s 2-D cases.
- **`pytest.ini:8`** — `roughness_dw_freq:` is unindented, so that marker is
  never registered by pytest. Python-repo-only; irrelevant to
  `rust/pyproject.toml`'s own (correct) marker list.
- **`_screening_for_tones`'s `"not-smoothed"` method** — the Aures/Terhardt
  alternative to the Bray & Caspary `"smoothed"` criteria. Every real caller
  (`_tnr_main_calc`, `_pr_main_calc`) hardcodes `method="smoothed"`, so
  `"not-smoothed"` is dead code from the public API's perspective. Not
  ported, matching this project's precedent of narrowing to what's actually
  reachable (e.g. `noct_synthesis`'s unexercised 2-D sub-cases).
- **`_spectrum_smoothing`'s `.ravel()`-order mismatch.** Python ravels its
  `freqs_in` argument (shape `(nseg, nfreqs)`) **segment-major** but
  separately ravels `spec` (already transposed to `(nfreqs, nseg)` at the
  call site) **frequency-major**, then indexes both flat arrays with the
  same position-based `bin_index`/`stop` machinery — implicitly assuming a
  flat position means the same thing in both, which is false for `nseg >
  1`. Very likely an unintentional bug (a `.T` before `.ravel()` changing
  iteration order is an easy mistake), but it's real MoSQITo behaviour with
  no standard to contradict it, so `mosqito-core::tonality::spectrum_smoothing`
  reproduces it exactly — see the function's own doc comment. Confirmed
  behaviourally significant (not inert) by testing: an earlier attempt at a
  "corrected" single consistent raveling order failed the multi-segment
  golden vectors captured from real MoSQITo.
- **`_spectrum_smoothing`'s `nb_bands`/`i` bookkeeping bug.** Its `while
  nb_bands > 0` loop decrements `nb_bands` *twice* when a 1/24-octave band's
  frequency range matches no spectral bin (once inside the `if`, once
  unconditionally after) but only removes *one* array element and always
  advances `i` by exactly one — so the loop can terminate before visiting
  every band, and does not re-align `i` to the post-removal array shape
  either. Reproduced exactly (see the function's doc comment); not "fixed",
  since it's paired with the deliberately-not-reproduced item below and
  changing one without the other would invalidate the golden-vector
  comparison this was validated against.
- **`_spectrum_smoothing`'s uninitialised-memory placement gaps — not
  reproduced.** The bug above (plus coarse low/high frequency-to-grid-index
  snapping near the low-frequency edge) can leave some output positions
  never written by any band's `smooth_spec[low:high, i] = ...` slice.
  Real MoSQITo's `smooth_spec = numpy.empty(...)` then returns whatever was
  already in that memory — confirmed directly (a subnormal float and other
  implausible values were observed at those positions, not a stable or
  algorithmically meaningful result). This port fills any such gap with
  `0.0` instead: a defined, deterministic choice. Verified inert for every
  signal in `golden_tonality.rs`/`test_tonality.py`: the screening criterion
  this feeds (`spec_db[temp] > smooth_spec[temp] + 6`) only ever reads it at
  genuine local-maxima positions, none of which fell in the observed
  low-frequency/Nyquist-edge dead zones (9 of 943 positions in the
  multi-segment golden case). Reproducing non-deterministic memory contents
  bit-for-bit isn't meaningful to begin with — there is no single "correct"
  value to match.
- **A fresh `_getFrequencies` port, not a reuse of `slm::center_freq`/
  `filter_bandwidth`.** Phase 1's `center_freq` rounds `log(f/fr)/log(u)` to
  the nearest band number and `filter_bandwidth` derives bandwidth from a
  Butterworth quality factor — genuinely different formulas from
  `_getFrequencies`'s direct increment-and-test loop and its `G^(±1/(2b))`
  band-edge ratio. Reusing the Phase 1 functions here would have been an
  unverified assumption of equivalence between two different algorithms;
  `mosqito-core::tonality::get_frequencies` instead matches
  `_getFrequencies` line for line.
- **`_lower_critical_band`/`_upper_critical_band`'s confusingly-renamed
  tuple unpacking.** `_lower_critical_band` does `f2, _ = _critical_band(f0)`
  — assigning `_critical_band`'s **first** return value (the central band's
  lower edge) to a local variable it calls `f2`. `_upper_critical_band`
  does the mirror: `_, f1 = _critical_band(f0)`, assigning the central
  band's **upper** edge to a local `f1`. Not a bug — the lower/upper bands
  genuinely are meant to be contiguous with the central one at exactly
  those edges, confirmed against real MoSQITo output (`lf2 == cf1`, `uf1 ==
  cf2` to machine precision) — but the naming inverted this port's first
  attempt at a direct translation (`critical_band`'s tuple element mapped to
  the wrong local), caught by the `critical_band` golden vectors. Recorded
  since it's exactly the kind of transcription risk this project's "read
  the Python source directly" discipline exists to catch.
- **`_tnr_main_calc`'s `delta_ftot` indexing quirk.** `low_limit_idx`/
  `high_limit_idx` are computed as positions in the *frequency-of-interest-
  filtered* `fr` array (`argmin(abs(fr - f1))`), but then used directly to
  index the *original, unfiltered* `frs`/`freq_axis` array
  (`frs[high_limit_idx] - frs[low_limit_idx]`) without adjusting for the
  filter's offset. Reproduced exactly in
  `mosqito-core::tonality::tnr::evaluate_segment` (see its own comment) —
  real MoSQITo behaviour, no standard to check it against.
- **`_screening_for_tones`'s `stop` table — an off-by-one this port got
  wrong, now fixed.** MoSQITo builds `stop = arange(1, n+1) * m - 1` for
  several segments (and `[m]` for one), then picks the segment whose
  `stop[i] - m < peak_index < stop[i]`. An earlier version of this port used
  `i * m`, which shifted the matched window by one and left each segment's
  *first* bin unmatched; that bin then fell back to segment 0 and was
  recorded with its un-offset flat index — out of range for segment 0's own
  arrays, and an out-of-bounds panic once `tnr_main_calc` indexed with it.
  Now computed as `peak_index / m`, which equals Python's answer for every
  index Python's scan matches, and gives the one index it leaves unmatched
  (each segment's *last* bin, where Python falls through to a stale `block`
  or raises `UnboundLocalError`) its own correct segment. Caught in review;
  regression test:
  `screening_attributes_a_segment_boundary_candidate_to_its_own_segment`.
- **`_screening_for_tones`'s `low_limit` can go negative — now reproduced
  rather than underflowing.** `low_limit` starts at the candidate's original
  position, but the left-hand scan starts from wherever the right-hand scan
  moved the peak to, so it can be decremented past zero. Python lets it go
  negative and then negative-indexes `freqs` with it, wrapping to the end of
  the flat array; this port held it in a `usize`, which underflowed (a panic
  in debug, an out-of-bounds index in release). Now an `isize` read through
  a `python_index` helper that reproduces the wraparound. Caught in review.
- **`_find_highest_tone`'s `nb_tones` counter — simplified, not a
  behaviour change.** Python threads an explicit `nb_tones` integer
  alongside the candidate array through the recursion; every place that
  mutates one mutates the other identically, so it is always exactly
  `len(candidates)`. This port drops the parallel counter and lets callers
  read `.len()` instead.
- ECMA-418-2 **specific** roughness (per-band, not aggregate `R`) is validated
  in MoSQITo only against commercial HEAD Artemis output at ±10%
  (`validation_specific_roughness_ecma.xlsx`), not against the standard
  itself. (Carried over from Phase 1; still unaddressed, out of scope for
  either phase without access to that commercial reference.)

## Phase 3 — API completeness

### `time_segmentation` — exposed, `is_ecma=True` not supported

- **MoSQITo**: `mosqito/utils/time_segmentation.py`'s public
  `time_segmentation(sig, fs, nperseg=2048, noverlap=None, is_ecma=False)`
  has two code paths: the ordinary one (`is_ecma=False`), used by every real
  caller outside `loudness_ecma` (`roughness_dw`, `tnr_ecma_perseg`,
  `pr_ecma_perseg`), and an ECMA-418-2 §5.1.4 Eq. 16 variant (`is_ecma=True`,
  zero-pads the signal by one block length before segmenting) used only
  internally by `loudness_ecma/_band_pass_signals.py`.
- **`mosqito-rs`**: `mosqito-core::utils::time_segmentation` (already used
  internally by this port's `roughness_dw` and tonality `_perseg` entry
  points) is now also exposed publicly as `mosqito_rs.time_segmentation`,
  matching MoSQITo's signature — but only the `is_ecma=False` path;
  `is_ecma=True` raises `NotImplementedError`. This port's `loudness_ecma`
  implements its own block-index segmentation directly (see the "Noted, not
  changed — `_ecma_time_segmentation.py`'s block index formula" entry above)
  rather than routing through this shared function with a flag, so there is
  no `is_ecma=True` code path here to expose in the first place.
- **Why**: matches this project's established precedent of narrowing a
  ported function to what's actually reachable (e.g. `noct_synthesis`'s
  unported 2-D case, `loudness_zwst_freq`'s 1-D-only scope) rather than
  building a code path nothing outside `loudness_ecma`'s own internals uses.
- **Measured impact**: none — `is_ecma=False` is verified against real
  MoSQITo output in `tests/test_utils.py`; `is_ecma=True` was never
  reachable through this shared function in the first place for this port.

### `load` — ported for `.wav` only; `isoclose` not ported

- **MoSQITo**: `mosqito.utils.load` reads a `.wav`/`.mat`/`.uff` file,
  resamples to 48 kHz, and applies a calibration factor — file I/O, not a
  psychoacoustic algorithm; `.uff` support pulls in `pyuff`, a niche
  dependency for LMS Test.Lab-style file interchange, and `.mat` needs
  `scipy.io.loadmat` plus caller-supplied variable names. `mosqito.utils.isoclose`
  is a test/compliance-plotting helper (ISO 532-1 §5.1-style tolerance-band
  comparison) with a hard `matplotlib` dependency, used only in MoSQITo's own
  validation scripts, not by any metric.
- **`mosqito-rs`**: `python/mosqito_rs/utils.py`'s `load` is a thin
  **pure-Python** function (no Rust work — file I/O and a one-time
  `scipy.signal.resample` call, not a hot path) matching MoSQITo's `.wav`
  branch exactly, including its int16/int32/float calibration and the
  resample-to-48kHz step; `.mat`/`.uff` raise `NotImplementedError`.
  `isoclose` is not ported at all. This port's own test suite separately
  reimplements the same `.wav` calibration where it needs it without
  depending on either (`tests/conftest.py`'s `load_wav_calibrated`), and
  uses plain `numpy.testing` tolerance assertions in place of `isoclose`'s
  plotting-oriented comparison.
- **Why**: `load`'s `.wav` case is cheap to match exactly and completes the
  "switching is an import change" promise for the common case; `.mat`/`.uff`
  and `isoclose` are one-time, non-hot-path utilities better served by
  calling `scipy.io.loadmat`/`numpy.testing` directly than by adding
  `pyuff`/`matplotlib` dependencies to this project for functionality no
  metric here depends on.
- **Measured impact**: none on any metric — differential-tested directly
  against `mosqito.utils.load` in `tests/test_utils.py`.
