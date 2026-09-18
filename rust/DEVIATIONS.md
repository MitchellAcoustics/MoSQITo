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
same correction. Entries below are grouped by status: **implemented**,
**planned** (identified, not yet reached in the port order), and **deferred**
(Phase 2 scope).

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

### Dead code not ported — `_band_pass_signals.py`'s `_rectified_band_pass_signals`

A second, unused function duplicating `_band_pass_signals` plus
rectification and `mosqito.utils.time_segmentation`'s `is_ecma=True` branch.
`loudness_ecma.py` calls only `_band_pass_signals` (via `_ecma_time_segmentation`
for the actual segmentation), never this one. Not reproduced.

---

## Planned (identified during exploration, not yet implemented)

### D-ecma-1 — §7.1.2 envelope downsampling method

ECMA-418-2 does not specify how to downsample the envelopes. Wanty et al.
(2024) choose decimation over Fourier resampling, to avoid assuming the
signal is periodic — sanctioned, to be followed as written.
**Open discrepancy to resolve when implemented**: the paper's prose says an
"anti-aliasing FIR filter", but `roughness_ecma.py:133-134` calls
`scipy.signal.decimate` with its default IIR (Chebyshev type I) filter, not
FIR. `mosqito-rs` will match the code (IIR, via the `decimate` primitive
already implemented and golden-tested in `dsp::filter`), and this entry will
record the conflict as resolved in favour of the code once the roughness port
lands.

### D-ecma-2 — §7.1.5.1 Eqs. 73–76 replaced by a closed form

Wanty et al. give the analytic solution of the standard's quadratic peak-fit,
agreeing with the matrix solution to 1e-8 Hz and cheaper to compute. Already
implemented in MoSQITo at `_refinement.py:75`:

```
f_p = (k_p - (Phi[k+1] - Phi[k-1]) / (2*Phi[k-1] + 2*Phi[k+1] - 4*Phi[k])) * delta_f
```

`delta_f = 1500/512`. **Must be taken from the code, not the paper's PDF**:
text extraction from the PDF renders the numerator's minus sign as a plus,
which would silently invert the sub-bin correction. Boundary cases `k=0` and
`k=255` use truncated amplitude sums (`_refinement.py:66-71`).

### D-ecma-3 — §7.1.5.1 Eq. 78 is wrong as published

Corrected bias adjustment (already in MoSQITo, wrong published form commented
out beside it, `_refinement.py:31-40`):

```
rho = E(theta_corr) - (E(theta_corr) - E(theta_corr-1)) * beta(theta_corr) / (beta(theta_corr) - beta(theta_corr-1))
```

`E` is the 34-float bias table from the standard's Table 10
(`_refinement.py:18-19`).

### D-ecma-4 — §7.1.7 Eq. 105 fitting curve is wrong

Use the Sottek/Becker/Lobato (INTERNOISE 2020) form instead
(`_non_linear_transform.py:37`):

```
E(l50) = 0.25 * tanh(1.75 * (B(l50) - 2.5)) + 0.7
```

### D-ecma-5 — calibration factor `c_R`

A consequence of D-ecma-4. The standard permits only 0.25% variation on `c_R`;
MoSQITo's `c_R = 0.045` deviates far beyond that (`_non_linear_transform.py`).
**Coupled to an unsanctioned bug** (see below): `c_R = 0.045` was fitted with
that bug in place, so it cannot be carried over unchanged once the bug is
fixed. When the roughness port lands, `c_R` will be re-derived by fitting
against the ECMA Annex C reference values in
`validations/sq_metrics/roughness_ecma/input/references.py:1132-1172`, and
both the old and new constants will be recorded here with fit residuals.

### D-ecma-6 — amplitude floor 0.074376

Kept as-is; the paper notes the standard gives no origin for this threshold
(`roughness_ecma.py:182`).

### D-ecma-7 — edition

Targets ECMA-418-2 **2nd edition (December 2022)**. The 1st edition (2020)
had an error in Eqs. 13 & 14, corrected in the 2nd; MoSQITo 1.2 implements the
corrected version, and `mosqito-rs` will too.

### Bug fix (unsanctioned by the paper) — `roughness_ecma/_lowpass_filter.py`

Computes `R_spec` along `axis=-1` (across critical bands), discards it, then
computes `R_time_spec` from `R_hat[1,:]` and `tau[1,:]` — single rows
broadcast across every time frame, rather than `R_hat[1:,:]`/`tau[1:,:]`.
`R_rising` likewise diffs across bands. ECMA Eq. 109/110 specify a first-order
lowpass **over time**, τ = 0.0625 rising / 0.5 falling. This is an
unambiguous bug the INTERNOISE 2024 paper does not mention. **Directly
coupled to D-ecma-5** — see there.

### `_noise_reduction.py:37`

The guard term is written `10e-10` (= 1e-9); almost certainly intended as
`1e-10`. Low impact; will be decided and recorded (not silently transcribed)
when the roughness port lands.

---

## Deferred to Phase 2 (recorded now so they aren't lost)

- `roughness_dw/_roughness_dw_main_calc.py:173` — `hBP[i].all() != 0` compares
  a bool to `0`, which is true almost always; the guard is effectively
  inert.
- `_main_sii.py:108` — compares `method` against `"critical_bands"` /
  `"equal_critical_bands"`, values the validator never allows — dead code.
- `tnr_ecma_perseg.py:127` / `pr_ecma_perseg.py:129` — the pre-segmented-input
  branch references an undefined `sig`.
- `pytest.ini:8` — `roughness_dw_freq:` is unindented, so that marker is
  never registered by pytest.
- TNR/PR has no standard-anchored reference in the MoSQITo repository at all
  — only regression pins against MoSQITo's own past output at `decimal=7`.
  Under a standards-conformance target these pins are not expected to hold;
  whether to build an ECMA-74 worked-example reference or defer TNR/PR
  indefinitely is an open decision for Phase 2, not a deviation to record yet.
- ECMA-418-2 **specific** roughness (per-band, not aggregate `R`) is validated
  in MoSQITo only against commercial HEAD Artemis output at ±10%
  (`validation_specific_roughness_ecma.xlsx`), not against the standard
  itself.
