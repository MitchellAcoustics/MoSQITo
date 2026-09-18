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

### `_calc_slopes.py:112`

`zup_ea` gets a trailing `0` appended and is indexed with `i-1`, relying on
Python negative-index wraparound at `i=0`. Not a numeric deviation — the
Rust port will make the same lookup explicit rather than relying on
wraparound, with no behavioural change. Will land with the `loudness_zwst`
port.

### `_nonlinear_decay.py:90-91`

`col-1` at `col=0` wraps to the last column of `uo_mat`/`u2_mat`. Needs
verification during the `loudness_zwtv` port as to whether this is load-bearing
(the arrays may be zero-initialized such that it's inert) or a real bug;
recorded here as a flag to check, not yet a resolved entry.

### `sharpness_din_from_loudness.py:114/146`

`UnboundLocalError` for 2-D `N` — `ind` is only bound when `N.ndim <= 1` but
read unconditionally later. Will be fixed (not reproduced) when the
`sharpness_din` port lands, since there is no standards question here at all.

### `sharpness_din_st.py:97-113`

Duplicated dead resample block (copy-paste artifact, not executed). Not
reproduced.

### `conversion/amp2db.py:25`

Mutates its input array in place. `mosqito-rs`'s conversions (Phase 2) will
not do this; noted so nobody relies on the mutation accidentally being
preserved.

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
