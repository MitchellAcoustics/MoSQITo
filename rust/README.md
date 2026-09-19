# mosqito-rs

Fast, standards-conformant psychoacoustic sound quality metrics, in Rust with
Python bindings.

This is a port of [MoSQITo](https://github.com/Eomys/MoSQITo)'s metrics. The
Python API mirrors MoSQITo's signatures exactly, so switching is an import
change. The Rust core is usable on its own as the
[`mosqito-core`](crates/mosqito-core) crate, with no Python dependency.

## What it targets

Conformance is to the **published standards** — ISO 532-1, ECMA-418-2 (2nd
edition, 2022), DIN 45692 — rather than to MoSQITo's Python output. Where the
two differ, the standard wins and the divergence is recorded in
[`DEVIATIONS.md`](DEVIATIONS.md). For ECMA-418-2 roughness, the corrections
follow Wanty, Glesser & Casagrande Hirono, *"ECMA-418-2 roughness, a
challenging implementation"*, INTER-NOISE 2024 — plus one further,
unsanctioned bug fix (`_lowpass_filter.py`) and a re-derived calibration
factor, both recorded in `DEVIATIONS.md`.

## What's implemented (Phase 1)

| Metric | Standard | Conformance gate |
| --- | --- | --- |
| `noct_spectrum` / `noct_synthesis` | ANSI S1.1-1986 | Direct cross-check against MoSQITo, 1e-6 relative |
| `loudness_zwst` (+ `_freq`, `_perseg`) | ISO 532-1:2017 | Annex B reference values, ±5%/±0.1 sone |
| `sharpness_din` (all 5 variants) | DIN 45692:2009 | Chapter 6 reference signals, ±5%/±0.05 acum |
| `loudness_zwtv` | ISO 532-1:2017 | Annex B.4/B.5, ±5%/±0.1 sone, ≤1% samples outside |
| `loudness_ecma` | ECMA-418-2:2022 §5 | Golden vectors + 40-phon anchor (no digitized standard corpus exists) |
| `roughness_ecma` | ECMA-418-2:2022 §7.1 | Annex C (all 7 fc × 15 fmod) + Zwicker-Fastl reference curve |

Every metric is also validated against golden vectors captured from the real
`mosqito` package, and (for the metrics with an independent reference
implementation) cross-checked against `sottek-hearing-model`.

## What's implemented (Phase 2)

| Metric | Standard | Conformance gate |
| --- | --- | --- |
| `sii_ansi` / `sii_ansi_freq` / `sii_ansi_level` | ANSI S3.5-1997 | Standard's own worked example, wider of ±1% or ±0.01 |
| `roughness_dw` (+ `_freq`) | Daniel & Weber (1997) | Zwicker-Fastl reference curve, ±0.1 asper, ≥90% of the 84-point (fc, fmod) grid (MoSQITo's own implementation does not reach 100% either — confirmed directly) |
| `utils` conversions (`bark2freq`, `freq2bark`, `db2amp`, `spectrum2dBA`, `LTQ`) | — | Golden vectors against MoSQITo |
| `sound_level_meter` (`comp_spectrum`, `freq_band_synthesis`) | — | Golden vectors against MoSQITo |
| Signal generators (`sine_wave`, `am_sine`, `am_noise`, `fm_sine`) | — | Golden vectors / statistical checks against MoSQITo (`am_noise_generator`'s RNG is seeded, not bit-parity with `numpy.random.default_rng` — see `DEVIATIONS.md`) |
| `sone_to_phon`, `equal_loudness_contours` | ISO 226 | Golden vectors against MoSQITo |
| `tnr_ecma_st` / `_freq` / `_perseg` | ECMA-74 Annex D, ECMA TR/108 | Golden vectors against MoSQITo (private-function-level and full entry points; no digitized standard corpus exists) |
| `pr_ecma_st` / `_freq` / `_perseg` | ECMA-74 Annex D, ECMA TR/108 | Golden vectors against MoSQITo (private-function-level and full entry points; no digitized standard corpus exists) |
| `time_segmentation` | — | Differential test against MoSQITo (`is_ecma=False` case only — see `DEVIATIONS.md`) |

Phase 2 is now complete. `tnr_ecma_perseg`/`pr_ecma_perseg` only implement
the 1-D-signal branch of their MoSQITo counterparts — the 2-D-signal branch
has a real, unreproduced `NameError` in MoSQITo itself; see `DEVIATIONS.md`.

## Layout

| Path | What it is |
| --- | --- |
| `crates/mosqito-core` | Pure Rust, no Python. Publishable to crates.io. |
| `crates/mosqito-py` | PyO3 bindings; marshalling only, no algorithms. |
| `python/mosqito_rs` | Python package mirroring MoSQITo's public API. |
| `tools/gen_golden*.py` | Regenerates the golden-vector reference files. |
| `tools/gen_reference_*.py` | Extracts standards reference values (ISO/ECMA Annexes) into JSON the Rust conformance tests read. |
| `tools/refit_c_r.py` | Re-derives ECMA-418-2 roughness's `c_R` calibration factor (see `DEVIATIONS.md`). |

The split keeps the Rust crate independently useful and leaves the door open to
folding this into `mosqito` itself later as a compiled core.

## Development

```bash
# From the repository root: create the dev environment once.
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy scipy maturin pytest

cd rust
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release   # unit + golden-vector + standards conformance

../.venv/bin/maturin develop --release   # build and install into the venv
../.venv/bin/pytest tests/ -v -m conformance
```

### Differential and benchmark tiers

```bash
# Diagnostic-only cross-checks against installed mosqito / sottek-hearing-model
# (not gating — never auto-failed, since the standard, not either Python
# implementation, is the authority).
uv pip install --python ../.venv/bin/python mosqito sottek-hearing-model
../.venv/bin/pytest tests/ -v -m differential

# Wall-time benchmarks. Criterion (mosqito-core only, covers every metric
# from both phases, 10 s signals for several, rayon thread-count scaling):
cargo bench -p mosqito-core   # HTML reports under target/criterion/

# pytest-benchmark (mosqito vs. mosqito_rs on identical inputs):
uv pip install --python ../.venv/bin/python pytest-benchmark mosqito
../.venv/bin/pytest tests/test_benchmark.py -v -m benchmark --benchmark-only
```

## CI

`.github/workflows/`:
- `ci.yml` — on every push/PR: `cargo fmt --check`, `cargo clippy -D
  warnings`, the full Rust test suite (including standards conformance), then
  `maturin develop` + the Python conformance tier.
- `wheels.yml` — on a `v*` tag or manual dispatch: builds wheels for Linux
  (x86_64/aarch64), macOS (x86_64/aarch64) and Windows via
  `PyO3/maturin-action`, abi3-py39 (one wheel per platform covers CPython
  3.9–3.13), then installs the Linux wheel and re-runs the conformance suite
  against it before uploading artifacts. Does not publish to PyPI.
- `bench.yml` — manual or nightly: Criterion + pytest-benchmark, uploaded as
  artifacts.

### Why the DSP primitives follow SciPy

`mosqito-core::dsp` reimplements `lfilter`, `sosfilt`, `filtfilt`,
`sosfiltfilt`, `decimate`, `hilbert`, `resample`, `butter`, `cheby1`,
`find_peaks` prominence, PCHIP and the linear-interpolation percentile, matching
SciPy's semantics rather than a textbook's.

That is deliberate. The standards leave several steps unspecified — ECMA-418-2
§7.1.2 does not say how to downsample the envelopes, for instance — and the
reference values these metrics are validated against were produced through
SciPy. Matching SciPy is what makes the published numbers reachable.

`tools/gen_golden.py` captures SciPy's output into
`crates/mosqito-core/tests/golden.json`, and the tests in
`crates/mosqito-core/tests/golden.rs` assert against it. The file records the
numpy and scipy versions used, so a future mismatch can be attributed to a SciPy
change rather than to a port bug.
