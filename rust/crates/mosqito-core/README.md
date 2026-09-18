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
challenging implementation"*, INTER-NOISE 2024.

## Layout

| Path | What it is |
| --- | --- |
| `crates/mosqito-core` | Pure Rust, no Python. Publishable to crates.io. |
| `crates/mosqito-py` | PyO3 bindings; marshalling only, no algorithms. |
| `python/mosqito_rs` | Python package mirroring MoSQITo's public API. |
| `tools/gen_golden.py` | Regenerates the SciPy reference vectors. |

The split keeps the Rust crate independently useful and leaves the door open to
folding this into `mosqito` itself later as a compiled core.

## Development

```bash
# From the repository root: create the dev environment once.
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy scipy maturin

cd rust
cargo test --workspace          # unit + SciPy golden-vector tests
../.venv/bin/maturin develop    # build and install into the venv
```

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
