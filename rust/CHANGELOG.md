# Changelog

All notable changes to `mosqito-core`/`mosqito-rs` are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/) once a first
version is actually released.

Both crates (`mosqito-core`, `mosqito-py`) and the `mosqito-rs` Python
package are versioned together and currently sit at `0.1.0` — no version has
been published to crates.io or PyPI yet, so this file's job for now is to
keep an accurate record ready for whenever that first release happens,
rather than to track a real release history.

## [Unreleased]

### Added — Phase 1

DSP foundation (`dsp::design`, `dsp::fft`, `dsp::filter`, `dsp::interp`,
`dsp::peaks`, `dsp::stats`, `dsp::windows`), `noct_spectrum`/`noct_synthesis`
(ANSI S1.1-1986), `loudness_zwst` (+ `_freq`, `_perseg`, ISO 532-1
stationary), `sharpness_din` (5 variants, DIN 45692), `loudness_zwtv`
(ISO 532-1 time-varying), `loudness_ecma` (ECMA-418-2 §5), `roughness_ecma`
(ECMA-418-2 §7.1, with the INTERNOISE 2024 corrections from Wanty, Glesser &
Casagrande Hirono). CI (`ci.yml`, `wheels.yml`, `bench.yml`) and the
Criterion/pytest-benchmark harness.

### Added — Phase 2

`utils::conversion` (`bark2freq`, `freq2bark`, `db2amp`, `spectrum2dBA`),
`utils::LTQ`, signal generators (`sine_wave`, `am_sine`, `am_noise`,
`fm_sine`), `slm::comp_spectrum`, `slm::freq_band_synthesis`,
`loudness::utils` (`sone_to_phon`, `equal_loudness_contours`),
`speech_intelligibility` (`sii_ansi` family, ANSI S3.5-1997), `roughness_dw`
(+ `_freq`, Daniel & Weber 1997), tonality (`tnr_ecma_*`/`pr_ecma_*`,
ECMA-74 Annex D + ECMA TR/108), `time_segmentation` (public binding), `load`
(pure-Python, `.wav` case).

### Notes

- Every deliberate divergence from MoSQITo's Python behaviour is recorded in
  [`DEVIATIONS.md`](DEVIATIONS.md), not repeated here.
- Benchmarked speedups over the pure-Python `mosqito` package (pytest-benchmark,
  mean times): `loudness_zwst` 4.9x, `loudness_zwtv` 73x, `sharpness_din_st`
  5.9x, `loudness_ecma` 11.1x, `roughness_ecma` 2.8x, `sii_ansi` 2.8x,
  `roughness_dw` 4.8x, `tnr_ecma_st` 2.4x, `pr_ecma_st` 2.4x.

[Unreleased]: https://github.com/MitchellAcoustics/MoSQITo/compare/master...claude/phase-2-rust-port
