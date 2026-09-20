"""Differential tests for `mosqito_rs`'s tonality (TNR/PR) functions,
through the Python API.

The deeper conformance gate (private-function-level `_tnr_main_calc`/
`_pr_main_calc`/`_screening_for_tones`/`_spectrum_smoothing`, both the
single-spectrum and multi-segment-shared-frequency-axis cases, plus the
full entry points) lives at the Rust level
(`crates/mosqito-core/tests/golden_tonality.rs`); this file differentially
checks the Python wrapper glue (unit conversion, argument dispatch, the
`prominence` mask) against an installed `mosqito`.

TNR/PR has no digitised standards-anchored reference corpus in MoSQITo's
own repository (only a single orphan wav under `validations/`, no script —
confirmed during Phase 2 research), so unlike the other Phase 2 metrics
there is no separate Rust-level conformance test beyond the golden-vector
comparison against real MoSQITo output.

Only `tnr_ecma_perseg`/`pr_ecma_perseg`'s 1-D-signal branch is ported (see
`DEVIATIONS.md` for the 2-D-signal branch's real, unreproduced `NameError`
in MoSQITo), so this file only exercises that branch.
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs


def _two_tone_stimulus(fs, duration, dB=60, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(0, duration, 1 / fs)
    stimulus = (
        np.sin(2 * np.pi * 1000 * time)
        + 0.5 * np.sin(2 * np.pi * 3000 * time)
        + rng.normal(0, 0.3, len(time))
    )
    rms = np.sqrt(np.mean(stimulus**2))
    ampl = 0.00002 * 10 ** (dB / 20) / rms
    return stimulus * ampl


@pytest.mark.differential
def test_tnr_ecma_st_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 24000
    stimulus = _two_tone_stimulus(fs, 0.5, seed=1)

    t_rs, tnr_rs, prom_rs, tf_rs = mosqito_rs.tnr_ecma_st(stimulus, fs, prominence=False)
    t_py, tnr_py, prom_py, tf_py = mosqito.sq_metrics.tnr_ecma_st(stimulus, fs, prominence=False)

    np.testing.assert_allclose(t_rs, t_py, rtol=1e-6)
    np.testing.assert_allclose(tnr_rs, np.asarray(tnr_py, dtype=float), rtol=1e-6)
    np.testing.assert_array_equal(prom_rs, np.asarray(prom_py, dtype=bool))
    np.testing.assert_allclose(tf_rs, np.asarray(tf_py, dtype=float), rtol=1e-6)


@pytest.mark.differential
def test_tnr_ecma_st_prominence_filter_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 24000
    stimulus = _two_tone_stimulus(fs, 0.5, seed=1)

    t_rs, tnr_rs, prom_rs, tf_rs = mosqito_rs.tnr_ecma_st(stimulus, fs, prominence=True)
    t_py, tnr_py, prom_py, tf_py = mosqito.sq_metrics.tnr_ecma_st(stimulus, fs, prominence=True)

    np.testing.assert_allclose(t_rs, t_py, rtol=1e-6)
    np.testing.assert_allclose(tnr_rs, np.asarray(tnr_py, dtype=float), rtol=1e-6)
    np.testing.assert_array_equal(prom_rs, np.asarray(prom_py, dtype=bool))
    np.testing.assert_allclose(tf_rs, np.asarray(tf_py, dtype=float), rtol=1e-6)


@pytest.mark.differential
def test_tnr_ecma_freq_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 24000
    stimulus = _two_tone_stimulus(fs, 0.5, seed=2)
    spec, freqs = mosqito.sound_level_meter.comp_spectrum(stimulus, fs, db=False)

    t_rs, tnr_rs, prom_rs, tf_rs = mosqito_rs.tnr_ecma_freq(np.abs(spec), freqs, prominence=False)
    t_py, tnr_py, prom_py, tf_py = mosqito.sq_metrics.tnr_ecma_freq(
        np.abs(spec), freqs, prominence=False
    )

    np.testing.assert_allclose(t_rs, t_py, rtol=1e-6)
    np.testing.assert_allclose(tnr_rs, np.asarray(tnr_py, dtype=float), rtol=1e-6)
    np.testing.assert_array_equal(prom_rs, np.asarray(prom_py, dtype=bool))
    np.testing.assert_allclose(tf_rs, np.asarray(tf_py, dtype=float), rtol=1e-6)


@pytest.mark.differential
def test_tnr_ecma_perseg_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 24000
    stimulus = _two_tone_stimulus(fs, 1.2, seed=3)

    t_rs, tnr_rs, promi_rs, freqs_rs, time_rs = mosqito_rs.tnr_ecma_perseg(
        stimulus, fs, prominence=False, overlap=0.5
    )
    t_py, tnr_py, promi_py, freqs_py, time_py = mosqito.sq_metrics.tnr_ecma_perseg(
        stimulus, fs, prominence=False, overlap=0.5
    )

    np.testing.assert_allclose(t_rs, t_py, rtol=1e-6)
    np.testing.assert_allclose(freqs_rs, freqs_py, rtol=1e-9)
    np.testing.assert_allclose(time_rs, time_py, rtol=1e-9)
    np.testing.assert_allclose(
        np.nan_to_num(tnr_rs, nan=-999.0),
        np.nan_to_num(tnr_py, nan=-999.0),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(promi_rs, promi_py)


@pytest.mark.differential
def test_pr_ecma_st_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 24000
    stimulus = _two_tone_stimulus(fs, 0.5, seed=1)

    t_rs, pr_rs, prom_rs, tf_rs = mosqito_rs.pr_ecma_st(stimulus, fs, prominence=False)
    t_py, pr_py, prom_py, tf_py = mosqito.sq_metrics.pr_ecma_st(stimulus, fs, prominence=False)

    np.testing.assert_allclose(t_rs, t_py, rtol=1e-6)
    np.testing.assert_allclose(pr_rs, np.asarray(pr_py, dtype=float), rtol=1e-6)
    np.testing.assert_array_equal(prom_rs, np.asarray(prom_py, dtype=bool))
    np.testing.assert_allclose(tf_rs, np.asarray(tf_py, dtype=float), rtol=1e-6)


@pytest.mark.differential
def test_pr_ecma_freq_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 24000
    stimulus = _two_tone_stimulus(fs, 0.5, seed=2)
    spec, freqs = mosqito.sound_level_meter.comp_spectrum(stimulus, fs, db=False)

    t_rs, pr_rs, prom_rs, tf_rs = mosqito_rs.pr_ecma_freq(np.abs(spec), freqs, prominence=False)
    t_py, pr_py, prom_py, tf_py = mosqito.sq_metrics.pr_ecma_freq(
        np.abs(spec), freqs, prominence=False
    )

    np.testing.assert_allclose(t_rs, t_py, rtol=1e-6)
    np.testing.assert_allclose(pr_rs, np.asarray(pr_py, dtype=float), rtol=1e-6)
    np.testing.assert_array_equal(prom_rs, np.asarray(prom_py, dtype=bool))
    np.testing.assert_allclose(tf_rs, np.asarray(tf_py, dtype=float), rtol=1e-6)


@pytest.mark.differential
def test_pr_ecma_perseg_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    fs = 24000
    stimulus = _two_tone_stimulus(fs, 1.2, seed=3)

    t_rs, pr_rs, promi_rs, freqs_rs, time_rs = mosqito_rs.pr_ecma_perseg(
        stimulus, fs, prominence=False, overlap=0.5
    )
    t_py, pr_py, promi_py, freqs_py, time_py = mosqito.sq_metrics.pr_ecma_perseg(
        stimulus, fs, prominence=False, overlap=0.5
    )

    np.testing.assert_allclose(t_rs, t_py, rtol=1e-6)
    np.testing.assert_allclose(freqs_rs, freqs_py, rtol=1e-9)
    np.testing.assert_allclose(time_rs, time_py, rtol=1e-9)
    np.testing.assert_allclose(
        np.nan_to_num(pr_rs, nan=-999.0),
        np.nan_to_num(pr_py, nan=-999.0),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(promi_rs, promi_py)
