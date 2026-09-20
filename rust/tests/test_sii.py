"""Differential tests for `mosqito_rs.sii_ansi*`, through the Python API.

The ANSI S3.5-1997 worked example itself (SII=0.504 for a custom
[50,40,40,30,20,0] dBSPL speech spectrum) cannot be exercised through
`sii_ansi_freq`/`sii_ansi_level`: MoSQITo's public API only ever supplies one
of its four standard speech spectra (`speech_level`), never an arbitrary
speech spectrum — that worked example is only reachable at the lower level
`_main_sii`/`main_sii` sits at, which is why the actual ANSI conformance gate
lives at the Rust level (`crates/mosqito-core/tests/conformance_sii.rs`) and
this file only differentially checks the Python wrapper glue (unit
conversion, `comp_spectrum`/`freq_band_synthesis` wiring, argument dispatch)
against an installed `mosqito`.

`threshold` as an explicit array is not exercised here: real MoSQITo's
`_main_sii.py` compares it with `threshold == "zwicker"`, which raises
`ValueError: The truth value of an array...` for any array threshold — a
real bug in the installed package, not something to differentially compare
against. `mosqito_rs`'s own support for an explicit threshold array is
instead golden-vector tested directly against `_main_sii` in
`crates/mosqito-core/tests/golden_sii.rs`.
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs


@pytest.mark.differential
@pytest.mark.parametrize("method", ["critical", "equally_critical", "third_octave", "octave"])
@pytest.mark.parametrize("threshold", [None, "zwicker"])
def test_sii_ansi_level_matches_mosqito(method, threshold):
    mosqito = pytest.importorskip("mosqito")
    sii_rs, spec_rs, freq_rs = mosqito_rs.sii_ansi_level(
        65.0, method=method, speech_level="raised", threshold=threshold
    )
    sii_py, spec_py, freq_py = mosqito.sq_metrics.sii_ansi_level(
        65.0, method=method, speech_level="raised", threshold=threshold
    )
    np.testing.assert_allclose(sii_rs, sii_py, rtol=1e-9)
    np.testing.assert_allclose(spec_rs, spec_py, rtol=1e-9)
    np.testing.assert_allclose(freq_rs, freq_py, rtol=1e-9)


@pytest.mark.differential
def test_sii_ansi_freq_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    rng = np.random.default_rng(0)
    freqs = np.array(
        [160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000],
        dtype=float,
    )
    spectrum = 40.0 - 0.001 * freqs + rng.uniform(-3, 3, len(freqs))

    sii_rs, spec_rs, _ = mosqito_rs.sii_ansi_freq(
        spectrum, freqs, method="third_octave", speech_level="normal"
    )
    sii_py, spec_py, _ = mosqito.sq_metrics.sii_ansi_freq(
        spectrum, freqs, method="third_octave", speech_level="normal"
    )
    np.testing.assert_allclose(sii_rs, sii_py, rtol=1e-9)
    np.testing.assert_allclose(spec_rs, spec_py, rtol=1e-9)


@pytest.mark.differential
def test_sii_ansi_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    from conftest import load_wav_calibrated

    sig, fs = load_wav_calibrated(repo_root / "tests/input/broadband_570.wav", wav_calib=1)
    sii_rs, spec_rs, freq_rs = mosqito_rs.sii_ansi(sig, fs, method="octave", speech_level="loud")
    sii_py, spec_py, freq_py = mosqito.sq_metrics.sii_ansi(
        sig, fs, method="octave", speech_level="loud"
    )
    np.testing.assert_allclose(sii_rs, sii_py, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(spec_rs, spec_py, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(freq_rs, freq_py, rtol=1e-9)
