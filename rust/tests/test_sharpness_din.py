"""Conformance and differential tests for `mosqito_rs.sharpness_din*`.

The conformance test is the actual DIN 45692:2009 chapter 6 gate: the 41
broadband/narrowband reference signals, checked at the standard's own
tolerance (the wider of ±5% or ±0.05 acum, matching `_check_compliance` in
`validations/sq_metrics/sharpness_din/validation_sharpness_din.py`). The
broader sweep over all 41 signals is exercised at the Rust level
(`crates/mosqito-core/tests/conformance_sharpness_din.rs`); this file adds
the one worked example DIN 45692's own unit test uses, through the Python
API specifically, plus differential checks against an installed `mosqito`.
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs
from conftest import load_wav_calibrated


@pytest.mark.conformance
def test_sharpness_din_st_matches_din_45692_worked_example(repo_root):
    sig, fs = load_wav_calibrated(repo_root / "tests/input/broadband_570.wav", wav_calib=1)
    s = mosqito_rs.sharpness_din_st(sig, fs, weighting="din")
    want = 2.85
    band = max(0.05 * want, 0.05)
    assert abs(s - want) <= band


@pytest.mark.differential
@pytest.mark.parametrize("weighting", ["din", "aures", "bismarck", "fastl"])
def test_sharpness_din_st_matches_mosqito(repo_root, weighting):
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(repo_root / "tests/input/broadband_570.wav", wav_calib=1)
    s_rs = mosqito_rs.sharpness_din_st(sig, fs, weighting=weighting)
    s_py = mosqito.sq_metrics.sharpness_din_st(sig, fs, weighting=weighting)
    np.testing.assert_allclose(s_rs, s_py, rtol=1e-9)


@pytest.mark.differential
def test_sharpness_din_from_loudness_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(repo_root / "tests/input/broadband_570.wav", wav_calib=1)
    n, n_specific, _bark = mosqito_rs.loudness_zwst(sig, fs)
    s_rs = mosqito_rs.sharpness_din_from_loudness(n, n_specific, weighting="din")
    s_py = mosqito.sq_metrics.sharpness_din_from_loudness(n, n_specific, weighting="din")
    np.testing.assert_allclose(s_rs, s_py, rtol=1e-9)


@pytest.mark.differential
def test_sharpness_din_perseg_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )
    s_rs, time_rs = mosqito_rs.sharpness_din_perseg(sig, fs, nperseg=2**14, weighting="din")
    s_py, time_py = mosqito.sq_metrics.sharpness_din_perseg(sig, fs, nperseg=2**14, weighting="din")
    np.testing.assert_allclose(s_rs, s_py, rtol=1e-9)
    np.testing.assert_allclose(time_rs, time_py)


@pytest.mark.differential
def test_sharpness_din_freq_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )
    n_samples = len(sig)
    spectrum = 2 / np.sqrt(2) / n_samples * np.fft.fft(sig)[0 : n_samples // 2]
    freqs = np.fft.fftfreq(n_samples, 1 / fs)[0 : n_samples // 2]

    s_rs = mosqito_rs.sharpness_din_freq(np.abs(spectrum), freqs, weighting="din")
    s_py = mosqito.sq_metrics.sharpness_din_freq(np.abs(spectrum), freqs, weighting="din")
    np.testing.assert_allclose(s_rs, s_py, rtol=1e-9)


def test_sharpness_din_tv_is_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        mosqito_rs.sharpness_din_tv(np.zeros(48000), 48000)
