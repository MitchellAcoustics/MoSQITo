"""Conformance and differential tests for `mosqito_rs.loudness_zwst`.

The conformance tests are the actual ISO 532-1:2017 §5.1 gate: reference
values from Annex B2/B3, checked at the standard's own tolerance (the wider
of ±5% or ±0.1 sone, matching `mosqito.utils.isoclose` and
`test_loudness_zwst.py`). The differential tests additionally compare
directly against an installed `mosqito` package — a diagnostic, not a gate.
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs
from conftest import load_wav_calibrated

# The ISO 532-1 Annex B2 third-octave-spectrum case (N = 83.296) is exercised
# at the Rust level, in crates/mosqito-core/tests/conformance_iso532_1.rs:
# mosqito_rs's Python API only exposes the time-signal and fine-spectrum
# entry points, not a from-a-third-octave-spectrum one to feed that reference
# directly.


def _read_reference_csv(path) -> np.ndarray:
    return np.genfromtxt(path, skip_header=1)


def _isoclose(actual, desired) -> bool:
    band = np.maximum(0.05 * np.abs(desired), 0.1)
    return bool(np.all(np.abs(np.asarray(actual) - desired) <= band))


@pytest.mark.conformance
def test_loudness_zwst_matches_iso_532_1_annex_b3_pink_noise(repo_root):
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )
    n, n_specific, bark = mosqito_rs.loudness_zwst(sig, fs)
    assert _isoclose(n, 10.498)
    want_spec = _read_reference_csv(repo_root / "tests/input/test_signal_5.csv")
    assert _isoclose(n_specific, want_spec)


@pytest.mark.conformance
def test_loudness_zwst_freq_matches_iso_532_1_annex_b3_pink_noise(repo_root):
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )
    n_samples = len(sig)
    spectrum = 2 / np.sqrt(2) / n_samples * np.fft.fft(sig)[0 : n_samples // 2]
    freqs = np.fft.fftfreq(n_samples, 1 / fs)[0 : n_samples // 2]

    n, n_specific, bark = mosqito_rs.loudness_zwst_freq(np.abs(spectrum), freqs)
    assert _isoclose(n, 10.498)
    want_spec = _read_reference_csv(repo_root / "tests/input/test_signal_5.csv")
    assert _isoclose(n_specific, want_spec)


@pytest.mark.conformance
def test_loudness_zwst_matches_iso_532_1_annex_b3_44100hz(repo_root):
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 3 (1 kHz 60 dB)_44100Hz.wav", wav_calib=2 * 2**0.5
    )
    assert fs == 44100
    n, _n_specific, _bark = mosqito_rs.loudness_zwst(sig, fs)
    assert _isoclose(n, 4.019)


@pytest.mark.differential
def test_loudness_zwst_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )
    n_rs, spec_rs, bark_rs = mosqito_rs.loudness_zwst(sig, fs)
    n_py, spec_py, bark_py = mosqito.sq_metrics.loudness_zwst(sig, fs)
    np.testing.assert_allclose(n_rs, n_py, rtol=1e-9)
    np.testing.assert_allclose(spec_rs, spec_py, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(bark_rs, bark_py)


@pytest.mark.differential
def test_loudness_zwst_freq_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )
    n_samples = len(sig)
    spectrum = 2 / np.sqrt(2) / n_samples * np.fft.fft(sig)[0 : n_samples // 2]
    freqs = np.fft.fftfreq(n_samples, 1 / fs)[0 : n_samples // 2]

    n_rs, spec_rs, bark_rs = mosqito_rs.loudness_zwst_freq(np.abs(spectrum), freqs)
    n_py, spec_py, bark_py = mosqito.sq_metrics.loudness_zwst_freq(np.abs(spectrum), freqs)
    np.testing.assert_allclose(n_rs, n_py, rtol=1e-9)
    np.testing.assert_allclose(spec_rs, spec_py, rtol=1e-9, atol=1e-9)


@pytest.mark.differential
def test_loudness_zwst_perseg_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )
    n_rs, spec_rs, bark_rs, time_rs = mosqito_rs.loudness_zwst_perseg(
        sig, fs, nperseg=16384, noverlap=4096
    )
    n_py, spec_py, bark_py, time_py = mosqito.sq_metrics.loudness_zwst_perseg(
        sig, fs, nperseg=16384, noverlap=4096
    )
    np.testing.assert_allclose(n_rs, n_py, rtol=1e-9)
    np.testing.assert_allclose(spec_rs, spec_py, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(time_rs, time_py)
