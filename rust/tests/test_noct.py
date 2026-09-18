"""Conformance and differential tests for `mosqito_rs.sound_level_meter`.

The conformance test reproduces MoSQITo's own
`test_noct_synthesis_technical` exactly (same reference wav, same tolerance):
`noct_spectrum` (time domain) and `noct_synthesis` (frequency domain) must
agree within 0.3 dB on real pink-noise audio. That self-consistency is what
MoSQITo's own test suite gates on; nothing here should be stricter than that
without also being stricter for MoSQITo itself.

The differential test additionally compares `mosqito_rs` directly against an
installed `mosqito` package. It is a diagnostic, not a gate — per the plan,
tier 3 reports divergences rather than failing on them, since where the two
disagree either implementation could be the one at fault. In practice, for
`noct_spectrum`/`noct_synthesis`, agreement is near machine precision (no
deviation is claimed for this metric), so the test does assert tight
agreement; a real divergence here would be a porting bug worth knowing about
immediately, not something to quietly log.
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs
from conftest import load_wav_calibrated


def _db(amplitude: np.ndarray) -> np.ndarray:
    return 20 * np.log10(amplitude / 2e-5)


@pytest.mark.conformance
def test_noct_spectrum_and_synthesis_agree_on_reference_pink_noise(repo_root):
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )

    for order in (1, 3):
        spec_t, freq_t = mosqito_rs.noct_spectrum(sig, fs, fmin=24, fmax=12600, n=order)

        n = len(sig)
        spectrum = 2 / np.sqrt(2) / n * np.fft.fft(sig)[0 : n // 2]
        freqs = np.fft.fftfreq(n, 1 / fs)[0 : n // 2]
        spec_f, freq_f = mosqito_rs.noct_synthesis(
            np.abs(spectrum), freqs, fmin=24, fmax=12600, n=order
        )

        np.testing.assert_allclose(freq_t, freq_f)
        np.testing.assert_allclose(_db(spec_t), _db(spec_f), atol=0.3)


@pytest.mark.differential
def test_noct_spectrum_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )

    for order in (1, 3):
        spec_rs, freq_rs = mosqito_rs.noct_spectrum(sig, fs, fmin=24, fmax=12600, n=order)
        spec_py, freq_py = mosqito.sound_level_meter.noct_spectrum(
            sig, fs, fmin=24, fmax=12600, n=order
        )
        np.testing.assert_allclose(freq_rs, freq_py)
        np.testing.assert_allclose(spec_rs, spec_py, rtol=1e-9)


@pytest.mark.differential
def test_noct_synthesis_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )
    n = len(sig)
    spectrum = 2 / np.sqrt(2) / n * np.fft.fft(sig)[0 : n // 2]
    freqs = np.fft.fftfreq(n, 1 / fs)[0 : n // 2]

    for order in (1, 3):
        spec_rs, freq_rs = mosqito_rs.noct_synthesis(
            np.abs(spectrum), freqs, fmin=24, fmax=12600, n=order
        )
        spec_py, freq_py = mosqito.sound_level_meter.noct_synthesis(
            np.abs(spectrum), freqs, fmin=24, fmax=12600, n=order
        )
        np.testing.assert_allclose(freq_rs, freq_py)
        np.testing.assert_allclose(spec_rs, spec_py, rtol=1e-9)


@pytest.mark.differential
def test_noct_spectrum_matches_mosqito_multi_segment(repo_root):
    """Exercises the 2-D (nperseg, nseg) path `loudness_zwst_perseg` relies on."""
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav", wav_calib=2 * 2**0.5
    )
    sig2d = np.column_stack([sig[:48000], sig[100000:148000]])

    spec_rs, freq_rs = mosqito_rs.noct_spectrum(sig2d, fs, fmin=24, fmax=12600, n=3)
    spec_py, freq_py = mosqito.sound_level_meter.noct_spectrum(sig2d, fs, fmin=24, fmax=12600, n=3)
    assert spec_rs.shape == spec_py.shape == (len(freq_rs), 2)
    np.testing.assert_allclose(freq_rs, freq_py)
    np.testing.assert_allclose(spec_rs, spec_py, rtol=1e-9)
