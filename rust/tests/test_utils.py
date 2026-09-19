"""Differential tests for `mosqito_rs.time_segmentation`, through the
Python API.
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs


@pytest.mark.differential
def test_time_segmentation_matches_mosqito():
    mosqito = pytest.importorskip("mosqito")
    rng = np.random.default_rng(0)
    sig = rng.uniform(-1.0, 1.0, size=48000)

    blocks_rs, time_rs = mosqito_rs.time_segmentation(sig, 48000.0, nperseg=2048, noverlap=1024)
    blocks_py, time_py = mosqito.utils.time_segmentation(
        sig, 48000.0, nperseg=2048, noverlap=1024
    )
    np.testing.assert_allclose(blocks_rs, blocks_py, rtol=1e-9)
    np.testing.assert_allclose(time_rs, time_py, rtol=1e-9)


@pytest.mark.differential
def test_time_segmentation_matches_mosqito_default_noverlap():
    mosqito = pytest.importorskip("mosqito")
    rng = np.random.default_rng(1)
    sig = rng.uniform(-1.0, 1.0, size=20000)

    blocks_rs, time_rs = mosqito_rs.time_segmentation(sig, 48000.0, nperseg=2048)
    blocks_py, time_py = mosqito.utils.time_segmentation(sig, 48000.0, nperseg=2048)
    np.testing.assert_allclose(blocks_rs, blocks_py, rtol=1e-9)
    np.testing.assert_allclose(time_rs, time_py, rtol=1e-9)


def test_time_segmentation_rejects_is_ecma():
    sig = np.zeros(4096)
    with pytest.raises(NotImplementedError):
        mosqito_rs.time_segmentation(sig, 48000.0, nperseg=2048, is_ecma=True)


@pytest.mark.differential
def test_load_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    path = repo_root / "tests/input/Test signal 5 (pinknoise 60 dB).wav"

    sig_rs, fs_rs = mosqito_rs.load(str(path), wav_calib=2 * 2**0.5)
    sig_py, fs_py = mosqito.utils.load(str(path), wav_calib=2 * 2**0.5)
    assert fs_rs == fs_py
    np.testing.assert_allclose(sig_rs, sig_py, rtol=1e-9)


def test_load_rejects_non_wav():
    with pytest.raises(NotImplementedError):
        mosqito_rs.load("signal.mat", mat_signal="sig", mat_fs="fs")
