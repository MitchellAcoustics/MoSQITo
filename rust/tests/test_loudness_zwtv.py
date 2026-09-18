"""Conformance and differential tests for `mosqito_rs.loudness_zwtv`.

The real ISO 532-1 section 6.1 gate — all 20 Annex B.4/B.5 reference
signals against the standard's own published xlsx values, with the
standard's own ±2 ms realignment and ≤1% outlier allowance — is exercised
at the Rust level (`crates/mosqito-core/tests/conformance_loudness_zwtv.rs`),
since that's where the reference corpus and the wav-loading helpers already
live. This file adds a differential check against an installed `mosqito`,
through the Python API specifically.
"""

from __future__ import annotations

import numpy as np
import pytest

import mosqito_rs
from conftest import load_wav_calibrated


@pytest.mark.differential
def test_loudness_zwtv_matches_mosqito(repo_root):
    mosqito = pytest.importorskip("mosqito")
    sig, fs = load_wav_calibrated(
        repo_root / "tests/input/broadband_570.wav", wav_calib=1
    )
    n_rs, spec_rs, bark_rs, time_rs = mosqito_rs.loudness_zwtv(sig, fs)
    n_py, spec_py, bark_py, time_py = mosqito.sq_metrics.loudness_zwtv(sig, fs)
    np.testing.assert_allclose(n_rs, n_py, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(spec_rs, spec_py, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(bark_rs, bark_py)
    np.testing.assert_allclose(time_rs, time_py)
