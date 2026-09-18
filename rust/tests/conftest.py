"""Shared fixtures for mosqito_rs's pytest suite.

Conformance tests reuse MoSQITo's own reference corpus, which lives only in a
full git checkout (`MANIFEST.in` does not ship it in the wheel). This resolves
the repository root by walking up from this file, the same approach the Rust
integration tests use (`crates/mosqito-core/tests/golden_noct.rs`).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest


def _find_repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "tests" / "input").is_dir():
            return candidate
    raise RuntimeError(
        "could not find the MoSQITo repository root (looked for tests/input/) "
        f"starting from {here}"
    )


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return _find_repo_root()


def load_wav_calibrated(path: pathlib.Path, wav_calib: float) -> tuple[np.ndarray, int]:
    """Reads a mono 16-bit PCM wav with MoSQITo's exact calibration.

    Matches `mosqito/utils/load.py:43-89`'s wav branch: `wav_calib * signal /
    (2**15 - 1)` for int16 — note 32767, not 32768. Implemented directly with
    `scipy.io.wavfile` rather than importing `mosqito`, so the conformance
    tier does not depend on `mosqito` (or its transitive matplotlib/pyuff
    dependencies) being installed; only the differential tier does.
    """
    from scipy.io import wavfile

    fs, signal = wavfile.read(path)
    if signal.ndim > 1:
        signal = signal[:, 0]
    if signal.dtype == np.int16:
        signal = wav_calib * signal / (2**15 - 1)
    elif signal.dtype == np.int32:
        signal = wav_calib * signal / (2**31 - 1)
    else:
        signal = wav_calib * signal.astype(np.float64)
    return signal.astype(np.float64), fs
