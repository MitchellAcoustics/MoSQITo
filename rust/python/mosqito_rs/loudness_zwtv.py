"""ISO 532-1:2017 time-varying loudness (Zwicker method).

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.sq_metrics.loudness.loudness_zwtv``'s exact public signature.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["loudness_zwtv"]


def loudness_zwtv(
    signal: np.ndarray, fs: float, field_type: str = "free"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the loudness value from a time-varying time signal.

    Matches :func:`mosqito.sq_metrics.loudness_zwtv` (ISO 532-1:2017,
    Zwicker method, time-varying signals).

    Parameters
    ----------
    signal : numpy.ndarray
        Signal time values [Pa].
    fs : float
        Sampling frequency [Hz]. Resampled to 48 kHz first if below it.
    field_type : {'free', 'diffuse'}, default 'free'
        Type of sound field.

    Returns
    -------
    N : numpy.ndarray
        Overall loudness [sone], shape ``(Ntime,)``, at 2 ms temporal
        resolution.
    N_specific : numpy.ndarray
        Specific loudness [sone/Bark], shape ``(240, Ntime)``.
    bark_axis : numpy.ndarray
        Bark axis, shape ``(240,)``.
    time_axis : numpy.ndarray
        Time axis [s], shape ``(Ntime,)``.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    return _core.loudness_zwtv(signal, float(fs), str(field_type))
