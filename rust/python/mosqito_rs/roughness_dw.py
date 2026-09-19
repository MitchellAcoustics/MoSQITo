"""Daniel & Weber (1997) roughness.

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.sq_metrics.roughness.roughness_dw``'s exact public signatures.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["roughness_dw", "roughness_dw_freq"]


def roughness_dw(
    signal: np.ndarray, fs: float, overlap: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the roughness according to the Daniel and Weber method from a
    time signal.

    Matches :func:`mosqito.sq_metrics.roughness_dw`.

    Parameters
    ----------
    signal : numpy.ndarray
        Input time signal [Pa].
    fs : float
        Sampling frequency [Hz].
    overlap : float, default 0.5
        Overlapping coefficient for the 200 ms time windows.

    Returns
    -------
    R : numpy.ndarray
        Roughness value [asper], shape ``(Ntime,)``.
    R_spec : numpy.ndarray
        Specific roughness over the Bark axis, shape ``(47, Ntime)``.
    bark_axis : numpy.ndarray
        Frequency axis [Bark], shape ``(47,)``.
    time : numpy.ndarray
        Time axis [s], shape ``(Ntime,)``.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    return _core.roughness_dw(signal, float(fs), float(overlap))


def roughness_dw_freq(
    spectrum: np.ndarray, freqs: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the roughness according to the Daniel and Weber method from a
    fine-band spectrum.

    Matches :func:`mosqito.sq_metrics.roughness_dw_freq` for a 1-D spectrum;
    the 2-D (multi-segment) case has not been ported.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Input amplitude spectrum, shape ``(nperseg,)``.
    freqs : numpy.ndarray
        Input frequency axis [Hz], shape ``(nperseg,)``.

    Returns
    -------
    R : float
        Roughness value [asper].
    R_spec : numpy.ndarray
        Specific roughness over the Bark axis, shape ``(47,)``.
    bark_axis : numpy.ndarray
        Frequency axis [Bark], shape ``(47,)``.

    Warning
    -------
    The input spectrum must be an amplitude spectrum (use ``abs()`` on a
    complex spectrum).
    """
    spectrum = np.ascontiguousarray(spectrum, dtype=np.float64)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    return _core.roughness_dw_freq(spectrum, freqs)
