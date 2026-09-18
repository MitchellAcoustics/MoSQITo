"""ISO 532-1:2017 stationary loudness (Zwicker method).

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.sq_metrics.loudness.loudness_zwst``'s exact public signatures.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["loudness_zwst", "loudness_zwst_freq", "loudness_zwst_perseg"]


def loudness_zwst(
    signal: np.ndarray, fs: float, field_type: str = "free"
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the loudness value from a time signal.

    Matches :func:`mosqito.sq_metrics.loudness_zwst` (ISO 532-1:2017,
    Zwicker method, stationary signals).

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
    N : float
        Overall loudness [sone].
    N_specific : numpy.ndarray
        Specific loudness [sone/Bark], shape ``(240,)``.
    bark_axis : numpy.ndarray
        Bark axis, shape ``(240,)``.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    return _core.loudness_zwst(signal, float(fs), str(field_type))


def loudness_zwst_freq(
    spectrum: np.ndarray, freqs: np.ndarray, field_type: str = "free"
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the loudness value from a fine-band spectrum.

    Matches :func:`mosqito.sq_metrics.loudness_zwst_freq` for a 1-D
    spectrum; the 2-D (multi-segment) case is not yet ported.

    Parameters
    ----------
    spectrum : numpy.ndarray
        RMS amplitude spectrum (not dB), shape ``(nperseg,)``.
    freqs : numpy.ndarray
        Frequency axis [Hz], shape ``(nperseg,)``.
    field_type : {'free', 'diffuse'}, default 'free'
        Type of sound field.

    Returns
    -------
    N : float
        Overall loudness [sone].
    N_specific : numpy.ndarray
        Specific loudness [sone/Bark], shape ``(240,)``.
    bark_axis : numpy.ndarray
        Bark axis, shape ``(240,)``.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    if spectrum.ndim != 1 or freqs.ndim != 1:
        raise NotImplementedError(
            "mosqito_rs.loudness_zwst_freq currently supports a 1-D spectrum only; "
            "the 2-D (per-segment) case has not been ported yet"
        )
    return _core.loudness_zwst_freq(
        np.ascontiguousarray(spectrum), np.ascontiguousarray(freqs), str(field_type)
    )


def loudness_zwst_perseg(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 4096,
    noverlap: int | None = None,
    field_type: str = "free",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the loudness value per time segment from a time signal.

    Matches :func:`mosqito.sq_metrics.loudness_zwst_perseg`.

    Parameters
    ----------
    signal : numpy.ndarray
        Input time signal [Pa].
    fs : float
        Sampling frequency [Hz]. Resampled to 48 kHz first if below it.
    nperseg : int, default 4096
        Length of each segment.
    noverlap : int, optional
        Hop size between segments. Defaults to ``nperseg // 2``.
    field_type : {'free', 'diffuse'}, default 'free'
        Type of sound field.

    Returns
    -------
    N : numpy.ndarray
        Overall loudness per segment [sone], shape ``(Ntime,)``.
    N_specific : numpy.ndarray
        Specific loudness [sone/Bark], shape ``(240, Ntime)``.
    bark_axis : numpy.ndarray
        Bark axis, shape ``(240,)``.
    time_axis : numpy.ndarray
        Time axis [s], shape ``(Ntime,)``.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    return _core.loudness_zwst_perseg(
        signal,
        float(fs),
        int(nperseg),
        None if noverlap is None else int(noverlap),
        str(field_type),
    )
