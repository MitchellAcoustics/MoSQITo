"""Tonality: tone-to-noise ratio (TNR) and prominence ratio (PR), per
ECMA-74 Annex D with the T-TNR/T-PR totals from ECMA TR/108.

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.sq_metrics``'s ``tnr_ecma_*``/``pr_ecma_*`` public signatures.

``tnr_ecma_perseg``/``pr_ecma_perseg`` only implement the 1-D-signal branch
of their MoSQITo counterparts — see ``DEVIATIONS.md`` for the 2-D-signal
branch's real ``NameError`` in MoSQITo, which is not reproduced here.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = [
    "tnr_ecma_st",
    "tnr_ecma_freq",
    "tnr_ecma_perseg",
    "pr_ecma_st",
    "pr_ecma_freq",
    "pr_ecma_perseg",
]


def _apply_prominence(t, values, prom, tones_freqs, prominence: bool):
    t_arr = np.array([t])
    if not prominence:
        return t_arr, values, prom, tones_freqs
    prom = np.asarray(prom, dtype=bool)
    return t_arr, values[prom], prom[prom], tones_freqs[prom]


def tnr_ecma_st(
    signal: np.ndarray, fs: float, prominence: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the tone-to-noise ratio from a stationary time signal.

    Matches :func:`mosqito.sq_metrics.tnr_ecma_st` (ECMA 418-1 / ECMA-74
    Annex D).

    Parameters
    ----------
    signal : numpy.ndarray
        Signal time values [Pa].
    fs : float
        Sampling frequency [Hz].
    prominence : bool, default True
        If True, only prominent tones are returned.

    Returns
    -------
    t_tnr : numpy.ndarray
        Global TNR value.
    tnr : numpy.ndarray
        TNR value for each detected tone.
    promi : numpy.ndarray
        Prominence criterion for each detected tone.
    tones_freqs : numpy.ndarray
        Frequency of each detected tone.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    t, tnr, prom, tones_freqs = _core.tnr_ecma_st(signal, float(fs))
    return _apply_prominence(t, tnr, prom, tones_freqs, prominence)


def tnr_ecma_freq(
    spectrum: np.ndarray, freqs: np.ndarray, prominence: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the tone-to-noise ratio from a fine-band spectrum.

    Matches :func:`mosqito.sq_metrics.tnr_ecma_freq` for a 1-D spectrum.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Amplitude or complex frequency spectrum.
    freqs : numpy.ndarray
        Frequency axis.
    prominence : bool, default True
        If True, only prominent tones are returned.

    Returns
    -------
    t_tnr, tnr, promi, tones_freqs
        As in :func:`tnr_ecma_st`.
    """
    spectrum_amp = np.ascontiguousarray(np.abs(spectrum), dtype=np.float64)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    t, tnr, prom, tones_freqs = _core.tnr_ecma_freq(spectrum_amp, freqs)
    return _apply_prominence(t, tnr, prom, tones_freqs, prominence)


def tnr_ecma_perseg(
    signal: np.ndarray, fs: float, prominence: bool = False, overlap: float = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the tone-to-noise ratio per time segment.

    Matches :func:`mosqito.sq_metrics.tnr_ecma_perseg` for a 1-D signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal time values [Pa].
    fs : float
        Sampling frequency [Hz].
    prominence : bool, default False
        If True, only prominent tones are placed on the returned grid.
    overlap : float, default 0
        Overlapping coefficient for the 500 ms time windows.

    Returns
    -------
    t_tnr : numpy.ndarray
        Global TNR value per segment.
    tnr : numpy.ndarray
        TNR values on a (frequency, segment) grid, NaN where no tone was
        detected.
    promi : numpy.ndarray
        Prominence criterion on the same grid.
    freqs : numpy.ndarray
        Frequency axis of the grid [Hz].
    time : numpy.ndarray
        Time axis [s].
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    return _core.tnr_ecma_perseg(signal, float(fs), float(overlap), bool(prominence))


def pr_ecma_st(
    signal: np.ndarray, fs: float, prominence: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the prominence ratio from a stationary time signal.

    Matches :func:`mosqito.sq_metrics.pr_ecma_st` (ECMA 418-1 / ECMA-74
    Annex D).

    Returns
    -------
    t_pr, pr, promi, tones_freqs
        As in :func:`tnr_ecma_st`.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    t, pr, prom, tones_freqs = _core.pr_ecma_st(signal, float(fs))
    return _apply_prominence(t, pr, prom, tones_freqs, prominence)


def pr_ecma_freq(
    spectrum: np.ndarray, freqs: np.ndarray, prominence: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the prominence ratio from a fine-band spectrum.

    Matches :func:`mosqito.sq_metrics.pr_ecma_freq` for a 1-D spectrum.

    Returns
    -------
    t_pr, pr, promi, tones_freqs
        As in :func:`tnr_ecma_st`.
    """
    spectrum_amp = np.ascontiguousarray(np.abs(spectrum), dtype=np.float64)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    t, pr, prom, tones_freqs = _core.pr_ecma_freq(spectrum_amp, freqs)
    return _apply_prominence(t, pr, prom, tones_freqs, prominence)


def pr_ecma_perseg(
    signal: np.ndarray, fs: float, prominence: bool = True, overlap: float = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the prominence ratio per time segment.

    Matches :func:`mosqito.sq_metrics.pr_ecma_perseg` for a 1-D signal.

    Returns
    -------
    t_pr, pr, promi, freqs, time
        As in :func:`tnr_ecma_perseg`.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    return _core.pr_ecma_perseg(signal, float(fs), float(overlap), bool(prominence))
