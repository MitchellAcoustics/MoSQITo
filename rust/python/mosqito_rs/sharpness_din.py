"""DIN 45692:2009 sharpness (and the Aures/von Bismarck/Fastl weightings).

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.sq_metrics.sharpness.sharpness_din``'s exact public signatures.
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._validation import require_1d

__all__ = [
    "sharpness_din_from_loudness",
    "sharpness_din_st",
    "sharpness_din_freq",
    "sharpness_din_perseg",
    "sharpness_din_tv",
]


def sharpness_din_from_loudness(
    N: float | np.ndarray, N_specific: np.ndarray, weighting: str = "din"
) -> float | np.ndarray:
    """Compute the sharpness value from loudness.

    Matches :func:`mosqito.sq_metrics.sharpness_din_from_loudness`.

    Parameters
    ----------
    N : float or array_like
        Overall loudness [sones], shape ``(Ntime,)``.
    N_specific : numpy.ndarray
        Specific loudness [sones/bark], shape ``(240,)`` or ``(240, Ntime)``.
    weighting : {'din', 'aures', 'bismarck', 'fastl'}, default 'din'
        Weighting function used for the sharpness computation.

    Returns
    -------
    S : float or numpy.ndarray
        Sharpness value [acum], shape ``(Ntime,)``.

    Notes
    -----
    MoSQITo's Python dispatches on whether the resulting sharpness array has
    exactly one element: only when it has more than one does it zero out any
    segment whose loudness is below 0.1 sone (a scalar `N`, or an `N` array
    of length 1, never gets that masking — see
    ``sharpness_din_from_loudness.py:110-146``). This wrapper reproduces that
    exact dispatch: it does not additionally support broadcasting a single
    ``N_specific`` column against multiple values of `N`, a shape combination
    no caller in MoSQITo actually uses.
    """
    n_arr = np.atleast_1d(np.asarray(N, dtype=np.float64))
    n_specific_arr = np.asarray(N_specific, dtype=np.float64)
    if n_specific_arr.ndim == 1:
        n_specific_arr = n_specific_arr[:, np.newaxis]
    if n_specific_arr.shape[0] != 240:
        raise ValueError(
            f"N_specific must have 240 bark bands, got shape {n_specific_arr.shape}"
        )

    if n_arr.size == 1:
        n_specific_col = np.ascontiguousarray(n_specific_arr[:, 0])
        return _core.sharpness_din_from_loudness_scalar(
            float(n_arr.reshape(-1)[0]), n_specific_col, str(weighting)
        )

    if n_specific_arr.shape[1] != n_arr.size:
        raise ValueError(
            "N_specific must have one column per element of N: "
            f"N has {n_arr.size}, N_specific has {n_specific_arr.shape[1]}"
        )
    s = _core.sharpness_din_from_loudness_segmented(
        np.ascontiguousarray(n_arr), np.ascontiguousarray(n_specific_arr), str(weighting)
    )
    return np.asarray(s)


def sharpness_din_st(
    signal: np.ndarray, fs: float, weighting: str = "din", field_type: str = "free"
) -> float:
    """Compute the sharpness value from a time signal.

    Matches :func:`mosqito.sq_metrics.sharpness_din_st` (DIN 45692:2009,
    stationary signals).

    Parameters
    ----------
    signal : numpy.ndarray
        Input time signal [Pa].
    fs : float
        Sampling frequency [Hz]. Resampled to 48 kHz first if below it.
    weighting : {'din', 'aures', 'bismarck', 'fastl'}, default 'din'
        Weighting function used for the sharpness computation.
    field_type : {'free', 'diffuse'}, default 'free'
        Type of sound field.

    Returns
    -------
    S : float
        Sharpness value [acum].
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    return _core.sharpness_din_st(signal, float(fs), str(weighting), str(field_type))


def sharpness_din_freq(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    weighting: str = "din",
    field_type: str = "free",
) -> float:
    """Compute the sharpness value from a fine-band spectrum.

    Matches :func:`mosqito.sq_metrics.sharpness_din_freq` for a 1-D spectrum;
    the 2-D (multi-segment) case is not ported — MoSQITo's own Python always
    raises for it too, after needlessly computing the loudness first
    (``sharpness_din_freq.py:164-165``), so no behaviour is missing.

    Parameters
    ----------
    spectrum : numpy.ndarray
        A RMS amplitude spectrum (not dB), shape ``(nperseg,)``.
    freqs : numpy.ndarray
        Frequency axis [Hz], shape ``(nperseg,)``.
    weighting : {'din', 'aures', 'bismarck', 'fastl'}, default 'din'
        Weighting function used for the sharpness computation.
    field_type : {'free', 'diffuse'}, default 'free'
        Type of sound field.

    Returns
    -------
    S : float
        Sharpness value [acum].
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    require_1d(spectrum, freqs, fn_name="sharpness_din_freq")
    return _core.sharpness_din_freq(
        np.ascontiguousarray(spectrum),
        np.ascontiguousarray(freqs),
        str(weighting),
        str(field_type),
    )


def sharpness_din_perseg(
    signal: np.ndarray,
    fs: float,
    weighting: str = "din",
    nperseg: int = 4096,
    noverlap: int | None = None,
    field_type: str = "free",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the sharpness value per time segment from a time signal.

    Matches :func:`mosqito.sq_metrics.sharpness_din_perseg`.

    Parameters
    ----------
    signal : numpy.ndarray
        Input time signal [Pa].
    fs : float
        Sampling frequency [Hz]. Resampled to 48 kHz first if below it.
    weighting : {'din', 'aures', 'bismarck', 'fastl'}, default 'din'
        Weighting function used for the sharpness computation.
    nperseg : int, default 4096
        Length of each segment.
    noverlap : int, optional
        Hop size between segments. Defaults to ``nperseg // 2``.
    field_type : {'free', 'diffuse'}, default 'free'
        Type of sound field.

    Returns
    -------
    S : numpy.ndarray
        Sharpness value [acum], shape ``(Ntime,)``.
    time_axis : numpy.ndarray
        Time axis [s], shape ``(Ntime,)``.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    return _core.sharpness_din_perseg(
        signal,
        float(fs),
        int(nperseg),
        None if noverlap is None else int(noverlap),
        str(weighting),
        str(field_type),
    )


def sharpness_din_tv(
    signal: np.ndarray,
    fs: float,
    weighting: str = "din",
    field_type: str = "free",
    skip: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the sharpness value along time from a time-varying signal.

    Matches :func:`mosqito.sq_metrics.sharpness_din_tv`.

    Parameters
    ----------
    signal : numpy.ndarray
        Input time signal [Pa].
    fs : float
        Sampling frequency [Hz]. Resampled to 48 kHz first if below it.
    weighting : {'din', 'aures', 'bismarck', 'fastl'}, default 'din'
        Weighting function used for the sharpness computation.
    field_type : {'free', 'diffuse'}, default 'free'
        Type of sound field.
    skip : float, default 0.0
        Number of seconds to cut at the beginning of the analysis, to skip
        the transient effect of `loudness_zwtv`'s nonlinear decay stage.

    Returns
    -------
    S : numpy.ndarray
        Sharpness value [acum], shape ``(Ntime,)``.
    time_axis : numpy.ndarray
        Time axis [s], shape ``(Ntime,)``.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    return _core.sharpness_din_tv(signal, float(fs), str(weighting), str(field_type), float(skip))
