"""ECMA-418-2:2022 (2nd Ed) loudness (the Sottek Hearing Model's loudness
stage, for stationary signals).

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.sq_metrics.loudness.loudness_ecma``'s public signature, for the
scalar ``sb``/``sh`` case (the per-band list case is not ported — see
``DEVIATIONS.md``).
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["loudness_ecma"]


def loudness_ecma(
    signal: np.ndarray, fs: float, sb: int = 2048, sh: int = 1024
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the specific and total loudness according to ECMA-418-2
    (2nd Ed, 2022), Section 5.

    Matches :func:`mosqito.sq_metrics.loudness_ecma` for scalar ``sb``/
    ``sh`` (a single block/hop size shared by all 53 bands — the per-band
    list case MoSQITo's Python also accepts is not supported here).

    Parameters
    ----------
    signal : numpy.ndarray
        Signal time values [Pa]. Resampled to 48 kHz first if not already.
    fs : float
        Sampling frequency [Hz].
    sb : int, default 2048
        Block size.
    sh : int, default 1024
        Hop size.

    Returns
    -------
    N : float
        Overall loudness representative value [sone_HMS].
    N_time : numpy.ndarray
        Loudness over time [sone_HMS], shape ``(Ntime,)``.
    N_specific : numpy.ndarray
        Specific loudness [sone_HMS/bark], shape ``(53, Ntime)``.
    bark_axis : numpy.ndarray
        Bark axis, shape ``(53,)``.
    time_axis : numpy.ndarray
        Time axis [s], shape ``(53, Ntime)`` — matching MoSQITo's Python,
        which returns one time axis per band (`time_axis[0]` is used
        directly in its own example). A scalar ``sb``/``sh`` gives every
        band the same block layout, so all 53 rows are identical; the Rust
        core computes the shared axis once and this wrapper broadcasts it
        to the documented per-band shape.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    n, n_time, n_specific, bark_axis, time_axis = _core.loudness_ecma(
        signal, float(fs), int(sb), int(sh)
    )
    time_axis = np.tile(time_axis, (53, 1))
    return n, n_time, n_specific, bark_axis, time_axis
