"""Small shared utilities: time segmentation.

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.utils.time_segmentation``'s exact public signature.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["time_segmentation"]


def time_segmentation(
    sig: np.ndarray,
    fs: float,
    nperseg: int = 2048,
    noverlap: int | None = None,
    is_ecma: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment a time signal into overlapping blocks.

    Matches :func:`mosqito.utils.time_segmentation` for its ``is_ecma=False``
    case — the only one any real MoSQITo caller uses (``roughness_dw``,
    ``tnr_ecma_perseg``, ``pr_ecma_perseg``). ``loudness_ecma``'s own
    internal segmentation (``is_ecma=True``) is implemented separately in
    this port's ``loudness_ecma`` module rather than through this shared
    function, so that path isn't reachable here either — see
    ``DEVIATIONS.md``.

    Parameters
    ----------
    sig : numpy.ndarray
        A 1-dimensional time signal array.
    fs : float
        The time signal sampling frequency.
    nperseg : int, default 2048
        Length of each segment.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg / 2``. Defaults to None.
    is_ecma : bool, default False
        Not supported — raises :class:`NotImplementedError` if True.

    Returns
    -------
    block_array : numpy.ndarray
        A 2-dimensional array of shape ``(nperseg, nseg)`` containing the
        segmented signal.
    time : numpy.ndarray
        The time axis corresponding to the segmented signal, shape ``(nseg,)``.
    """
    if is_ecma:
        raise NotImplementedError(
            "mosqito_rs.time_segmentation does not support is_ecma=True; "
            "loudness_ecma's own segmentation is implemented separately, "
            "see DEVIATIONS.md"
        )
    sig = np.ascontiguousarray(sig, dtype=np.float64)
    return _core.time_segmentation(sig, float(fs), int(nperseg), noverlap)
