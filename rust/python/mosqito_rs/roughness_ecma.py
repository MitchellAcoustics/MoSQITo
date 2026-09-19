"""ECMA-418-2:2022 (2nd Ed) roughness (the Sottek Hearing Model's roughness
stage, for stationary signals).

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.sq_metrics.roughness.roughness_ecma``'s public signature.

Unlike ``mosqito.sq_metrics.roughness_ecma``, this implementation applies
the corrections described in Wanty, Glesser & Casagrande Hirono, *"ECMA-418-2
roughness, a challenging implementation"* (INTERNOISE 2024), plus a fix to
an unsanctioned bug in the lowpass-filtering stage (with its calibration
factor ``c_R`` re-derived accordingly). See ``DEVIATIONS.md`` for the full
register of divergences (D-ecma-1 through D-ecma-7 and D-ecma-5's `c_R`
re-fit).
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["roughness_ecma"]


def roughness_ecma(
    signal: np.ndarray, fs: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the specific and total roughness according to ECMA-418-2
    (2nd Ed, 2022), Section 7.1.

    Matches :func:`mosqito.sq_metrics.roughness_ecma`.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal time values [Pa]. Resampled to 48 kHz first if not already.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    R : float
        Overall roughness representative value [asper_HMS].
    R_time : numpy.ndarray
        Roughness over time [asper_HMS], shape ``(Ntime,)``.
    R_specific : numpy.ndarray
        Specific roughness [asper_HMS/bark], shape ``(53,)`` — the time
        average over ``R_time_spec[10:, :]``, per §7.1.8.
    bark_axis : numpy.ndarray
        Bark axis, shape ``(53,)``.
    time_axis : numpy.ndarray
        Time axis [s] at 50 Hz, shape ``(Ntime,)``.
    """
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    return _core.roughness_ecma(signal, float(fs))
