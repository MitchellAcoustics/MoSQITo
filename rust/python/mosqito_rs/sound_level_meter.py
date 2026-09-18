"""N-th octave band spectrum analysis (ANSI S1.1-1986).

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.sound_level_meter``'s exact public signatures — including accepting
either a 1-D or 2-D signal and ``numpy.squeeze``-ing the result the same way,
and the capitalised ``G`` keyword argument.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["noct_spectrum", "noct_synthesis"]


def noct_spectrum(
    sig: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    n: int = 3,
    G: int = 10,
    fr: float = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the n-th octave band spectrum of a time signal.

    Matches :func:`mosqito.sound_level_meter.noct_spectrum`: computes the RMS
    level of ``sig`` in each n-th octave band between ``fmin`` and ``fmax``.

    Parameters
    ----------
    sig : numpy.ndarray
        Time signal, shape ``(nperseg,)`` or ``(nperseg, nseg)``.
    fs : float
        Sampling frequency [Hz].
    fmin, fmax : float
        Band range [Hz].
    n : int, default 3
        Number of bands per octave.
    G : int, default 10
        Frequency ratio system: base 2 or base 10.
    fr : float, default 1000
        Reference frequency [Hz].

    Returns
    -------
    spec : numpy.ndarray
        Band levels, shape ``(nbands,)`` or ``(nbands, nseg)`` — squeezed
        exactly as MoSQITo's Python does, so a single-segment 2-D input still
        collapses to 1-D.
    fpref : numpy.ndarray
        Each band's nominal (preferred) center frequency, shape ``(nbands,)``.
    """
    sig = np.asarray(sig, dtype=np.float64)
    sig2d = sig[:, np.newaxis] if sig.ndim == 1 else sig

    spec, fpref = _core.noct_spectrum(
        np.ascontiguousarray(sig2d), float(fs), float(fmin), float(fmax), int(n), int(G), float(fr)
    )
    return np.squeeze(np.asarray(spec)), np.asarray(fpref)


def noct_synthesis(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    n: int = 3,
    G: int = 10,
    fr: float = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a frequency spectrum to n-th octave band levels.

    Matches :func:`mosqito.sound_level_meter.noct_synthesis` for a 1-D
    spectrum. ``freqs`` must span up to (approximately) 24 kHz, implying a
    48 kHz sampling rate — as in ISO 532-1, which is what this function feeds.

    Parameters
    ----------
    spectrum : numpy.ndarray
        RMS amplitude one-sided spectrum, shape ``(nperseg,)``.
    freqs : numpy.ndarray
        Frequency axis, shape ``(nperseg,)``.
    fmin, fmax : float
        Band range [Hz].
    n : int, default 3
        Number of bands per octave.
    G : int, default 10
        Frequency ratio system: base 2 or base 10.
    fr : float, default 1000
        Reference frequency [Hz].

    Returns
    -------
    spec : numpy.ndarray
        Band levels, shape ``(nbands,)``.
    fpref : numpy.ndarray
        Each band's nominal (preferred) center frequency, shape ``(nbands,)``.

    Notes
    -----
    Only a 1-D ``spectrum``/``freqs`` is currently supported. MoSQITo's Python
    also accepts a 2-D spectrum (one column per segment, with either a shared
    or a per-segment frequency axis); that case is not yet ported.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    if spectrum.ndim != 1 or freqs.ndim != 1:
        raise NotImplementedError(
            "mosqito_rs.noct_synthesis currently supports a 1-D spectrum only; "
            "the 2-D (per-segment) case has not been ported yet"
        )

    spec, fpref = _core.noct_synthesis(
        np.ascontiguousarray(spectrum),
        np.ascontiguousarray(freqs),
        float(fmin),
        float(fmax),
        int(n),
        int(G),
        float(fr),
    )
    return np.asarray(spec), np.asarray(fpref)
