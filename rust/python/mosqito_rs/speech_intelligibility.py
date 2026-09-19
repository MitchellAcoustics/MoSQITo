"""ANSI S3.5-1997 Speech Intelligibility Index (SII).

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.sq_metrics.speech_intelligibility.sii_ansi``'s exact public
signatures.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["sii_ansi", "sii_ansi_freq", "sii_ansi_level"]


def _resolve_threshold(threshold) -> tuple[bool, np.ndarray | None]:
    """Splits MoSQITo's polymorphic `threshold` (`None`/`'zwicker'`/array)
    into the `(use_zwicker_threshold, custom_threshold)` pair the compiled
    bindings take, avoiding a PyO3-side union type for it.
    """
    if threshold is None:
        return False, None
    if isinstance(threshold, str):
        if threshold != "zwicker":
            raise ValueError(
                f"threshold must be None, 'zwicker', or an array, got {threshold!r}"
            )
        return True, None
    return False, np.ascontiguousarray(threshold, dtype=np.float64)


def sii_ansi(
    noise: np.ndarray,
    fs: float,
    method: str,
    speech_level: str,
    threshold=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the speech intelligibility index from a noise time signal.

    Matches :func:`mosqito.sq_metrics.sii_ansi` (ANSI S3.5-1997).

    Parameters
    ----------
    noise : numpy.ndarray
        Noise time signal [Pa].
    fs : float
        Sampling frequency [Hz].
    method : {'critical', 'equally_critical', 'third_octave', 'octave'}
        Type of frequency band to use for the calculation.
    speech_level : {'normal', 'raised', 'loud', 'shout'}
        Speech level to assess.
    threshold : array_like, 'zwicker', or None, default None
        Threshold of hearing [dB ref. 2e-5 Pa], one value per band, or
        ``'zwicker'`` to use the standard threshold. Defaults to zero on
        every band.

    Returns
    -------
    sii : float
        Overall SII value.
    specific_sii : numpy.ndarray
        Specific SII values along the frequency axis.
    freq_axis : numpy.ndarray
        Frequency axis corresponding to `method`.
    """
    noise = np.ascontiguousarray(noise, dtype=np.float64)
    use_zwicker, custom = _resolve_threshold(threshold)
    return _core.sii_ansi(noise, float(fs), str(method), str(speech_level), use_zwicker, custom)


def sii_ansi_freq(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    method: str,
    speech_level: str,
    threshold=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the speech intelligibility index from a noise spectrum in dB.

    Matches :func:`mosqito.sq_metrics.sii_ansi_freq` (ANSI S3.5-1997).

    Parameters
    ----------
    spectrum : numpy.ndarray
        Noise spectrum [dB ref. 2e-5 Pa].
    freqs : numpy.ndarray
        Frequency axis [Hz] of the spectrum.
    method : {'critical', 'equally_critical', 'third_octave', 'octave'}
        Type of frequency band to use for the calculation.
    speech_level : {'normal', 'raised', 'loud', 'shout'}
        Speech level to assess.
    threshold : array_like, 'zwicker', or None, default None
        Threshold of hearing [dB ref. 2e-5 Pa], one value per band, or
        ``'zwicker'`` to use the standard threshold. Defaults to zero on
        every band.

    Returns
    -------
    sii : float
        Overall SII value.
    specific_sii : numpy.ndarray
        Specific SII values along the frequency axis.
    freq_axis : numpy.ndarray
        Frequency axis corresponding to `method`.
    """
    spectrum = np.ascontiguousarray(spectrum, dtype=np.float64)
    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    use_zwicker, custom = _resolve_threshold(threshold)
    return _core.sii_ansi_freq(
        spectrum, freqs, str(method), str(speech_level), use_zwicker, custom
    )


def sii_ansi_level(
    noise_level: float,
    method: str,
    speech_level: str,
    threshold=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the speech intelligibility index from an overall noise level.

    Matches :func:`mosqito.sq_metrics.sii_ansi_level` (ANSI S3.5-1997). The
    overall level is spread uniformly across every band of `method`.

    Parameters
    ----------
    noise_level : float
        Overall noise level [dB ref. 2e-5 Pa].
    method : {'critical', 'equally_critical', 'third_octave', 'octave'}
        Type of frequency band to use for the calculation.
    speech_level : {'normal', 'raised', 'loud', 'shout'}
        Speech level to assess.
    threshold : array_like, 'zwicker', or None, default None
        Threshold of hearing [dB ref. 2e-5 Pa], one value per band, or
        ``'zwicker'`` to use the standard threshold. Defaults to zero on
        every band.

    Returns
    -------
    sii : float
        Overall SII value.
    specific_sii : numpy.ndarray
        Specific SII values along the frequency axis.
    freq_axis : numpy.ndarray
        Frequency axis corresponding to `method`.
    """
    use_zwicker, custom = _resolve_threshold(threshold)
    return _core.sii_ansi_level(
        float(noise_level), str(method), str(speech_level), use_zwicker, custom
    )
