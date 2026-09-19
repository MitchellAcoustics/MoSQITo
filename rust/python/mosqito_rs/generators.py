"""Test-signal generators.

Wraps the compiled bindings in :mod:`mosqito_rs._core` to match
``mosqito.utils.{sine_wave,am_sine,am_noise,fm_sine}_generator``'s exact
public signatures.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = [
    "sine_wave_generator",
    "am_sine_generator",
    "am_noise_generator",
    "fm_sine_generator",
]


def sine_wave_generator(
    fs: float, d: float, freq: float, spl_level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a sine wave signal.

    Matches :func:`mosqito.utils.sine_wave_generator`.

    Parameters
    ----------
    fs : int
        Sampling frequency [Hz].
    d : int
        Signal duration [s].
    freq : int
        Sine wave frequency [Hz].
    spl_level : int
        Sound pressure level [dB SPL, ref 2e-5 Pa].

    Returns
    -------
    signal : numpy.ndarray
        Signal time values [Pa].
    time : numpy.ndarray
        Time axis [s].
    """
    return _core.sine_wave_generator(float(fs), float(d), float(freq), float(spl_level))


def am_sine_generator(
    xmod: np.ndarray, fs: float, fc: float, spl_level: float, print_m: bool = False
) -> tuple[np.ndarray, float]:
    """Generate an amplitude-modulated sine wave with a sinusoidal carrier.

    Matches :func:`mosqito.utils.am_sine_generator`.

    Parameters
    ----------
    xmod : numpy.ndarray
        Modulating signal.
    fs : float
        Sampling frequency [Hz].
    fc : float
        Carrier frequency [Hz]. Must be less than ``fs / 2``.
    spl_level : float
        Sound pressure level [dB SPL, ref 2e-5 Pa] of the modulated signal.
    print_m : bool, default False
        If True, print the modulation index.

    Returns
    -------
    y : numpy.ndarray
        Amplitude-modulated signal with sine carrier [Pa].
    m : float
        Modulation index.
    """
    xmod = np.ascontiguousarray(xmod, dtype=np.float64)
    y, m = _core.am_sine_generator(xmod, float(fs), float(fc), float(spl_level))
    if print_m:
        print(f"AM Modulation index = {m}")
    if m > 1:
        print(
            "Warning ['am_sine_generator']: modulation index m > 1\n"
            "\tSignal is overmodulated!"
        )
    return y, m


def am_noise_generator(
    xmod: np.ndarray, spl_level: float, print_m: bool = False, seed: int | None = None
) -> tuple[np.ndarray, float]:
    """Generate an amplitude-modulated broadband noise signal.

    Matches :func:`mosqito.utils.am_noise_generator`.

    Warning
    -------
    MoSQITo's Python draws its Gaussian noise carrier from
    ``numpy.random.default_rng()``, freshly seeded from OS entropy on every
    call — not reproducible run to run, even in the original package. This
    port instead draws from an explicitly seeded RNG (`seed`, randomised by
    default so the *default* behaviour still varies call to call like
    MoSQITo's does) that does not match NumPy's generator bit-for-bit; pass
    `seed` for a reproducible signal instead. See ``DEVIATIONS.md``.

    Parameters
    ----------
    xmod : numpy.ndarray
        Modulating signal.
    spl_level : float
        Sound pressure level [dB SPL, ref 2e-5 Pa] of the modulated signal.
    print_m : bool, default False
        If True, print the modulation index.
    seed : int, optional
        Seed for the noise carrier's random generator. Defaults to a fresh
        random seed on every call (matching MoSQITo's own non-determinism).

    Returns
    -------
    y_am : numpy.ndarray
        Amplitude-modulated noise signal [Pa].
    m : float
        Modulation index.
    """
    xmod = np.ascontiguousarray(xmod, dtype=np.float64)
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**63))
    y, m = _core.am_noise_generator(xmod, float(spl_level), int(seed))
    if print_m:
        print(f"AM Modulation index = {m}")
    if m > 1:
        print(
            "Warning ['am_noise_generator']: modulation index m > 1\n"
            "\tSignal is overmodulated!"
        )
    return y, m


def fm_sine_generator(
    xmod: np.ndarray, fs: float, fc: float, k: float, spl_level: float, print_info: bool = False
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Generate a frequency-modulated sine wave with a sinusoidal carrier.

    Matches :func:`mosqito.utils.fm_sine_generator`.

    Parameters
    ----------
    xmod : numpy.ndarray
        Modulating signal.
    fs : float
        Sampling frequency [Hz].
    fc : float
        Carrier frequency [Hz]. Must be less than ``fs / 2``.
    k : float
        Frequency sensitivity of the modulator.
    spl_level : float
        Sound pressure level [dB SPL, ref 2e-5 Pa] of the modulated signal.
    print_info : bool, default False
        If True, print the maximum frequency deviation and modulation index.

    Returns
    -------
    y_fm : numpy.ndarray
        Frequency-modulated signal with sine carrier [Pa].
    inst_freq : numpy.ndarray
        Instantaneous frequency.
    max_freq_deviation : float
        Maximum frequency deviation [Hz].
    FM_modulation_index : float
        Modulation index.
    """
    xmod = np.ascontiguousarray(xmod, dtype=np.float64)
    y, inst_freq, f_delta, m = _core.fm_sine_generator(
        xmod, float(fs), float(fc), float(k), float(spl_level)
    )
    if print_info:
        print(f"\tMax freq deviation: {f_delta} Hz")
        print(f"\tFM modulation index: {m:.2f}")
    return y, inst_freq, f_delta, m
