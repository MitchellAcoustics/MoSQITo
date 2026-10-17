# -*- coding: utf-8 -*-

from numpy import mean, linspace, sum, log10, ones

# Project Imports
from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import (
    _auditory_filters_centre_freq,
)
from mosqito.sq_metrics.loudness.loudness_ecma._band_pass_signals import (
    _band_pass_signal,
    _ear_filtering,
)

from mosqito.sq_metrics.loudness.loudness_ecma._ecma_time_segmentation import (
    _segment_band,
)

from mosqito.sq_metrics.loudness.loudness_ecma._loudness_from_bandpass import (
    _band_specific_loudness,
)

from mosqito.sq_metrics.loudness.loudness_ecma._preprocessing import (
    _preprocessing,
)
from mosqito.utils._parallel import parallel_map, resolve_workers


def loudness_ecma(signal, fs, sb=2048, sh=1024, workers=None):
    """Calculation of the specific and total loudness according to ECMA-418-2
    (2nd Ed, 2022), Section 5.

    This function computes the acoustic loudness according to ECMA-418-2 section 5 method for
    stationary signals.

    Parameters
    ----------
    signal: numpy.array
        Signal time values [Pa]. The sampling frequency of the signal must be 48000 Hz.
    sb: int or list of int
        Block size.
    sh: int or list of int
        Hop size.
    workers: int, optional
        Number of threads (default 1).

    Returns
    -------
    N : float
        Overall loudness representative value [sone_HMS].
    N_time : numpy.ndarray
        Loudness over time [sone_HMS], size (Ntime,).
    N_specific : numpy.ndarray
        Specific loudness [sone_HMS/bark], size (Nbark, Ntime).
	    Each of the 53 elements of the list corresponds to the time-dependant specific loudness for a given bark band. Can be a ragged array if a different sb/sh are used for each band.
    bark_axis : numpy.ndarray
        Corresponding bark axis, size (Nbark,).
    time_array : numpy.ndarray
        Time axis, size (Nbark, Ntime).

    Warning
    -------
    The sampling frequency of the signal must be 48 kHz.

    See Also
    --------
    .loudness_zwst : Zwicker and Fastl loudness computation for a stationary time signal
    .loudness_zwtv : Zwicker and Fastl loudness computation for a non-stationary time signal

    References
    ----------
    :cite:empty:`L_ecma-ECMA-418-2`

    .. bibliography::
        :keyprefix: L_ecma-

    Examples
    --------
    .. plot::
       :include-source:

       >>> from mosqito.sq_metrics import loudness_ecma
       >>> import matplotlib.pyplot as plt
       >>> import numpy as np
       >>> f=1000
       >>> fs=48000
       >>> d=0.2
       >>> dB=60
       >>> time = np.arange(0, d, 1/fs)
       >>> stimulus = 0.5 * (1 + np.sin(2 * np.pi * f * time))
       >>> rms = np.sqrt(np.mean(np.power(stimulus, 2)))
       >>> ampl = 0.00002 * np.power(10, dB / 20) / rms
       >>> stimulus = stimulus * ampl
       >>> N, N_time, N_spec, bark_axis, time_array = loudness_ecma(stimulus, fs)
       >>> plt.plot(time_array[0], N_time)
       >>> plt.xlabel("Time [s]")
       >>> plt.ylabel("Loudness [Sone]")
    """
    workers = resolve_workers(workers)
    if fs != 48000:
        print(
            "[Warning] Signal resampled to 48 kHz fulfill the standard requirements and allow calculation."
        )
        from scipy.signal import resample

        signal = resample(signal, int(48000 * len(signal) / fs))
        fs = 48000

    # Windowing and zero-padding (5.1.2)
    signal, n_new = _preprocessing(signal, sb, sh)

    if isinstance(sb, int):
        sb = sb * ones(53, dtype=int)
    elif len(sb) != 53:
        raise ValueError("ERROR: len(sb) shall be either 1 or 53")
    if isinstance(sh, int):
        sh = sh * ones(53, dtype=int)
    elif len(sh) != 53:
        raise ValueError("ERROR: len(sh) shall be either 1 or 53")
    # Eq. (19)
    i_start = sb[0] - sb

    # Outer and middle ear filtering (5.1.2) and centre frequencies, once for all bands
    signal_filtered = _ear_filtering(signal)
    centre_freq = _auditory_filters_centre_freq()

    # Each band is filtered (5.1.3 to 5.1.4), segmented into blocks (5.1.5) and
    # reduced to its specific loudness (5.1.6 to 5.1.9) on its own, so that the
    # blocks of all 53 bands are never held in memory at once.
    def band_loudness(band_number):
        band_pass_signal = _band_pass_signal(signal_filtered, centre_freq[band_number])
        blocks, time = _segment_band(
            band_pass_signal,
            sb[band_number],
            sh[band_number],
            i_start[band_number],
            n_new,
        )
        return _band_specific_loudness(blocks, band_number), time

    results = parallel_map(band_loudness, range(53), workers)
    N_spec = [r[0] for r in results]
    time_array = [r[1] for r in results]

    # Time dependent values (eq. 116)
    delta_z = 0.5
    N_time = sum(N_spec, axis=0) * delta_z

    # Single-value loudness (eq. 117)
    e = 1 / log10(2)
    N = (mean(N_time**e)) ** (1 / e)

    bark_axis = linspace(0.5, 26.5, num=53, endpoint=True)
    
    return N, N_time, N_spec, bark_axis, time_array
