# -*- coding: utf-8 -*-

from numpy import sum, log10, mean, linspace

# Project Imports
from mosqito.sq_metrics.loudness.loudness_ecma._band_blocks import (
    N_BANDS,
    band_blocks,
    band_specific_loudness,
    check_block_sizes,
    ear_filtered,
    ecma_block_index,
    ecma_time_axis,
)
from mosqito.sq_metrics.loudness.loudness_ecma._preprocessing import (
    _preprocessing,
)
from mosqito.utils._parallel import parallel_map, resolve_workers

# Data import
# Threshold in quiet
from mosqito.sq_metrics.loudness.loudness_ecma._loudness_ecma_data import ltq_z


def loudness_ecma(signal, fs, sb=2048, sh=1024, workers=None):
    """Calculation of the specific and total loudness according to ECMA-418-2
    (2nd Ed, 2022), Section 5.

    This function computes the acoustic loudness according to ECMA-418-2 section 5 method for
    stationary signals.

    Parameters
    ----------
    signal: numpy.array
        Signal time values [Pa]. The sampling frequency of the signal must be 48000 Hz.
    sb: int
        Block size.
    sh: int
        Hop size.
    workers: int, optional
        Number of threads used to process the 53 bands in parallel. Default is
        the ``MOSQITO_WORKERS`` environment variable if set, else 1 (no threads).
        More than the number of cores brings nothing.

    Returns
    -------
    N : float
        Overall loudness representative value [sone_HMS].
    N_time : numpy.ndarray
        Loudness over time [sone_HMS], size (Ntime,).
    N_specific : numpy.ndarray
        Specific loudness [sone_HMS/bark], size (Nbark, Ntime).
	    Each of the 53 elements of the list corresponds to the time-dependant specific loudness for a given bark band. 
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

    check_block_sizes(sb, sh)

    # Windowing and zero-padding (5.1.2); a copy, the caller's array is untouched
    signal, n_new = _preprocessing(signal, sb, sh)

    # Outer and middle ear filtering (5.1.2)
    ear_signal = ear_filtered(signal)

    # Block indices (5.1.5), the same for every band
    index = ecma_block_index(n_new, sb, sh)
    time = ecma_time_axis(len(signal), index)

    # Band-pass filtering and segmentation (5.1.3 to 5.1.5), rectification,
    # rms and non-linearity (5.1.6 to 5.1.9), one band at a time
    def one_band(band):
        return band_specific_loudness(band_blocks(ear_signal, band, index), band)

    N_spec = parallel_map(one_band, range(N_BANDS), workers)
    time_array = [time.copy() for _ in range(N_BANDS)]

    # Time dependent values (eq. 116)
    delta_z = 0.5
    N_time = sum(N_spec, axis=0) * delta_z

    # Single-value loudness (eq. 117)
    e = 1 / log10(2)
    N = (mean(N_time**e)) ** (1 / e)

    bark_axis = linspace(0.5, 26.5, num=53, endpoint=True)
    
    return N, N_time, N_spec, bark_axis, time_array
