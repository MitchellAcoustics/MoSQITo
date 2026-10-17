# -*- coding: utf-8 -*-

from functools import lru_cache

from numpy import ones, ceil, linspace, arange, mean, int32, ascontiguousarray
import numpy as np

def _ecma_time_segmentation(signal_block, sb, sh, n_new):
    """Function used for the segmentation of a time signal into
    smaller parts of audio (blocks) following Formulas 18 to 20 (section 5.1.5)
    of ECMA-418-2:2022.

    Parameters
    ----------
    signal_block: list
        List of Numpy arrays containing bandpassed signals per critical band

    sb: int or list of int
        Block size, or list of block sizes per band

    sh: int or list of int
        Hop size, or list of hop sizes per band

    n_new : int
        Number of samples in signal after zero padding (Eq. 3)

    Returns
    -------
    block_array: list
        List of 53 two-dimensional arrays of size (nperseg, nseg)
        containing the segmented signal per critical band.

    time: list
        List of  Numpy arrays of size (nseg,) containing the time axis
        corresponding to each segmented signal. For each block, the time
        value chosen is the mean of the segmented time axis.
    """

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

    block_array = []
    time_array = []

    for z in range(53):
        block, time = _segment_band(signal_block[z], sb[z], sh[z], i_start[z], n_new)
        block_array.append(block)
        time_array.append(time)

    return block_array, time_array


@lru_cache(maxsize=1)
def _block_layout(n_samples, sb, sh, i_start, n_new):
    """Sample indices of the blocks (Formulas 18 to 20) and mean time of each
    block. Identical for the bands that share sb, sh and i_start, so computed
    once for them."""
    # Sampling frequency must be 48 kHz for ECMA-418-2 (2022)
    fs = 48000

    # Eq. (20) - number of blocks
    L_last = int(ceil((n_new + sh) / sh) - 1)

    # build time vector for signal
    time = linspace(0, (n_samples - 1) / fs, n_samples)
    # Eq. (18)
    L = arange(L_last)
    idx = linspace(L * sh + i_start, L * sh + i_start + sb, sb).astype(int32).T
    block_time = mean(time[idx], axis=1)
    idx.setflags(write=False)
    block_time.setflags(write=False)
    return idx, block_time


def _segment_band(signal, sb, sh, i_start, n_new):
    """Segmentation of the band-pass signal of one critical band into blocks
    (Formulas 18 to 20), for the block size sb, hop size sh and start index
    i_start of this band.

    Returns
    -------
    blocks: numpy.ndarray
        Array of size (nseg, nperseg).
    time: numpy.ndarray
        Array of size (nseg,): mean time of each block.
    """
    idx, time = _block_layout(signal.shape[0], sb, sh, i_start, n_new)
    # C-contiguous, as when the blocks of all bands are stacked into one array
    return ascontiguousarray(signal[idx]), time.copy()
