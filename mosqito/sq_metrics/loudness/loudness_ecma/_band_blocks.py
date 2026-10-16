# -*- coding: utf-8 -*-
"""Per-band processing shared by loudness_ecma and roughness_ecma.

The reference implementation filtered all 53 bands, gathered every band into
one (53, blocks, block size) array, and only then processed it. Blocks overlap
(by 50 % for loudness, 75 % for roughness), so that array holds every sample
two to four times over and reaches gigabytes for a 20 s signal. Here each band
is filtered, cut into blocks and reduced on its own, which gives the same numbers
with a small fraction of the memory, and makes the bands independent of each
other so that they can be processed on several threads.
"""
import numbers

from numpy import arange, ceil, clip, int32, linspace, mean, sqrt
from scipy.signal import lfilter, sosfilt

from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import (
    _auditory_filters_centre_freq,
)
from mosqito.sq_metrics.loudness.loudness_ecma._ear_filter_design import _ear_filter_design
from mosqito.sq_metrics.loudness.loudness_ecma._gammatone import _gammatone
from mosqito.sq_metrics.loudness.loudness_ecma._loudness_ecma_data import ltq_z
from mosqito.sq_metrics.loudness.loudness_ecma._nonlinearity import _nonlinearity

FS = 48000  # ECMA-418-2 is defined at 48 kHz
N_BANDS = 53


def check_block_sizes(sb, sh):
    """Block and hop sizes must be integers (per-band lists were never supported)."""
    for name, value in (("sb", sb), ("sh", sh)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError("%s must be a positive integer, got %r" % (name, value))


def ecma_block_index(n_new, sb, sh):
    """Sample indices of every block, shape (n_blocks, sb) (ECMA-418-2 eq. 18-20).

    The index is ``linspace(start, start + sb, sb)`` truncated to integers, as in
    the standard's reference code: each block holds ``sb - 1`` consecutive samples
    followed by the one at ``start + sb``. The same indices apply to every band.
    """
    starts = arange(int(ceil((n_new + sh) / sh) - 1))
    return linspace(starts * sh, starts * sh + sb, sb).astype(int32).T


def ecma_time_axis(n_samples, index):
    """Mean time of every block, shape (n_blocks,)."""
    time = linspace(0, (n_samples - 1) / FS, n_samples)
    return mean(time[index], axis=1)


def ear_filtered(signal):
    """Outer and middle ear filtering (ECMA-418-2 5.1.2)."""
    return sosfilt(_ear_filter_design(), signal, axis=0)


def band_blocks(ear_signal, band, index):
    """Band-pass signal of one critical band, cut into blocks (5.1.3 to 5.1.5).

    Returns an array of shape (n_blocks, block size).
    """
    b, a = _gammatone(_auditory_filters_centre_freq()[band], k=5, fs=float(FS))
    band_pass = 2.0 * lfilter(b, a, ear_signal, axis=0).real
    return band_pass[index]


def band_specific_loudness(blocks, band):
    """Specific loudness over time of one band from its blocks (5.1.6 to 5.1.9)."""
    rms = sqrt(2 * mean(clip(blocks, 0.0, None) ** 2, axis=1))
    a_prime = _nonlinearity(rms)
    a_prime[a_prime < ltq_z[band]] = ltq_z[band]
    return a_prime - ltq_z[band]
