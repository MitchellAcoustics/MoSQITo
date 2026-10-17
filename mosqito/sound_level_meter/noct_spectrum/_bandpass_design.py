# -*- coding: utf-8 -*-
"""Cached design of the Butterworth band-pass filters of the n-octave analysis."""
from functools import lru_cache

from scipy.signal import butter


@lru_cache(maxsize=256)
def _design_cached(order, w1, w2):
    return butter(order, (w1, w2), "bandpass", analog=False, output="sos")


def _design_sos(order, w1, w2):
    """Second-order sections of a band-pass Butterworth filter.

    Designs are cached; each call returns a copy so callers cannot alter the cache.
    """
    return _design_cached(order, w1, w2).copy()
