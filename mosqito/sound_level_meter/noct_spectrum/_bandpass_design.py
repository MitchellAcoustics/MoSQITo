# -*- coding: utf-8 -*-
"""Cached design of the Butterworth band-pass filters of the n-octave analysis."""
from functools import lru_cache

from scipy.signal import butter


@lru_cache(maxsize=256)
def _design_cached(order, w1, w2):
    return butter(int(order), (w1, w2), "bandpass", analog=False, output="sos")


def _design_sos(order, w1, w2):
    """Second-order sections of a band-pass Butterworth filter.

    Designing the same bands again on every call (38 filters for a third-octave
    analysis) took more time than filtering a short signal, so designs are
    cached. Each call returns its own copy (a few numbers), because SciPy's
    filtering routines do not accept read-only arrays and a caller must not be
    able to alter the cached design.
    """
    return _design_cached(order, w1, w2).copy()
