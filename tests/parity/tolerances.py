# -*- coding: utf-8 -*-
"""Tolerances for the golden-file parity tests.

They are deliberately loose enough to survive ordinary NumPy/SciPy upgrades,
which change the last digits of FFT and filter results: ``rtol=1e-7`` with an
absolute floor of ``1e-9 * max|golden|`` (the ``numpy.testing.assert_allclose``
convention). Tight equivalence between a refactor and its parent commit is
checked separately by ``strict_compare.py``.

ISO 532-1 totals are rounded to 0.001 sone (0.01 sone above 16 sone), so a value
sitting on a rounding boundary may legitimately move by one quantum; those
totals get one quantum of absolute tolerance.
"""
import numpy as np

RTOL = 1e-7
ATOL_SCALE = 1e-9

# Keys (``<metric>`` / ``<signal>__<output>`` suffix) holding ISO-rounded totals.
ISO_ROUNDED = {
    ("loudness_zwst", "N"),
    ("loudness_zwst_perseg", "N"),
    ("kernel_calc_slopes", "N"),
    ("kernel_calc_slopes_1d", "N"),
}


def iso_quantum(values):
    """One rounding quantum of an ISO 532-1 total (0.001, or 0.01 above 16 sone)."""
    return np.where(np.abs(values) > 16.0, 1e-2, 1e-3) + 1e-12


def allclose_kwargs(metric, key, golden):
    """Return ``(rtol, atol)`` for comparing ``golden`` of ``metric``/``key``."""
    atol = ATOL_SCALE * float(np.max(np.abs(golden))) if golden.size else 0.0
    if (metric, key.split("__", 1)[-1]) in ISO_ROUNDED:
        atol = np.maximum(atol, iso_quantum(golden))
    return RTOL, atol
