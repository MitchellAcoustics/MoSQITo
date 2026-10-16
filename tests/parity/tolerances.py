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

ECMA-418-2 specific loudness is ``a' - LTQ(z)`` with ``a'`` clipped at the hearing
threshold ``LTQ(z)``, i.e. a small difference of two larger numbers. Last-digit
changes in the band-pass filter output (``lfilter`` on a high-Q cascade, whose
rounding error scales with the loudest earlier part of the signal, not with the
current block) reach ``a'`` as a relative error of up to ~4e-7 across NumPy 2.2 /
SciPy 1.15 vs NumPy 2.4 / SciPy 1.17, and show up in the difference as an
*absolute* error of that fraction of ``a'``, however small the result. So for
``loudness_ecma`` specific loudness (and its band sum, ``N_time``) the absolute
floor is ``ECMA_A_RTOL * LTQ(z)``, i.e. a relative tolerance on ``a'`` of 1e-6
(under 3x the largest drift observed, still far below the effect of any real change
to the filter bank or the non-linearity), instead of a fraction of the output's own maximum.
"""
import numpy as np

from mosqito.sq_metrics.loudness.loudness_ecma._loudness_ecma_data import ltq_z

RTOL = 1e-7
ATOL_SCALE = 1e-9

# Keys (``<metric>`` / ``<signal>__<output>`` suffix) holding ISO-rounded totals.
ISO_ROUNDED = {
    ("loudness_zwst", "N"),
    ("loudness_zwst_perseg", "N"),
    ("kernel_calc_slopes", "N"),
    ("kernel_calc_slopes_1d", "N"),
}


ECMA_A_RTOL = 1e-6
LTQ = np.asarray(ltq_z, dtype=float)


def _ecma_floor(key, golden):
    """Absolute floor for ECMA-418-2 ``N_spec`` (bands x blocks) / ``N_time`` or None."""
    if key.endswith("N_spec") and golden.ndim == 2 and golden.shape[0] == LTQ.size:
        return ECMA_A_RTOL * LTQ[:, None]
    if key.endswith("N_time"):  # 0.5 * sum over bands of N_spec (eq. 116)
        return ECMA_A_RTOL * 0.5 * LTQ.sum()
    return None


def iso_quantum(values):
    """One rounding quantum of an ISO 532-1 total (0.001, or 0.01 above 16 sone)."""
    return np.where(np.abs(values) > 16.0, 1e-2, 1e-3) + 1e-12


def allclose_kwargs(metric, key, golden):
    """Return ``(rtol, atol)`` for comparing ``golden`` of ``metric``/``key``."""
    atol = ATOL_SCALE * float(np.max(np.abs(golden))) if golden.size else 0.0
    if metric == "loudness_ecma":
        floor = _ecma_floor(key, golden)
        if floor is not None:
            atol = np.maximum(atol, floor)
    if (metric, key.split("__", 1)[-1]) in ISO_ROUNDED:
        atol = np.maximum(atol, iso_quantum(golden))
    return RTOL, atol


def assert_close(actual, golden, metric, key, note=""):
    """Assert ``actual`` matches ``golden`` within this module's tolerances.

    Same criterion as ``numpy.testing.assert_allclose`` (``|a - g| <= atol + rtol * |g|``),
    written out because that function cannot take an array-valued ``atol``.
    """
    actual, golden = np.asarray(actual, dtype=float), np.asarray(golden, dtype=float)
    assert actual.shape == golden.shape, "%s / %s: shape %s, golden %s" % (metric, key, actual.shape, golden.shape)
    rtol, atol = allclose_kwargs(metric, key, golden)
    err = np.abs(actual - golden)
    bound = atol + rtol * np.abs(golden)
    bad = ~((err <= bound) | (np.isnan(actual) & np.isnan(golden)))
    assert not bad.any(), "%s / %s: %d of %d values differ from the golden, worst %.3e (allowed %.3e). %s" % (
        metric, key, bad.sum(), bad.size, err[bad].max(), np.broadcast_to(bound, err.shape)[bad][err[bad].argmax()], note)
