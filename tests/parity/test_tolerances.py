# -*- coding: utf-8 -*-
"""Sanity checks of the parity tolerance policy: it must accept version drift and reject real changes."""
import numpy as np
import pytest

from tolerances import assert_close

pytestmark = pytest.mark.parity


def test_iso_total_allows_one_quantum_only():
    assert_close(np.array([5.001]), np.array([5.000]), "loudness_zwst", "x__N")
    with pytest.raises(AssertionError):
        assert_close(np.array([5.003]), np.array([5.000]), "loudness_zwst", "x__N")


def test_generic_rtol():
    g = np.linspace(1.0, 2.0, 10)
    assert_close(g * (1 + 5e-8), g, "roughness_dw", "x__R")
    with pytest.raises(AssertionError):
        assert_close(g * (1 + 1e-5), g, "roughness_dw", "x__R")


def test_ecma_n_spec_floor_follows_threshold_not_output_max():
    from mosqito.sq_metrics.loudness.loudness_ecma._loudness_ecma_data import ltq_z

    ltq = np.asarray(ltq_z)[:, None]
    g = np.zeros((53, 4))
    g[:, 0] = 1.0  # a loud block, so max|g| is ~1
    # drift of 4e-7 of a' on a near-threshold block: accepted
    assert_close(g + 4e-7 * ltq, g, "loudness_ecma", "x__N_spec")
    # 1e-4 of a' (a real change) is rejected, as are changes in other metrics' absolute drift
    with pytest.raises(AssertionError):
        assert_close(g + 1e-4 * ltq, g, "loudness_ecma", "x__N_spec")
    with pytest.raises(AssertionError):
        assert_close(g + 4e-7 * ltq, g, "roughness_ecma", "x__R_spec")


def test_ecma_n_time_floor():
    g = np.array([0.0, 0.0, 5.0])
    assert_close(g + 2e-7, g, "loudness_ecma", "x__N_time")
    with pytest.raises(AssertionError):
        assert_close(g + 1e-3, g, "loudness_ecma", "x__N_time")
