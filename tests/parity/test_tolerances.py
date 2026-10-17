# -*- coding: utf-8 -*-
"""Sanity checks of the parity tolerance policy: it must accept drift of the documented size and reject real changes."""
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
    g[:, 0] = 1.0  # a loud block (max|g| = 1, so the ATOL_SCALE floor of ~1e-8 is irrelevant)
    # drift of 4e-7 of a' on a near-threshold block: accepted
    assert_close(g + 4e-7 * ltq, g, "loudness_ecma", "x__N_spec")
    # 1e-4 of a' (a real change) is rejected, and the LTQ-scaled floor does not apply to other metrics
    with pytest.raises(AssertionError):
        assert_close(g + 1e-4 * ltq, g, "loudness_ecma", "x__N_spec")
    with pytest.raises(AssertionError):
        assert_close(g + 4e-7 * ltq, g, "roughness_ecma", "x__R_spec")


def test_ecma_n_time_floor():
    g = np.array([0.0, 0.0, 5.0])
    assert_close(g + 2e-7, g, "loudness_ecma", "x__N_time")
    with pytest.raises(AssertionError):
        assert_close(g + 1e-3, g, "loudness_ecma", "x__N_time")


def test_iso_quantum_is_coarser_above_16_sone():
    assert_close(np.array([20.009]), np.array([20.0]), "loudness_zwst", "x__N")
    with pytest.raises(AssertionError):
        assert_close(np.array([20.03]), np.array([20.0]), "loudness_zwst", "x__N")


def test_iso_quantum_applies_to_kernel_and_perseg_totals_only():
    assert_close(np.array([5.001]), np.array([5.0]), "kernel_calc_slopes", "seeded__N")
    assert_close(np.array([5.001]), np.array([5.0]), "loudness_zwst_perseg", "x__N")
    for metric in ("loudness_zwtv", "sharpness_din_st"):  # derived from a rounded total: no quantum
        with pytest.raises(AssertionError):
            assert_close(np.array([5.001]), np.array([5.0]), metric, "x__N")


def test_shape_mismatch_and_nan_handling():
    with pytest.raises(AssertionError, match="shape"):
        assert_close(np.zeros((3, 1)), np.zeros((1, 3)), "roughness_dw", "x__R")
    assert_close(np.array([np.nan, 1.0]), np.array([np.nan, 1.0]), "roughness_dw", "x__R")
    with pytest.raises(AssertionError):
        assert_close(np.array([0.0, 1.0]), np.array([np.nan, 1.0]), "roughness_dw", "x__R")


def test_all_zero_golden_has_no_slack():
    assert_close(np.zeros(4), np.zeros(4), "roughness_ecma", "x__R")
    with pytest.raises(AssertionError):
        assert_close(np.full(4, 1e-12), np.zeros(4), "roughness_ecma", "x__R")
