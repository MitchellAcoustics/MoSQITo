# -*- coding: utf-8 -*-
"""Metrics must not write into the caller's signal array."""
import numpy as np
import pytest

from mosqito.sq_metrics import loudness_ecma, roughness_ecma


@pytest.mark.parametrize("metric", [
    pytest.param(loudness_ecma, marks=pytest.mark.loudness_ecma),
    pytest.param(roughness_ecma, marks=pytest.mark.roughness_ecma),
])
def test_ecma_metric_leaves_input_unchanged(metric):
    t = np.arange(24000) / 48000
    signal = 0.05 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)
    before = signal.copy()
    metric(signal, 48000)
    np.testing.assert_array_equal(signal, before)
