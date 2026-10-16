# -*- coding: utf-8 -*-
"""Metrics must not write into the caller's signal array."""
import numpy as np
import pytest

from mosqito.sq_metrics import loudness_ecma, roughness_ecma


def _signal():
    t = np.arange(24000) / 48000
    return 0.05 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)


@pytest.mark.loudness_ecma
def test_loudness_ecma_leaves_input_unchanged():
    signal = _signal()
    before = signal.copy()
    loudness_ecma(signal, 48000)
    np.testing.assert_array_equal(signal, before)


@pytest.mark.roughness_ecma
def test_roughness_ecma_leaves_input_unchanged():
    signal = _signal()
    before = signal.copy()
    roughness_ecma(signal, 48000)
    np.testing.assert_array_equal(signal, before)
