# -*- coding: utf-8 -*-
"""Compare current outputs with the goldens in ``tests/parity/data``.

The goldens are a development aid: they flag a change in a metric's output, and
document that a refactor kept it. A failure means "look at this", and a NumPy or
SciPy upgrade that moves the last digits is not a regression. The tolerance is
therefore a single relative one plus an absolute one scaled to the output's
largest value. ``strict_compare.py`` makes the tight comparison between commits.
"""
import functools
import json
import os

import numpy as np
import pytest

import generate_goldens as gen

pytestmark = pytest.mark.parity

RTOL = 1e-6
ATOL_SCALE = 1e-6  # times the largest absolute value of the golden array


@pytest.fixture(scope="module")
def current():
    return gen.compute_all()


@functools.cache
def _versions_note():
    import scipy

    with open(os.path.join(gen.DATA_DIR, "manifest.json")) as fh:
        m = json.load(fh)
    return "goldens made with numpy %s / scipy %s, running numpy %s / scipy %s" % (
        m["numpy"], m["scipy"], np.__version__, scipy.__version__)


def _golden_files():
    return sorted(f[:-4] for f in os.listdir(gen.DATA_DIR) if f.endswith(".npz"))


def test_every_metric_has_a_golden(current):
    assert _golden_files() == sorted(current)


@pytest.mark.parametrize("metric", _golden_files())
def test_matches_golden(metric, current):
    golden = np.load(os.path.join(gen.DATA_DIR, metric + ".npz"))
    now = current[metric]
    assert sorted(golden.files) == sorted(now), "set of outputs changed for " + metric
    for key in golden.files:
        g = golden[key]
        atol = ATOL_SCALE * np.max(np.abs(g[np.isfinite(g)]), initial=0.0)
        np.testing.assert_allclose(now[key], g, rtol=RTOL, atol=atol,
                                   err_msg="%s / %s (%s)" % (metric, key, _versions_note()))
