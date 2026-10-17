# -*- coding: utf-8 -*-
"""Compare current outputs with the frozen goldens in ``tests/parity/data``."""
import functools
import json
import os

import numpy as np
import pytest

import generate_goldens as gen
from tolerances import assert_close

pytestmark = pytest.mark.parity


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


def test_every_metric_has_a_golden_and_vice_versa(current):
    assert _golden_files(), "no goldens found in " + gen.DATA_DIR
    assert _golden_files() == sorted(current)


@pytest.mark.parametrize("metric", _golden_files())
def test_matches_golden(metric, current):
    golden = np.load(os.path.join(gen.DATA_DIR, metric + ".npz"))
    now = current[metric]
    assert sorted(golden.files) == sorted(now), "set of outputs changed for " + metric
    assert golden.files, "empty golden for " + metric
    for key in golden.files:
        assert golden[key].size and np.isfinite(golden[key]).all(), "%s / %s: golden is empty or non-finite" % (metric, key)
        assert_close(now[key], golden[key], metric, key, note=_versions_note())
    # an all-zero (or constant) metric would make the comparison vacuous
    assert np.ptp(np.concatenate([golden[k].ravel() for k in golden.files])) > 0, "every golden of %s is constant" % metric
