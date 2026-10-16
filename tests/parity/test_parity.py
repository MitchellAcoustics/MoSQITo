# -*- coding: utf-8 -*-
"""Compare current outputs with the frozen goldens in ``tests/parity/data``."""
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


def _versions_note():
    import scipy

    with open(os.path.join(gen.DATA_DIR, "manifest.json")) as fh:
        m = json.load(fh)
    return "goldens made with numpy %s / scipy %s, running numpy %s / scipy %s" % (
        m["numpy"], m["scipy"], np.__version__, scipy.__version__)


def _golden_files():
    return sorted(f[:-4] for f in os.listdir(gen.DATA_DIR) if f.endswith(".npz"))


@pytest.mark.parametrize("metric", _golden_files())
def test_matches_golden(metric, current):
    golden = np.load(os.path.join(gen.DATA_DIR, metric + ".npz"))
    now = current[metric]
    assert sorted(golden.files) == sorted(now), "set of outputs changed for " + metric
    for key in golden.files:
        assert_close(now[key], golden[key], metric, key, note=_versions_note())
