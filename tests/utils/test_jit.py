# -*- coding: utf-8 -*-
"""Optional numba backend: lazy import, fallback, and exact agreement with plain Python."""
import subprocess
import sys

import numpy as np
import pytest

from mosqito.utils import _jit

needs_numba = pytest.mark.skipif(
    not _jit.HAS_NUMBA, reason="numba is not installed or disabled (MOSQITO_DISABLE_NUMBA=1)"
)


def _run(code, env_extra=None):
    import os

    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    env.pop("MOSQITO_DISABLE_NUMBA", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)


# ---------------------------------------------------------------- wrapper
def test_wrapper_keeps_plain_function_and_works_without_compiling():
    @_jit.jit()
    def add(a, b):
        return a + b

    assert add.py_func(1, 2) == 3
    assert add.__name__ == "add"
    assert not add.is_compiled
    assert add(2, 3) == 5


@needs_numba
def test_wrapper_compiles_on_first_call():
    @_jit.jit()
    def twice(x):
        return 2 * x

    assert not twice.is_compiled
    assert twice(4.0) == 8.0
    assert twice.is_compiled


# ---------------------------------------------------------------- import behaviour
def test_import_mosqito_does_not_import_numba():
    res = _run("import sys, mosqito; print('numba' in sys.modules)")
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "False"


def test_works_when_numba_is_missing():
    code = (
        "import sys; sys.modules['numba'] = None\n"
        "import numpy as np, mosqito\n"
        "from mosqito.utils import _jit\n"
        "from mosqito.sq_metrics import loudness_zwtv\n"
        "x = 0.02*np.random.default_rng(0).standard_normal(4800)\n"
        "N = loudness_zwtv(x, 48000)[0]\n"
        "print(_jit.HAS_NUMBA, len(N))"
    )
    res = _run(code)
    assert res.returncode == 0, res.stderr
    assert res.stdout.split()[0] == "False"


def test_disable_environment_variable():
    code = (
        "import numpy as np, mosqito\n"
        "from mosqito.utils import _jit\n"
        "from mosqito.sq_metrics import loudness_zwtv\n"
        "x = 0.02*np.random.default_rng(0).standard_normal(4800)\n"
        "loudness_zwtv(x, 48000)\n"
        "import sys; print(_jit.HAS_NUMBA, 'numba' in sys.modules)"
    )
    res = _run(code, {"MOSQITO_DISABLE_NUMBA": "1"})
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["False", "False"]


@needs_numba
def test_numba_is_imported_on_first_use_only():
    code = (
        "import sys, numpy as np, mosqito\n"
        "from mosqito.sq_metrics import loudness_zwtv\n"
        "before = 'numba' in sys.modules\n"
        "loudness_zwtv(0.02*np.random.default_rng(0).standard_normal(4800), 48000)\n"
        "print(before, 'numba' in sys.modules)"
    )
    res = _run(code)
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == ["False", "True"]


# ---------------------------------------------------------------- kernels
def _core_loudness():
    rng = np.random.default_rng(4)
    core = np.abs(rng.standard_normal((21, 60))) * rng.uniform(0, 6, (21, 1))
    core[:, ::4] *= 0.01
    return core


@needs_numba
@pytest.mark.loudness_zwtv
def test_decay_kernel_compiled_equals_plain():
    from mosqito.sq_metrics.loudness.loudness_zwtv import _nonlinear_decay as nd

    core = _core_loudness()
    nl_iter = 24
    ui = np.ascontiguousarray(
        (core[:, :, None] + np.arange(nl_iter) * 0.0).reshape(core.shape[0], -1)
    )
    B = nd._nl_coefficients(2000, nl_iter)
    compiled = np.empty(core.shape)
    plain = np.empty(core.shape)
    nd._nl_kernel(ui, B, compiled, nl_iter)
    nd._nl_kernel.py_func(ui, B, plain, nl_iter)
    assert nd._nl_kernel.is_compiled
    np.testing.assert_array_equal(compiled, plain)


@needs_numba
@pytest.mark.loudness_zwst
def test_slopes_kernel_compiled_equals_plain_and_vectorised():
    from mosqito.sq_metrics.loudness.loudness_zwst import _calc_slopes as cs
    from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness

    spec = np.random.default_rng(5).uniform(10, 100, (28, 40))
    spec[:11] = np.minimum(spec[:11], 119)
    nm = _main_loudness(spec, "free")
    args = (cs._ZUP, cs._ZUP_EA, cs._RNS, cs._RNS_R, cs._USL)
    frames = np.ascontiguousarray(nm.T)
    results = {}
    for name, fn in (("compiled", cs._calc_slopes_frames), ("plain", cs._calc_slopes_frames.py_func)):
        N, S = np.empty(frames.shape[0]), np.empty((frames.shape[0], 240))
        fn(frames, N, S, *args)
        results[name] = (N, S)
    np.testing.assert_array_equal(results["compiled"][0], results["plain"][0])
    np.testing.assert_array_equal(results["compiled"][1], results["plain"][1])
    N_vec, S_vec = cs._calc_slopes_vectorised(nm.copy())
    np.testing.assert_array_equal(results["compiled"][0], N_vec)
    np.testing.assert_allclose(results["compiled"][1].T, S_vec, rtol=0, atol=1e-12)
