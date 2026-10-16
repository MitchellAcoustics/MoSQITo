# -*- coding: utf-8 -*-
"""Compute (and optionally store) the reference outputs used by the parity tests.

Usage, from the repository root::

    python tests/parity/generate_goldens.py            # rewrite tests/parity/data
    python tests/parity/generate_goldens.py --out DIR  # write elsewhere (strict_compare)

Regenerate the committed goldens only in a dedicated, explained change: the
manifest records the NumPy/SciPy versions they were produced with.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import mosqito  # noqa: F401
except ImportError:  # running from a checkout that is not installed
    sys.path.append(os.path.abspath(os.path.join(HERE, "..", "..")))

import signals as sig_factory  # noqa: E402

DATA_DIR = os.path.join(HERE, "data")


def _flat(prefix, values):
    """Map a metric's tuple of outputs to ``{prefix__name: ndarray}``."""
    return {"%s__%s" % (prefix, k): np.asarray(v, dtype=float) for k, v in values.items()}


def _metrics():
    """``{metric: (function, names of the outputs it returns)}``."""
    from mosqito.sq_metrics import (
        loudness_ecma, loudness_zwst, loudness_zwst_perseg, loudness_zwtv,
        roughness_dw, roughness_ecma, sharpness_din_st, sharpness_din_tv,
    )
    from mosqito.sound_level_meter import noct_spectrum

    return {
        "loudness_zwst": (loudness_zwst, ("N", "N_spec", "bark")),
        "loudness_zwst_perseg": (loudness_zwst_perseg, ("N", "N_spec", "bark", "time")),
        "loudness_zwtv": (loudness_zwtv, ("N", "N_spec", "bark", "time")),
        "sharpness_din_st": (lambda x, fs: (sharpness_din_st(x, fs),), ("S",)),
        "sharpness_din_tv": (sharpness_din_tv, ("S", "time")),
        "loudness_ecma": (loudness_ecma, ("N", "N_time", "N_spec", "bark", "time")),
        "roughness_ecma": (roughness_ecma, ("R", "R_time", "R_spec", "bark", "time")),
        "roughness_dw": (roughness_dw, ("R", "R_spec", "bark", "time")),
        "noct_spectrum": (lambda x, fs: noct_spectrum(x, fs, 24, 12500), ("spec", "fpref")),
    }


def _kernels():
    from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import _calc_slopes
    from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness
    from mosqito.sq_metrics.loudness.loudness_zwtv._nonlinear_decay import _nl_loudness

    inp = sig_factory.kernel_inputs()
    out = {}
    out["nl_loudness"] = {"in": inp["core_loudness"], "out": _nl_loudness(inp["core_loudness"].copy())}
    nm = _main_loudness(inp["third_octave_levels"].copy(), "free")
    N, spec = _calc_slopes(nm.copy())
    out["calc_slopes"] = {"nm": nm, "N": N, "N_spec": spec}
    N1, spec1 = _calc_slopes(nm[:, 0].copy())  # 1-D input path
    out["calc_slopes_1d"] = {"nm": nm[:, 0], "N": N1, "N_spec": spec1}
    return out


def compute_all():
    """Return ``{metric: {key: ndarray}}`` for every metric and kernel."""
    warnings.simplefilter("ignore")
    result = {}
    test_signals = sig_factory.signals()
    for name, (fn, outputs) in _metrics().items():
        entry = {}
        for sname, (x, fs) in test_signals.items():
            entry.update(_flat(sname, dict(zip(outputs, fn(x.copy(), fs), strict=True))))
        result[name] = entry
    for name, values in _kernels().items():
        result["kernel_" + name] = _flat("seeded", values)
    return result


def write(out_dir, results):
    os.makedirs(out_dir, exist_ok=True)
    for name, entry in results.items():
        np.savez_compressed(os.path.join(out_dir, name + ".npz"), **entry)
    manifest = {"numpy": np.__version__, "scipy": scipy.__version__}
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=DATA_DIR)
    args = parser.parse_args()
    import mosqito as m

    print("mosqito from", os.path.dirname(m.__file__))
    write(args.out, compute_all())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
