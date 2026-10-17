# -*- coding: utf-8 -*-
"""Compute and store the reference outputs used by the parity tests (``compute_all`` only computes).

Usage, from the repository root::

    python tests/parity/generate_goldens.py            # rewrite tests/parity/data
    python tests/parity/generate_goldens.py --out DIR  # write elsewhere (strict_compare)

The goldens are a development aid, not a specification: they show that the
outputs of a change stay where they were. They were made with the NumPy/SciPy
versions in ``manifest.json``; commit ``data/*.npz`` and ``manifest.json``
together when regenerating.
"""
import argparse
import json
import os

import numpy as np
import scipy

import mosqito
import signals

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def _flat(prefix, values):
    """Map a metric's dict of outputs to ``{prefix__name: ndarray}``."""
    return {"%s__%s" % (prefix, k): np.asarray(v, dtype=float) for k, v in values.items()}


def _metrics():
    """``{metric: (function, names of the outputs it returns, signals to run it on)}``."""
    from mosqito.sq_metrics import (
        loudness_ecma, loudness_zwst, loudness_zwst_perseg, loudness_zwtv,
        roughness_dw, roughness_ecma,
    )
    from mosqito.sound_level_meter import noct_spectrum

    iso, ecma, rough = signals.ISO_SIGNALS, signals.ECMA_SIGNALS, signals.ROUGHNESS_SIGNALS
    return {
        "loudness_zwst": (loudness_zwst, ("N", "N_spec", "bark"), iso),
        "loudness_zwst_perseg": (loudness_zwst_perseg, ("N", "N_spec", "bark", "time"), iso),
        "loudness_zwtv": (loudness_zwtv, ("N", "N_spec", "bark", "time"), iso),
        "loudness_ecma": (loudness_ecma, ("N", "N_time", "N_spec", "bark", "time"), ecma),
        "roughness_ecma": (roughness_ecma, ("R", "R_time", "R_spec", "bark", "time"), rough),
        "roughness_dw": (roughness_dw, ("R", "R_spec", "bark", "time"), rough),
        # 24 and 12500 are fmin and fmax in Hz
        "noct_spectrum": (lambda x, fs: noct_spectrum(x, fs, 24, 12500), ("spec", "fpref"), iso),
    }


def _kernels():
    from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import _calc_slopes
    from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness
    from mosqito.sq_metrics.loudness.loudness_zwtv._nonlinear_decay import _nl_loudness

    inp = signals.kernel_inputs()
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
    result = {}
    test_signals = signals.all_signals()
    for name, (fn, outputs, signal_names) in _metrics().items():
        entry = {}
        for sname in signal_names:
            x, fs = test_signals[sname]
            entry.update(_flat(sname, dict(zip(outputs, fn(x.copy(), fs), strict=True))))
        result[name] = entry
    for name, values in _kernels().items():
        result["kernel_" + name] = {k: np.asarray(v, dtype=float) for k, v in values.items()}
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
    print("mosqito from", os.path.dirname(mosqito.__file__))
    write(args.out, compute_all())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
