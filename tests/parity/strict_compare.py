# -*- coding: utf-8 -*-
"""Strict equivalence check between a baseline commit and the working tree.

Run by the author of a refactoring change (not part of the test suite)::

    uv run python tests/parity/strict_compare.py --baseline <commit-or-branch>

The baseline is checked out in a temporary ``git worktree``; the same generator
is run once against it and once against this checkout, in the same environment,
and the outputs are compared at tight tolerances: exact for the nonlinear-decay
kernel, the slopes totals and ``noct_spectrum``, ``rtol=1e-10`` elsewhere. This
catches what the loose golden tolerances cannot, independent of package versions.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXACT = ("kernel_nl_loudness", "noct_spectrum")
EXACT_KEYS = {("kernel_calc_slopes", "N"), ("kernel_calc_slopes_1d", "N")}
STRICT_RTOL = 1e-10


def _generate(pythonpath_root, out_dir):
    env = dict(os.environ, PYTHONPATH=pythonpath_root)
    res = subprocess.run(
        [sys.executable, os.path.join(HERE, "generate_goldens.py"), "--out", out_dir],
        env=env, cwd=pythonpath_root, capture_output=True, text=True,
    )
    if res.returncode:
        sys.exit("generator failed under %s:\n%s" % (pythonpath_root, res.stderr[-2000:]))
    marker = "mosqito from"
    if marker not in res.stdout:
        sys.exit("generator under %s did not report where mosqito was imported from" % pythonpath_root)
    used = os.path.abspath(res.stdout.split(marker, 1)[1].split("\n", 1)[0].strip())
    if os.path.commonpath([used, os.path.abspath(pythonpath_root)]) != os.path.abspath(pythonpath_root):
        sys.exit("imported mosqito from %s, expected it under %s" % (used, pythonpath_root))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", required=True, help="commit-ish to compare against")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        worktree = os.path.join(tmp, "baseline")
        added = subprocess.run(["git", "worktree", "add", "--detach", worktree, args.baseline], cwd=REPO,
                               capture_output=True, text=True)
        if added.returncode:
            sys.exit("git worktree add failed:\n" + added.stderr)
        try:
            base_dir, new_dir = os.path.join(tmp, "base"), os.path.join(tmp, "new")
            _generate(worktree, base_dir)
            _generate(REPO, new_dir)
        finally:
            removed = subprocess.run(["git", "worktree", "remove", "--force", worktree], cwd=REPO,
                                     capture_output=True, text=True)
            if removed.returncode:
                print("warning: could not remove worktree %s:\n%s" % (worktree, removed.stderr), file=sys.stderr)
        base_files = sorted(f for f in os.listdir(base_dir) if f.endswith(".npz"))
        new_files = sorted(f for f in os.listdir(new_dir) if f.endswith(".npz"))
        if not base_files or base_files != new_files:
            sys.exit("metrics differ between baseline %s and this checkout %s" % (base_files, new_files))
        failures, worst = [], {}
        for fname in base_files:
            metric = fname[:-4]
            a, b = np.load(os.path.join(base_dir, fname)), np.load(os.path.join(new_dir, fname))
            if sorted(a.files) != sorted(b.files):
                failures.append((metric, "set of outputs", float("nan")))
                continue
            for key in a.files:
                x, y = a[key], b[key]
                if x.shape != y.shape:
                    failures.append((metric, key + " (shape %s vs %s)" % (x.shape, y.shape), float("nan")))
                    continue
                exact = metric in EXACT or (metric, key.split("__", 1)[-1]) in EXACT_KEYS
                scale = max(float(np.max(np.abs(x))), 1e-300) if x.size else 1.0
                diff = float(np.max(np.abs(x - y))) if x.size else 0.0
                worst[metric] = max(worst.get(metric, 0.0), diff / scale)
                ok = np.array_equal(x, y, equal_nan=True) if exact else np.allclose(y, x, rtol=STRICT_RTOL, atol=STRICT_RTOL * scale, equal_nan=True)
                if not ok:
                    failures.append((metric, key, diff / scale))
        print("%-26s %s" % ("metric", "worst relative difference"))
        for metric, d in worst.items():
            print("%-26s %.2e%s" % (metric, d, "  (exact required)" if metric in EXACT or any(m == metric for m, _ in EXACT_KEYS) else ""))
        if failures:
            print("\nDIFFERENCES BEYOND STRICT TOLERANCE:")
            for metric, key, d in failures:
                print("  %s / %s  rel diff %.3e" % (metric, key, d))
            sys.exit(1)
        print("\nOK: no difference beyond the strict tolerances.")


if __name__ == "__main__":
    main()
