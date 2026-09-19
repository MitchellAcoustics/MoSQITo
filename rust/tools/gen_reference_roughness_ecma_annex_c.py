#!/usr/bin/env python3
"""Extract ECMA-418-2 Annex C's own published roughness reference values,
plus the Zwicker & Fastl (1990, fig. 11.2) reference curve at the same grid
points, from `validations/sq_metrics/roughness_ecma/input/references.py`.

This is the standard's own oracle (`ref_ecma`) plus the classic psychoacoustic
reference (`ref_zf`) MoSQITo's own validation script gates against — not
MoSQITo's computed `roughness_ecma` output, which this port does not use as
an oracle (see `DEVIATIONS.md` and `conformance_roughness_ecma.rs`'s module
doc: MoSQITo's own `_lowpass_filter` bug and uncalibrated `c_R` make its
output non-conformant).

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_reference_roughness_ecma_annex_c.py
"""

from __future__ import annotations

import json
import pathlib
import sys

OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "crates/mosqito-core/tests/roughness_ecma_annex_c_reference.json"
)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "validations/sq_metrics/roughness_ecma"))

from input.references import ref_ecma, ref_zf  # noqa: E402

FC_LIST = [125, 250, 500, 1000, 2000, 4000, 8000]
FMOD_VECTOR = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 300, 400]


def main() -> None:
    out = []
    for fc in FC_LIST:
        for fmod in FMOD_VECTOR:
            out.append(
                {
                    "fc": fc,
                    "fmod": fmod,
                    "ref_ecma": float(ref_ecma(fc, fmod)),
                    "ref_zf": float(ref_zf(fc, fmod)),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes), {len(out)} points")


if __name__ == "__main__":
    main()
