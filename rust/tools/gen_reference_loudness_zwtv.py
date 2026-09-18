#!/usr/bin/env python3
"""Extract ISO 532-1 Annex B.4/B.5's own published reference values for
time-varying loudness (not MoSQITo's computed output) from the xlsx corpus
under `validations/sq_metrics/loudness_zwtv/input/`.

This is the standard's own oracle: column B (N vs. time) and column L
(specific loudness vs. time, at a single reference Bark value) of each
signal's sheet, `skiprows=10` as MoSQITo's own validation script reads it
(`validations/sq_metrics/loudness_zwtv/validation_loudness_zwtv.py:257-283`).
Written once as JSON so the Rust conformance test doesn't need an xlsx
parser; re-run only if the corpus itself changes.

Run from the ``rust/`` directory::

    ../.venv/bin/python tools/gen_reference_loudness_zwtv.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from pandas import ExcelFile, read_excel

OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "crates/mosqito-core/tests/reference_loudness_zwtv_annex_b.json"
)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
INPUT_ROOT = REPO_ROOT / "validations/sq_metrics/loudness_zwtv/input"

SIGNALS = [
    {
        "data_file": "ISO_532-1/Annex B.4/Test signal 6 (tone 250 Hz 30 dB - 80 dB).wav",
        "xls": "ISO_532-1/Annex B.4/Results and tests for synthetic signals (time varying loudness).xlsx",
        "tab": "Test signal 6",
        "n_specif_bark": 2.5,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.4/Test signal 7 (tone 1 kHz 30 dB - 80 dB).wav",
        "xls": "ISO_532-1/Annex B.4/Results and tests for synthetic signals (time varying loudness).xlsx",
        "tab": "Test signal 7",
        "n_specif_bark": 8.5,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.4/Test signal 8 (tone 4 kHz 30 dB - 80 dB).wav",
        "xls": "ISO_532-1/Annex B.4/Results and tests for synthetic signals (time varying loudness).xlsx",
        "tab": "Test signal 8",
        "n_specif_bark": 17.5,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.4/Test signal 9 (pink noise 0 dB - 50 dB).wav",
        "xls": "ISO_532-1/Annex B.4/Results and tests for synthetic signals (time varying loudness).xlsx",
        "tab": "Test signal 9",
        "n_specif_bark": 17.5,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.4/Test signal 10 (tone pulse 1 kHz 10 ms 70 dB).wav",
        "xls": "ISO_532-1/Annex B.4/Results and tests for synthetic signals (time varying loudness).xlsx",
        "tab": "Test signal 10",
        "n_specif_bark": 8.5,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.4/Test signal 11 (tone pulse 1 kHz 50 ms 70 dB).wav",
        "xls": "ISO_532-1/Annex B.4/Results and tests for synthetic signals (time varying loudness).xlsx",
        "tab": "Test signal 11",
        "n_specif_bark": 8.5,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.4/Test signal 12 (tone pulse 1 kHz 500 ms 70 dB).wav",
        "xls": "ISO_532-1/Annex B.4/Results and tests for synthetic signals (time varying loudness).xlsx",
        "tab": "Test signal 12",
        "n_specif_bark": 8.5,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.4/Test signal 13 (combined tone pulses 1 kHz).wav",
        "xls": "ISO_532-1/Annex B.4/Results and tests for synthetic signals (time varying loudness).xlsx",
        "tab": "Test signal 13",
        "n_specif_bark": 8.5,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 14 (propeller-driven airplane).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 14",
        "n_specif_bark": None,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 15 (vehicle interior 40 kmh).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 15",
        "n_specif_bark": None,
        "field": "diffuse",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 16 (hairdryer).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 16",
        "n_specif_bark": None,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 17 (machine gun).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 17",
        "n_specif_bark": None,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 18 (hammer).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 18",
        "n_specif_bark": None,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 19 (door creak).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 19",
        "n_specif_bark": None,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 20 (shaking coins).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 20",
        "n_specif_bark": None,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 21 (jackhammer).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 21",
        "n_specif_bark": None,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 22 (ratchet wheel (large)).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 22",
        "n_specif_bark": None,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 23 (typewriter).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 23",
        "n_specif_bark": None,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 24 (woodpecker).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 24",
        "n_specif_bark": None,
        "field": "free",
    },
    {
        "data_file": "ISO_532-1/Annex B.5/Test signal 25 (full can rattle).wav",
        "xls": "ISO_532-1/Annex B.5/Results and tests for technical signals (time varying loudness).xlsx",
        "tab": "Test signal 25",
        "n_specif_bark": None,
        "field": "free",
    },
]


def extract_column(xls_file: ExcelFile, tab: str, col: str) -> list[float]:
    values = (
        read_excel(xls_file, sheet_name=tab, header=None, skiprows=10, usecols=col)
        .squeeze("columns")
        .to_numpy()
    )
    values = values[~np.isnan(values)]
    return values.tolist()


def main() -> None:
    out = []
    for sig in SIGNALS:
        xls_file = ExcelFile(INPUT_ROOT / sig["xls"])
        n_iso = extract_column(xls_file, sig["tab"], "B")
        n_specif_iso = extract_column(xls_file, sig["tab"], "L")
        out.append(
            {
                "data_file": sig["data_file"],
                "tab": sig["tab"],
                "field": sig["field"],
                "n_specif_bark": sig["n_specif_bark"],
                "N_iso": n_iso,
                "N_specif_iso": n_specif_iso,
            }
        )
        print(f"{sig['tab']}: {len(n_iso)} N samples")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes), {len(out)} signals")


if __name__ == "__main__":
    main()
