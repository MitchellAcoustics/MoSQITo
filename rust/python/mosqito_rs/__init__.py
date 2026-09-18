"""Fast, standards-conformant psychoacoustic sound quality metrics.

``mosqito_rs`` is a Rust implementation of the metrics provided by the
`MoSQITo <https://github.com/Eomys/MoSQITo>`_ toolbox. The public functions
mirror MoSQITo's signatures exactly, so existing code can switch by changing
only the import::

    from mosqito_rs import loudness_zwst

Unlike MoSQITo, this package targets the published standards directly rather
than reproducing MoSQITo's Python output bit for bit. Every deliberate
divergence — including the ECMA-418-2 corrections from Wanty, Glesser and
Casagrande Hirono (INTER-NOISE 2024) — is recorded in ``DEVIATIONS.md``.
"""

from __future__ import annotations

from . import _core
from .loudness_ecma import loudness_ecma
from .loudness_zwst import loudness_zwst, loudness_zwst_freq, loudness_zwst_perseg
from .loudness_zwtv import loudness_zwtv
from .sharpness_din import (
    sharpness_din_freq,
    sharpness_din_from_loudness,
    sharpness_din_perseg,
    sharpness_din_st,
    sharpness_din_tv,
)
from .sound_level_meter import noct_spectrum, noct_synthesis

__all__ = [
    "__version__",
    "core_version",
    "noct_spectrum",
    "noct_synthesis",
    "loudness_zwst",
    "loudness_zwst_freq",
    "loudness_zwst_perseg",
    "loudness_zwtv",
    "loudness_ecma",
    "sharpness_din_from_loudness",
    "sharpness_din_st",
    "sharpness_din_freq",
    "sharpness_din_perseg",
    "sharpness_din_tv",
]

__version__ = "0.1.0"


def core_version() -> str:
    """Return the version of the compiled ``mosqito-core`` crate.

    Useful when diagnosing a mismatch between the installed wheel and the
    Python wrapper, which are versioned together but built separately.
    """
    return _core.core_version()
