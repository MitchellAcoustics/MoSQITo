"""Small validation helpers shared by mosqito_rs's Python wrappers."""

from __future__ import annotations

import numpy as np


def require_1d(*arrays: np.ndarray, fn_name: str) -> None:
    """Raises ``NotImplementedError`` if any of `arrays` is not 1-D.

    Several entry points (`loudness_zwst_freq`, `sharpness_din_freq`,
    `noct_synthesis`) only port MoSQITo's 1-D case for now; this is the
    message and check all of them share.
    """
    if any(a.ndim != 1 for a in arrays):
        raise NotImplementedError(
            f"mosqito_rs.{fn_name} currently supports a 1-D spectrum only; "
            "the 2-D (per-segment) case has not been ported yet"
        )
