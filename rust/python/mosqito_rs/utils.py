"""Small shared utilities: time segmentation and signal loading.

``time_segmentation`` wraps the compiled bindings in :mod:`mosqito_rs._core`
to match ``mosqito.utils.time_segmentation``'s exact public signature.
``load`` is a thin pure-Python convenience function (file I/O, not a hot
path) rather than a Rust-backed one — see ``DEVIATIONS.md``.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["load", "time_segmentation"]


def time_segmentation(
    sig: np.ndarray,
    fs: float,
    nperseg: int = 2048,
    noverlap: int | None = None,
    is_ecma: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment a time signal into overlapping blocks.

    Matches :func:`mosqito.utils.time_segmentation` for its ``is_ecma=False``
    case — the only one any real MoSQITo caller uses (``roughness_dw``,
    ``tnr_ecma_perseg``, ``pr_ecma_perseg``). ``loudness_ecma``'s own
    internal segmentation (``is_ecma=True``) is implemented separately in
    this port's ``loudness_ecma`` module rather than through this shared
    function, so that path isn't reachable here either — see
    ``DEVIATIONS.md``.

    Parameters
    ----------
    sig : numpy.ndarray
        A 1-dimensional time signal array.
    fs : float
        The time signal sampling frequency.
    nperseg : int, default 2048
        Length of each segment.
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg / 2``. Defaults to None.
    is_ecma : bool, default False
        Not supported — raises :class:`NotImplementedError` if True.

    Returns
    -------
    block_array : numpy.ndarray
        A 2-dimensional array of shape ``(nperseg, nseg)`` containing the
        segmented signal.
    time : numpy.ndarray
        The time axis corresponding to the segmented signal, shape ``(nseg,)``.
    """
    if is_ecma:
        raise NotImplementedError(
            "mosqito_rs.time_segmentation does not support is_ecma=True; "
            "loudness_ecma's own segmentation is implemented separately, "
            "see DEVIATIONS.md"
        )
    sig = np.ascontiguousarray(sig, dtype=np.float64)
    return _core.time_segmentation(sig, float(fs), int(nperseg), noverlap)


def load(
    file: str,
    wav_calib: float | None = None,
    mat_signal: str = "",
    mat_fs: str = "",
) -> tuple[np.ndarray, int]:
    """Load a signal from a ``.wav`` file, resampled to 48 kHz.

    Matches :func:`mosqito.utils.load` for its ``.wav`` case only — the only
    one ported. ``.mat`` and ``.uff`` are not supported (the latter needs
    ``pyuff``, a niche dependency this project does not otherwise need);
    both raise :class:`NotImplementedError`. See ``DEVIATIONS.md``.

    Parameters
    ----------
    file : str
        Path to the signal file. Must end in ``.wav``/``.WAV``.
    wav_calib : float, optional
        Calibration factor [Pa/FS]: level of the signal in Pa_peak
        corresponding to the full scale of the .wav file. If None, a
        calibration factor of 1 is considered.
    mat_signal, mat_fs : str, unused
        Accepted for signature compatibility with MoSQITo; only meaningful
        for the unsupported ``.mat`` case.

    Returns
    -------
    signal : numpy.ndarray
        Time signal values [Pa].
    fs : int
        Sampling frequency [Hz], always 48000 (resampled if needed).
    """
    if not (file.endswith(".wav") or file.endswith(".WAV")):
        raise NotImplementedError(
            "mosqito_rs.load currently supports .wav files only; "
            ".mat/.uff support has not been ported, see DEVIATIONS.md"
        )

    from scipy.io import wavfile
    from scipy.signal import resample

    fs, signal = wavfile.read(file)
    if signal.ndim > 1:
        signal = signal[:, 0]

    calib = 1.0 if wav_calib is None else wav_calib
    if np.issubdtype(signal.dtype, np.int16):
        signal = calib * signal / (2**15 - 1)
    elif np.issubdtype(signal.dtype, np.int32):
        signal = calib * signal / (2**31 - 1)
    elif np.issubdtype(signal.dtype, np.floating):
        signal = calib * signal
    else:
        raise NotImplementedError(
            f"mosqito_rs.load does not support wav sample dtype {signal.dtype}"
        )

    if fs != 48000:
        signal = resample(signal, int(48000 * len(signal) / fs))
        fs = 48000

    return signal.astype(np.float64), fs
