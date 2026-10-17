# -*- coding: utf-8 -*-
"""Deterministic test signals shared by the parity goldens and their tests.

Every signal is a ``(samples, fs)`` pair and is deterministic (closed form,
seeded RNG, or a committed wav), so the generator and the tests always see
identical input. None is longer than 1 s.

Each metric is only run on the signals it is meant for (``ISO_SIGNALS``,
``ECMA_SIGNALS``, ``ROUGHNESS_SIGNALS``): the roughness metrics are defined for
modulated tones, the ECMA-418-2 metrics for 48 kHz input.
"""
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PINK_WAV = os.path.join(REPO_ROOT, "tests", "input", "Test signal 5 (pinknoise 60 dB).wav")


FS = 48000


def _scale_to_spl(x, spl_db):
    """Scale ``x`` to the given RMS sound pressure level (re 20 uPa)."""
    rms = np.sqrt(np.mean(x ** 2))
    return x * (2e-5 * 10 ** (spl_db / 20) / rms)


def am_tone(fs=FS, dur=1.0):
    """1 kHz carrier, 70 Hz sinusoidal modulation, 60 dB (as in the roughness validations)."""
    t = np.arange(int(fs * dur)) / fs
    x = 0.5 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)
    return _scale_to_spl(x, 60)


def white_noise():
    return _scale_to_spl(np.random.default_rng(0).standard_normal(FS), 60)


def two_tones():
    """500 + 600 Hz: a tone complex with a sparse spectrum."""
    t = np.arange(FS) / FS
    return _scale_to_spl(np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 600 * t), 60)


def burst_then_quiet():
    """0.25 s of loud noise followed by 0.75 s about 40 dB lower."""
    x = np.random.default_rng(1).standard_normal(FS)
    x[12000:] *= 0.01
    return _scale_to_spl(x, 70)


def odd_length():
    """Modulated 1 kHz tone of 44 123 samples: not a multiple of any hop size used by the metrics."""
    t = np.arange(44123) / FS
    return _scale_to_spl(np.sin(2 * np.pi * 1000 * t) * (1 + 0.5 * np.sin(2 * np.pi * 50 * t)), 65)


def pink_noise_wav():
    """First second of the ISO 532-1 pink noise test signal (60 dB), in Pa.

    ``2 * sqrt(2)`` is the ``wav_calib`` the loudness tests use for this file.
    """
    from scipy.io import wavfile

    fs, x = wavfile.read(PINK_WAV)
    return x[:fs].astype(float) / 32768.0 * 2 * np.sqrt(2), fs


def signals():
    """Return ``{name: (samples, fs)}``."""
    wav, wav_fs = pink_noise_wav()
    return {
        "am_tone": (am_tone(), FS),
        "white_noise": (white_noise(), FS),
        "two_tones": (two_tones(), FS),
        "burst_then_quiet": (burst_then_quiet(), FS),
        "odd_length": (odd_length(), FS),
        "am_tone_44k1": (am_tone(fs=44100), 44100),
        "pink_noise_wav": (wav, wav_fs),
    }


ISO_SIGNALS = ("am_tone", "white_noise", "two_tones", "burst_then_quiet", "odd_length", "am_tone_44k1", "pink_noise_wav")
ECMA_SIGNALS = tuple(s for s in ISO_SIGNALS if s != "am_tone_44k1")  # ECMA-418-2 needs 48 kHz
ROUGHNESS_SIGNALS = ("am_tone", "odd_length")


def kernel_inputs():
    """Seeded inputs for the kernel-level goldens.

    ``core_loudness`` is 21 bands x 400 frames with every third frame scaled down
    by 100 so the non-linear decay sees both rises and decays;
    ``third_octave_levels`` is 28 bands x 300 frames, with the lowest 11 bands
    capped at 119 dB (the range ``_main_loudness`` accepts).
    """
    rng = np.random.default_rng(2)
    core = np.abs(rng.standard_normal((21, 400))) * rng.uniform(0, 5, (21, 1))
    core[:, ::3] *= 0.01
    third_octave = rng.uniform(10, 100, (28, 300)) + rng.uniform(-15, 15, (28, 1))
    third_octave[:11] = np.minimum(third_octave[:11], 119)
    return {"core_loudness": core, "third_octave_levels": third_octave}
