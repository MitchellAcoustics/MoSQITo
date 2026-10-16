# -*- coding: utf-8 -*-

# Standard library imports
from math import floor
from numpy import array, append, tile, empty, asarray, ascontiguousarray, int64

# Upper limits of approximated critical bands in terms of critical band rate
_ZUP = array(
    [0.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6, 12.3, 13.8, 15.2, 16.7,
     18.1, 19.3, 20.6, 21.8, 22.7, 23.6, 24]
)
# Range of specific loudness for the determination of the steepness of the
# upper slopes in the specific loudness - critical band rate pattern
_RNS = array(
    [21.5, 18, 15.1, 11.5, 9, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.30, 0.22,
     0.15, 0.10, 0.035, 0]
)
# Steepness of the upper slopes in the specific loudness = Critical band rate
# pattern for the ranges 'rns' as a function of the number of the critical band
_USL = array(
    [
        (13, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5),
        (9, 7.5, 6, 5.1, 4.5, 4.5, 4.5, 4.5),
        (7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9),
        (6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2),
        (4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7),
        (3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2),
        (2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7),
        (2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3),
        (1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1),
        (1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82),
        (0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62),
        (0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42),
        (0.40, 0.33, 0.26, 0.24, 0.24, 0.22, 0.22, 0.22),
        (0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17),
        (0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11),
        (0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08),
        (0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05),
        (0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02),
    ]
)
# Position of each band's upper limit in the 240-point (0.1 bark) pattern
_ZUP_EA = (_ZUP * 10).astype(int64)

_SPEC_LENGTH = 240


def _calc_slopes_frames(nm, N, spec, zup, zup_ea, rns, usl):
    """Specific loudness pattern and total loudness for every frame.

    Walks the 0.1 bark grid of each frame one position at a time, switching to
    the next slope segment at most once per position, as the reference
    implementation does. Scalar loops only (no numpy calls on slices, no helper
    functions) so that it runs unchanged as plain Python or compiled by a JIT.

    Parameters
    ----------
    nm : numpy.ndarray
        Core loudness, shape (21, n_frames).
    N : numpy.ndarray
        Output total loudness, shape (n_frames,).
    spec : numpy.ndarray
        Output specific loudness, shape (240, n_frames).
    zup, zup_ea, rns, usl : numpy.ndarray
        Constant tables (band limits in bark and in grid positions, loudness
        ranges, slope steepness).
    """
    for f in range(nm.shape[1]):
        for k in range(240):
            spec[k, f] = 0.0
        # Baseline fill: each band at its own core loudness
        prev = 0
        for i in range(20):
            end = min(zup_ea[i], 240)
            for k in range(prev, end):
                spec[k, f] = nm[i, f]
            prev = end

        n_total = 0.0
        z1 = 0.0
        n1 = 0.0
        for i in range(21):
            zup_i = zup[i]
            zea = min(zup_ea[i], 240)
            zprev = 0
            if i > 0:
                zprev = zup_ea[i - 1]
            nm_i = nm[i, f]
            col = max(i - 1, 0)
            if col > 7:
                col = 7

            # Rising (or flat): one rectangle at the band's own level
            if round(n1 * 1e8) / 1e8 <= round(nm_i * 1e8) / 1e8:
                n_total += nm_i * (zup_i - z1)
                z1 = zup_i
                n1 = nm_i
                continue

            # Falling: first decaying segment
            row = 0
            for r in range(18):
                if round(n1 * 1e8) / 1e8 < round(rns[r] * 1e8) / 1e8:
                    row += 1
            row = min(row, 17)
            target = max(rns[row], nm_i)
            steep = usl[row, col]
            sz1 = z1
            sn1 = n1
            sz2 = (sn1 - target) / steep + sz1
            clipped = sz2 >= zup_i
            if clipped:
                sz2 = zup_i
                sn2 = sn1 - (sz2 - sz1) * steep
            else:
                sn2 = target
            n_total += (sz2 - sz1) * (sn1 + sn2) / 2.0

            reached = False
            for k in range(zprev, zea):
                z = (k + 1) * 0.1
                if round(sz2 * 1e8) / 1e8 <= round(z * 1e8) / 1e8:
                    if round(sn2 * 1e8) / 1e8 <= round(nm_i * 1e8) / 1e8:
                        n_total += nm_i * (zup_i - sz2)
                        reached = True
                        break
                    row = 0
                    for r in range(18):
                        if round(sn2 * 1e8) / 1e8 <= round(rns[r] * 1e8) / 1e8:
                            row += 1
                    row = min(row, 17)
                    target = max(rns[row], nm_i)
                    steep = usl[row, col]
                    sz1 = sz2
                    sn1 = sn2
                    sz2 = (sn1 - target) / steep + sz1
                    clipped = sz2 >= zup_i
                    if clipped:
                        sz2 = zup_i
                        sn2 = sn1 - (sz2 - sz1) * steep
                    else:
                        sn2 = target
                    n_total += (sz2 - sz1) * (sn1 + sn2) / 2.0
                spec[k, f] = sn1 - (z - sz1) * steep

            if reached:
                z1 = zup_i
                n1 = nm_i
            else:
                z1 = sz2
                n1 = sn2

        if n_total < 0.0:
            n_total = 0.0
        if n_total <= 16.0:
            n_total = floor(n_total * 1000.0 + 0.5) / 1000.0
        else:
            n_total = floor(n_total * 100.0 + 0.5) / 100.0
        N[f] = n_total


def _calc_slopes(nm):
    """Specific loudness pattern and total loudness from the core loudness.

    Parameters
    ----------
    nm : numpy.ndarray
        Core loudness of the 21 bands, shape (21,) or (21, n_frames).

    Returns
    -------
    N : float or numpy.ndarray
        Total loudness [sone], rounded as specified by ISO 532-1.
    N_specific : numpy.ndarray
        Specific loudness over the 240-point (0.1 bark) axis, shape (240,) or
        (240, n_frames).
    """
    nm = asarray(nm, dtype=float)
    one_frame = nm.ndim == 1
    if one_frame:
        nm = nm[:, None]
    nm = ascontiguousarray(nm)
    N = empty(nm.shape[1])
    N_specific = empty((_SPEC_LENGTH, nm.shape[1]))
    _calc_slopes_frames(nm, N, N_specific, _ZUP, _ZUP_EA, _RNS, _USL)
    if one_frame:
        return N[0], N_specific[:, 0]
    return N, N_specific
