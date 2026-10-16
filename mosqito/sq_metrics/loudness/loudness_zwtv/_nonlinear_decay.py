# -*- coding: utf-8 -*-


# Standard library imports
from math import sqrt, exp
from numpy import copy, roll, zeros, empty, arange, asarray, ascontiguousarray


def _nl_coefficients(sample_rate=2000, nl_iter=24):
    """Filter constants B of the nonlinear decay (ISO 532-1, time constants of
    5 ms, 15 ms and 75 ms), as an array of 6 values."""
    t_short = 0.005
    t_long = 0.015
    t_var = 0.075
    delta_t = 1 / (sample_rate * nl_iter)
    P = (t_var + t_long) / (t_var * t_short)
    Q = 1 / (t_short * t_var)
    lambda_1 = -P / 2 + sqrt(P * P / 4 - Q)
    lambda_2 = -P / 2 - sqrt(P * P / 4 - Q)
    den = t_var * (lambda_1 - lambda_2)
    e1 = exp(lambda_1 * delta_t)
    e2 = exp(lambda_2 * delta_t)
    return asarray(
        [
            (e1 - e2) / den,
            ((t_var * lambda_2 + 1) * e1 - (t_var * lambda_1 + 1) * e2) / den,
            ((t_var * lambda_1 + 1) * e1 - (t_var * lambda_2 + 1) * e2) / den,
            (t_var * lambda_1 + 1) * (t_var * lambda_2 + 1) * (e1 - e2) / den,
            exp(-delta_t / t_long),
            exp(-delta_t / t_var),
        ]
    )


def _nl_kernel(ui, B, out, nl_iter):
    """Sequential recurrence of the nonlinear decay for every band.

    Written with scalar loops only (no numpy calls on slices) so that it runs
    unchanged as plain Python or compiled by a JIT: for 21 bands the cost of a
    numpy call per step is far larger than the arithmetic.

    Parameters
    ----------
    ui : numpy.ndarray
        Input upsampled by ``nl_iter``, shape (n_bands, n_frames * nl_iter).
    B : numpy.ndarray
        Filter constants from ``_nl_coefficients``.
    out : numpy.ndarray
        Output, shape (n_bands, n_frames), filled with the filtered loudness at
        the original frame positions.
    nl_iter : int
        Upsampling factor.

    Notes
    -----
    The state before the first step is read from index -1, i.e. the last
    column (the input value there, with a zero second state), as in the
    reference implementation.
    """
    n_bands = ui.shape[0]
    n_cols = ui.shape[1]
    for r in range(n_bands):
        uo_prev = ui[r, n_cols - 1]
        u2_prev = 0.0
        for col in range(n_cols):
            u = ui[r, col]
            # uo: output state, follows the input unless the decay is slower
            if uo_prev > u2_prev:
                cand = uo_prev * B[2] - u2_prev * B[3]
            else:
                cand = uo_prev * B[4]
            if cand >= u:
                uo = cand
            else:
                uo = u
            # u2: second state
            u2 = uo
            if u < uo_prev and uo_prev > u2_prev:
                u22 = uo_prev * B[0] - u2_prev * B[1]
                if u22 <= uo:
                    u2 = u22
            if u >= uo_prev:
                if not (abs(u - uo_prev) < 1e-5 and uo <= u2_prev):
                    u2 = (u2_prev - u) * B[5] + u
            if col % nl_iter == 0:
                out[r, col // nl_iter] = uo
            uo_prev = uo
            u2_prev = u2


def _nl_loudness(core_loudness):
    """Simulate the nonlinear temporal decay of the hearing system

    Parameters
    ----------
    core_loudness : numpy.ndarray
        Core loudness

    Outputs
    -------
    nl_loudness :  numpy.ndarray
        Loudness with non linear temporal decay
    """
    # Factor for virtual upsampling/inner iterations
    nl_iter = 24
    B = _nl_coefficients(2000, nl_iter)

    # Linear interpolation of the input between frames, nl_iter points per frame
    # (the last frame ramps down to zero)
    delta = copy(core_loudness)
    delta = roll(delta, -1, axis=1)
    delta[:, -1] = 0
    delta = (delta - core_loudness) / nl_iter
    ui_delta = zeros((core_loudness.shape[0], core_loudness.shape[1], nl_iter))
    ui_delta[:, :, 0] = core_loudness
    for i_in in arange(1, nl_iter):
        ui_delta[:, :, i_in] = ui_delta[:, :, i_in - 1] + delta
    ui_delta = ascontiguousarray(
        ui_delta.reshape(core_loudness.shape[0], core_loudness.shape[1] * nl_iter)
    )

    nl_loudness = empty(core_loudness.shape)
    _nl_kernel(ui_delta, B, nl_loudness, nl_iter)
    return nl_loudness
