# -*- coding: utf-8 -*-

# Standard imports
from numpy import (
    concatenate, zeros, arange, where, interp, mean, real, power, sqrt, abs, empty,
    floor, ceil, minimum, take_along_axis, broadcast_to, errstate,
)
from numpy.fft import fft, ifft
import math

# Local imports
from mosqito.utils.LTQ import LTQ
from mosqito.sq_metrics.roughness.roughness_dw._ear_filter_coeff import (
    _ear_filter_coeff,
)
from mosqito.utils import freq2bark, db2amp, amp2db, bark2freq


def _roughness_dw_main_calc(spec, freq_axis, fs, gzi, hWeight):
    """
    Daniel and Weber roughness main calculation

    Parameters
    ----------
    spec : array
        An amplitude or complex spectrum.
    freq_axis : array
        Frequency axis in [Hz].
    fs : integer
        Sampling frequency.
    gzi : array
        gzi weighting function.
    hWeight : array
        H weighting function.

    Returns
    -------
    R : float
        Roughness computed for the given spectrum.
    R_spec : numpy.ndarray
        Specific roughness of the 47 channels.
    zi : numpy.ndarray
        Channel centres in bark.

    Notes
    -----
    The 47 channels are processed together as arrays (one batched FFT for all of
    them) instead of one channel and one spectral line at a time.
    """
    if len(spec) != len(freq_axis):
        raise ValueError(
            "spectrum and frequency axis should have the same number of points !"
        )

    # convert spectrum to 2-sided
    spec = concatenate((spec, spec[len(spec)::-1]))
    n = len(spec)

    # Frequency axis in Bark
    bark_axis = freq2bark(freq_axis)

    # Highest frequency
    nZ = arange(1, n // 2 + 1, 1)

    # Calculate Zwicker a0 factor (transfer characteristic of the outer and inner ear)
    a0 = zeros((n))
    a0[nZ - 1] = db2amp(_ear_filter_coeff(bark_axis), ref=1)
    spec = a0 * spec

    # Conversion of the spec into dB
    module = abs(spec[0 : n // 2])
    spec_dB = amp2db(module, ref=2e-5)

    # Find the audible components within the spec
    threshold = LTQ(bark_axis, reference="roughness")
    audible_index = where(spec_dB > threshold)[0]
    n_aud = len(audible_index)
    n_channel = 47

    # --------------------------------- stage 1 ------------------------------------
    # ----------------Creation of the specific excitations functions----------------
    level = spec_dB[audible_index]
    bark = bark_axis[audible_index]
    # Terhardt's slopes definition: lower slope [dB/Bark], upper slope [dB/Bark]
    s1 = -27
    s2 = minimum(-24 - (230 / freq_axis[audible_index]) + (0.2 * level), 0)

    # The excitation pattern are calculated for 47 overlapping 1-bark-wide channels
    zi = arange(1, n_channel + 1) / 2
    # Center frequencies for each channel
    zb = bark2freq(zi) * n / fs
    # Minimum excitation level
    minExcitDB = interp(zb, nZ, threshold)

    # Lower and higher channel limits of each component
    ch_low = floor(2 * bark) - 1
    ch_high = ceil(2 * bark) - 1

    # Excitation pattern, shape (n_aud, n_channel): a component excites the
    # channels below it along the lower slope and those above along the upper one
    channel = arange(n_channel)[None, :]
    centre = (channel + 1) * 0.5
    sl_low = s1 * (bark[:, None] - centre) + level[:, None]
    sl_high = s2[:, None] * (centre - bark[:, None]) + level[:, None]
    slopes = zeros((n_aud, n_channel))
    below = (channel <= ch_low[:, None]) & (sl_low > minExcitDB[None, :])
    slopes[below] = db2amp(sl_low[below], ref=0.00002)
    above = (channel >= ch_high[:, None]) & (sl_high > minExcitDB[None, :])
    slopes[above] = db2amp(sl_high[above], ref=0.00002)

    # Amplitude of each component in each channel, shape (n_channel, n_aud)
    i = arange(n_channel)[:, None]
    from_above = take_along_axis(
        slopes.T, broadcast_to(minimum(i + 1, n_channel - 1), (n_channel, n_aud)), axis=0
    )
    from_below = take_along_axis(slopes.T, broadcast_to(i - 1, (n_channel, n_aud)), axis=0)
    audible_module = module[audible_index][None, :]
    ampl = where(
        (ch_low[None, :] == i) | (ch_high[None, :] == i),
        1.0,
        where(ch_high[None, :] > i, from_above / audible_module, from_below / audible_module),
    )

    # Reconstruction of the spectrum of each channel
    exc = zeros((n_channel, n), dtype=complex)
    exc[:, audible_index] = ampl * spec[audible_index][None, :]

    # The temporal specific excitation functions are obtained by IFFT
    temporal_excitation = abs(n * real(ifft(exc, axis=1)))

    # ------------------------------- stage 2 --------------------------------------
    # ---------------------modulation depth calculation-----------------------------
    # The fluctuations of the envelope are contained in the low frequency part
    # of the spec of specific excitations in absolute value
    h0 = mean(temporal_excitation, axis=1)
    envelope_spec = fft(temporal_excitation - h0[:, None], axis=1)

    # This spec is weighted to model the low-frequency bandpass
    # characteristic of the roughness on modulation frequency
    envelope_spec = envelope_spec * hWeight

    # The time functions of the bandpass filtered envelopes hBPi(t)
    # are calculated via inverse Fourier transform :
    hBP = 2 * real(ifft(envelope_spec, axis=1))

    # Modulation depth estimation is given by envelope RMS values
    # and excitation functions time average :
    hBPrms = sqrt(mean(power(hBP, 2), axis=1))
    positive = h0 > 0
    mod_depth = zeros(n_channel)
    mod_depth[positive] = minimum(hBPrms[positive] / h0[positive], 1.0)

    # ------------------------------- stage 3 --------------------------------------
    # ----------------roughness calculation with cross correlation------------------
    # Crosscorrelation coefficients between the envelopes of the channels
    # i and i+2 with dz= 1 bark. Channels without modulation hold only rounding
    # noise (modulation depth of order 1e-16) and are left out.
    ki = zeros((47))
    pair = (mod_depth[:45] > 1e-10) & (mod_depth[2:47] > 1e-10)
    if pair.any():
        centred = hBP - hBP.mean(axis=1, keepdims=True)
        energy = (centred * centred).sum(axis=1)
        cross = (centred[:45] * centred[2:47]).sum(axis=1)
        ki[:45][pair] = cross[pair] / sqrt(energy[:45][pair] * energy[2:47][pair])

    # Specific roughness calculation with gzi the modulation depth weighting
    # function given by Aures
    R_spec = empty((47))
    R_spec[0] = gzi[0] * pow(mod_depth[0] * ki[0], 2)
    R_spec[1] = gzi[1] * pow(mod_depth[1] * ki[1], 2)
    for i in arange(2, 45):
        R_spec[i] = gzi[i] * pow(mod_depth[i] * ki[i] * ki[i - 2], 2)
    R_spec[45] = gzi[45] * pow(mod_depth[45] * ki[43], 2)
    R_spec[46] = gzi[46] * pow(mod_depth[46] * ki[44], 2)

    # Total roughness calculation with calibration factor of 0.25 given in the article
    # to produce a roughness of 1 asper for a 1-kHz, 60dB tone with carrier frequency
    # of 70 Hz and a modulation depth of 1
    R = 0.25 * sum(R_spec)

    return R, R_spec, zi
