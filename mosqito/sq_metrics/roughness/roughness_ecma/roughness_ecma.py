import numpy as np
from numpy.fft import fft
from scipy.signal import hilbert, resample, decimate

# Project Imports
from mosqito.sq_metrics.loudness.loudness_ecma._preprocessing import _preprocessing
from mosqito.sq_metrics.loudness.loudness_ecma._band_blocks import (
    band_blocks,
    band_specific_loudness,
    ear_filtered,
    ecma_block_index,
    ecma_time_axis,
)
from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import _auditory_filters_centre_freq
from mosqito.utils._parallel import parallel_map, resolve_workers

from mosqito.sq_metrics.roughness.roughness_ecma._weighting import _f_max, _r_max, _Q2_high, _Q2_low, _high_mod_rate_weighting, _low_mod_rate_weighting
from mosqito.sq_metrics.roughness.roughness_ecma._estimate_fund_mod_rate import _estimate_fund_mod_rate
from mosqito.sq_metrics.roughness.roughness_ecma._peak_picking import _peak_picking
from mosqito.sq_metrics.roughness.roughness_ecma._von_hann_window import _von_hann_window
from mosqito.sq_metrics.roughness.roughness_ecma._noise_reduction import _noise_reduction
from mosqito.sq_metrics.roughness.roughness_ecma._interpolation_50 import _interpolation_50
from mosqito.sq_metrics.roughness.roughness_ecma._non_linear_transform import _non_linear_transform
from mosqito.sq_metrics.roughness.roughness_ecma._lowpass_filter import _lowpass_filter

def roughness_ecma(signal, fs, workers=None):
    """Calculation of the specific and total roughness according to ECMA-418-2
    (2nd Ed, 2022).

    This function computes the acoustic loudness according to ECMA-418-2 (section 7) method for
    stationary signals. The calculation is based on the Hearing Model (HMS) used in loudness_ecma aswell.

    Parameters
    ----------
    signal: numpy.array
        Signal time values [Pa]. The sampling frequency of the signal must be 48000 Hz.
    fs: int
        Sampling frequency [Hz].
    workers: int, optional
        Number of threads used to process the 53 bands in parallel. Default is
        1 (no threads). More workers than bands have no effect; each thread
        holds the working arrays of one band, which grow with the signal duration.

    Returns
    -------
    R : float
        Overall roughness representative value [asper_HMS].
    R_time : numpy.ndarray
        Roughness over time [asper_HMS], size (Ntime,).
    R_specific : numpy.ndarray
        Specific roughness [asper_HMS/bark], size (Nbark, Ntime).
	    Each of the 53 elements of the list corresponds to the time-dependant specific roughness for a given bark band. 
    bark_axis : numpy.ndarray
        Corresponding bark axis, size (Nbark,).
    time_axis : numpy.ndarray
        Time axis, size (Ntime,).

    Warning
    -------
    The sampling frequency of the signal must be 48 kHz.

    See Also
    --------
    .roughness_dw : Daniel and Weber roughness computation
    .loudness_ecma : Loudness computation based on the hearing model of ECMA 418-2

    References
    ----------
    :cite:empty:`R_ecma-ECMA-418-2`

    .. bibliography::
        :keyprefix: R_ecma-

    Examples
    --------
    .. plot::
       :include-source:

        >>> from mosqito.sq_metrics import roughness_ecma
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> f=1000
        >>> fs=48000
        >>> d=1
        >>> dB=60
        >>> fmod = 70
        >>> fc = 1000
        >>> mdepth = 1
        >>> time = np.arange(0, d, 1/fs)
        >>> signal = (0.5* (1 + mdepth * (np.sin(2 * np.pi * fmod * time)))
        >>>         * np.sin(2 * np.pi * fc * time)
        >>>     )    
        >>> rms = np.sqrt(np.mean(np.power(signal, 2)))
        >>> ampl = 0.00002 * np.power(10, dB / 20) / rms
        >>> stimulus = signal * ampl
        >>> R, R_time, R_spec, bark_axis, time_axis = roughness_ecma(stimulus, fs)
        >>> plt.step(bark_axis, R_spec)
        >>> plt.xlabel("Bark axis [Bark]")
        >>> plt.ylabel("Specific roughness [Asper/Bark]")
        >>> plt.title("Roughness = " + f"{R:.2f}" + " [Asper]")
    """
    
    workers = resolve_workers(workers)

    # Check on the sampling frequency
    if fs != 48000:
        print(
            "[Warning] Signal resampled to 48 kHz fulfill the standard requirements and allow calculation."
        )
        signal = resample(signal, int(48000 * len(signal) / fs))
        fs = 48000
    
    # INITIALIZE COMPUTATION PARAMETERS
    # Number of critical bands and their center frequency
    CBF = 53
    center_freq = _auditory_filters_centre_freq()
    # Hop size and block size for specific loudness calculation (7.1.1)
    sb=16384
    sh=4096
    duration =  len(signal) / fs
        
    # Preprocessing 
    signal, n_new = _preprocessing(signal, sb, sh)
    # Outer and middle ear filtering
    ear_signal = ear_filtered(signal)
    # Time segmentation: the same block indices for every band
    index = ecma_block_index(n_new, sb, sh)
    time_axis = ecma_time_axis(len(signal), index)

    # Downsampling to 1500 Hz
    sbb = 512 
    downsampling_factor = 32

    # Each band is filtered, segmented, reduced to its specific loudness and
    # converted to a downsampled envelope on its own (7.1.2), so that the
    # (53, blocks, block size) array of all bands is never held in memory.
    def one_band(band):
        blocks = band_blocks(ear_signal, band, index)
        n_spec = band_specific_loudness(blocks, band)
        # ENVELOPE CALCULATION AND DOWNSAMPLING (7.1.2)
        envelope = abs(hilbert(blocks, axis=-1))
        envelope = decimate(envelope, downsampling_factor // 4, axis=-1)
        envelope = decimate(envelope, 4, axis=-1)
        return n_spec, envelope

    results = parallel_map(one_band, range(CBF), workers)
    # Specific loudness, shape (Ntime, 53), and envelopes, shape (Ntime, 53, sbb)
    N_specific = np.array([r[0] for r in results]).T
    envelopes_downsampled = np.stack([r[1] for r in results], axis=1)
    bark_axis = np.linspace(0.5, 26.5, num=53, endpoint=True)
    L = N_specific.shape[0]

    # CALCULATION OF SCALED POWER SPECTRUM (7.1.3)
    
    # Maximum loudness in each time block
    N_specific_max = np.asarray(N_specific).max(axis=1)
    
    # Hann window is precisely defined in the standard (different from numpy version)
    hann_window = _von_hann_window(sbb)
    phi_E0 = np.sum(np.power(envelopes_downsampled * hann_window,2), axis=2)
    den = N_specific_max[:,np.newaxis] * phi_E0
    
    dft = (abs(fft((envelopes_downsampled * hann_window), axis=2)[:,:,:sbb//2])/2*np.sqrt(2))**2
    scaling = np.zeros((L, CBF))
    scaling[den!=0] = np.power(N_specific[den!=0],2) / den[den!=0]
    phi_E = scaling[:, :, np.newaxis] * dft
    
    # NOISE REDUCTION OF THE ENVELOPES (7.1.4)
    Phi_E = _noise_reduction(phi_E)
    
    # Critical bands characteristics for the weightings to come
    fmax = _f_max(center_freq) # center_freq = central frequency of the band z (eq 86 section 7.1.5.2)
    rmax = _r_max(center_freq)
    q2_high = _Q2_high(center_freq)
    q2_low = _Q2_low(center_freq)

    amplitude = np.zeros((L,CBF))
    for l in range(L):       
        for z in range(CBF):
        
            # SPECTRAL WEIGHTING (7.1.5)
            f_p, Ai = _peak_picking(Phi_E[l, z, :])                            
            N_peak = len(f_p)
            
            if N_peak == 0:
                amplitude[l,z] = 0
            else:
                Ai_tilde = np.empty(N_peak)
                for i0 in range(N_peak):
                    # Weighting of high modulation rates
                    Ai_tilde[i0] = _high_mod_rate_weighting(f_p[i0], Ai[i0], fmax[z], rmax[z], q2_high[z])         
                
                # Estimation of fundamental modulation rate
                mod_rate, A_hat = _estimate_fund_mod_rate(f_p, Ai_tilde)
                
                # Weighting of low modulation rates
                amplitude[l,z] = _low_mod_rate_weighting(mod_rate, A_hat, fmax[z], q2_low[z])

    amplitude[amplitude<0.074376]=0
    
    # TODO: OPTIONAL ENTROPY WEIGHTING (specific to ITT equipment): needs a signal of rotational speed (7.1.6)

    # CALCULATION OF TIME DEPENDENT SPECIFIC ROUGHNESS (7.1.7)
    # Interpolation to 50 Hz
    amplitude_50, t_50 = _interpolation_50(amplitude, time_axis, duration)
    R_est = np.clip(amplitude_50, 0, None)
    # Non linear transformation 
    R_time_spec_temp = _non_linear_transform(R_est)
    # Lowpass filtering
    R_time_spec = _lowpass_filter(R_time_spec_temp)

    # CALCULATION OF REPRESENTATIVE VALUES (7.1.8)
    # CBF dependent value
    R_spec = np.mean(R_time_spec[10:,:], axis=0) 
    # Time_dependent value
    R_time = 0.5 * np.sum(R_time_spec, axis=1)
    # Single representative value
    R = np.percentile(R_time, 90)
        
    return R, R_time, R_spec, bark_axis, t_50
