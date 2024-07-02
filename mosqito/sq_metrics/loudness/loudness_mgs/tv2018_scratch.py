# Based on the Matlab code provided with ISO 532-3:2023
# tv2018.m

# calculate loudness according to Moore, Glasberg, & Schlittenlacher (2016)
# developed from:
# Glasberg & Moore (2002): time-varying part
# Moore & Glasberg (2007): binaural stationary loudness of each segment
# ANSI S3.4-2007: basis for Mooore & Glasberg (2007)

# %%
import numpy as np
from mosqito.sq_metrics.loudness.loudness_mgs.mgsutils import (
    sound_field_to_cochlea,
    filtered_signal_to_monaural_instantaneous_specific_loudness,
    instantaneous_specific_loudness_to_short_term_specific_loudness,
    monaural_specific_loudness_to_binaural_loudness_025,
    short_term_loudness_to_long_term_loudness,
    sone_to_phon_tv2018,
    load,
)


def tv2018(
    filename_sound: str,
    db_max: float,
    field_type: str = "free",
    s: np.array = None,
    fs: int = None,
):
    """
    Calculate loudness according to Moore, Glasberg, & Schlittenlacher (2016)

    Parameters
    ----------
    filename_sound : str
        path to the sound file
    db_max : float
        Calibration level: RMS level of a full-scale sinusoid
    filename_filter : str
        path to the filter file
    s : np.array
        Sound pressure time series
    fs : int
        Sampling frequency

    Returns
    -------
    loudness : float
    short_term_loudness : np.array
    long_term_loudness : np.array
    instantaneous_loudness_left : np.array
    instantaneous_loudness_right : np.array
    """
    if s is None or fs is None:
        print("[Info] Loading sound file...")
        s, fs = load(filename_sound)

    if fs != 32000:
        print("[Warning] Signal resampled to 32 kHz to allow calculation.")
        from scipy.signal import resample

        s = resample(s, int(32000 * len(s) / fs))
        fs = 32000

    # insert Fs / 1000 * 64 = 2048 zeros at the beginning and end to consider
    # components at all (specially at higher) frequencies at the beginning of
    # the first and at the end of the last block
    s = np.pad(s, ((2048, 2048), (0, 0)), mode="constant")

    print("[Info] Sound field to cochlea")
    s = sound_field_to_cochlea(s, field_type)

    print("[Info] Filtered signal to monaural instantaneous specific loudness")
    instantaneous_specific_loudness_left, instantaneous_specific_loudness_right = (
        filtered_signal_to_monaural_instantaneous_specific_loudness(s, fs, db_max)
    )

    # Shorten the first and last block
    instantaneous_specific_loudness_left = instantaneous_specific_loudness_left[
        32:-32, :
    ]
    instantaneous_specific_loudness_right = instantaneous_specific_loudness_right[
        32:-32, :
    ]

    # Start with zero
    instantaneous_specific_loudness_left[0, :] = 0
    instantaneous_specific_loudness_right[0, :] = 0

    print("[Info] Instantaneous specific loudness to short term specific loudness")
    short_term_specific_loudness_left, short_term_loudness_left = (
        instantaneous_specific_loudness_to_short_term_specific_loudness(
            instantaneous_specific_loudness_left
        )
    )  # NOTE: It's unclear why the short_term_loudness is returned from
    # instantaneous_specific_loudness_to_short_term_specific_loudness but not used
    short_term_specific_loudness_right, short_term_loudness_right = (
        instantaneous_specific_loudness_to_short_term_specific_loudness(
            instantaneous_specific_loudness_right
        )
    )

    short_term_loudness_left = np.zeros(short_term_specific_loudness_left.shape[0])
    short_term_loudness_right = np.zeros(short_term_specific_loudness_right.shape[0])

    print("[Info] Monaural specific loudness to binaural loudness")
    for i in range(short_term_specific_loudness_left.shape[0]):
        _, short_term_loudness_left[i], short_term_loudness_right[i] = (
            monaural_specific_loudness_to_binaural_loudness_025(
                short_term_specific_loudness_left[i, :],
                short_term_specific_loudness_right[i, :],
            )
        )

    print("[Info] Short term loudness to long term loudness")
    long_term_loudness_left = short_term_loudness_to_long_term_loudness(
        short_term_loudness_left
    )
    long_term_loudness_right = short_term_loudness_to_long_term_loudness(
        short_term_loudness_right
    )

    long_term_loudness = long_term_loudness_left + long_term_loudness_right

    loudness = np.max(long_term_loudness)

    short_term_loudness = short_term_loudness_left + short_term_loudness_right

    instantaneous_loudness_left = (
        np.sum(instantaneous_specific_loudness_left, axis=1) / 4
    )
    instantaneous_loudness_right = (
        np.sum(instantaneous_specific_loudness_right, axis=1) / 4
    )

    print(f"{filename_sound}\n\n")
    print(
        f"Calibration level:      {db_max} dB SPL (RMS level of a full-scale sinusoid)\n"
    )
    print(f"Field Type:  {field_type}\n\n")
    print(f"Maximum of long-term loudness:  {np.max(long_term_loudness):.2f} sone\n")
    print(
        f"                                {np.max(sone_to_phon_tv2018(long_term_loudness)):.2f} phon\n"
    )
    print(f"Maximum of short-term loudness: {np.max(short_term_loudness):.2f} sone\n")
    print(
        f"                                {np.max(sone_to_phon_tv2018(short_term_loudness)):.2f} phon\n"
    )
    print("Loudness over time:\n")
    print(
        f"                                {np.max(instantaneous_loudness_left):.2f} sone\n"
    )
    print(
        f"                                {np.max(instantaneous_loudness_right):.2f} sone\n"
    )

    return (
        loudness,
        short_term_loudness,
        long_term_loudness,
        instantaneous_loudness_left,
        instantaneous_loudness_right,
    )


if __name__ == "__main__":

    # tv2018(os.path.join(os.path.dirname(__file__), '../../../../tests/input/1k100ms.wav'), 50, 'free')
    # tv2018(
    #     "/Users/mitch/Downloads/ISO_532-3 3/code for tables/sounds Annex B/Testsignal14_propellerdrivenairplane.wav",
    #     100,
    # )
    tv2018('/Users/mitch/Downloads/ISO_532-3 3/code for tables/sounds Annex B/Testsignal16_hairdryer.wav', 100)

    # tv2018('/Users/mitch/Downloads/ISO_532-3 3/code for tables/sounds Annex B/Testsignal17_machinegun.wav', 100)
