"""
Acoustic feature extraction using praat-parselmouth.

Extracts jitter (5 variants), shimmer (6 variants), HNR, NHR, F0 stats,
and formants. Feature names match UCI Parkinsons dataset columns so the
same model can be used for training and live inference.
"""

import numpy as np
import parselmouth
from parselmouth.praat import call


def extract_acoustic_features(
    wav_path: str,
    f0_min: float = 75.0,
    f0_max: float = 500.0,
) -> dict:
    """
    Extract acoustic features from a WAV file using Praat via parselmouth.

    Args:
        wav_path: Path to a preprocessed WAV file (16kHz, mono).
        f0_min: Minimum expected F0 in Hz (75 for adult male, ~100 for female).
        f0_max: Maximum expected F0 in Hz.

    Returns:
        Dictionary of feature_name -> float. NaN for features that could
        not be computed (e.g. no voiced frames detected).
    """
    sound = parselmouth.Sound(wav_path)

    # --- Pitch / F0 ---
    pitch = call(sound, "To Pitch", 0.0, f0_min, f0_max)
    f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
    f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    f0_min_val = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    f0_max_val = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    # --- Point Process (needed for jitter and shimmer) ---
    point_process = call(
        sound, "To PointProcess (periodic, cc)", f0_min, f0_max
    )

    # --- Jitter (5 variants) ---
    jitter_local = call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_local_abs = call(
        point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_rap = call(
        point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_ppq5 = call(
        point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3
    )
    jitter_ddp = call(
        point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3
    )

    # --- Shimmer (6 variants) ---
    shimmer_local = call(
        [sound, point_process],
        "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
    )
    shimmer_local_dB = call(
        [sound, point_process],
        "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
    )
    shimmer_apq3 = call(
        [sound, point_process],
        "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
    )
    shimmer_apq5 = call(
        [sound, point_process],
        "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
    )
    shimmer_apq11 = call(
        [sound, point_process],
        "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
    )
    shimmer_dda = call(
        [sound, point_process],
        "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
    )

    # --- Harmonics-to-Noise Ratio ---
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    # NHR = inverse of HNR (noise-to-harmonics ratio)
    if hnr is not None and not np.isnan(hnr) and hnr > 0:
        nhr = 1.0 / (10 ** (hnr / 10))
    else:
        nhr = np.nan

    # --- Formants (F1-F4) ---
    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    f1_mean = call(formant, "Get mean", 1, 0, 0, "Hertz")
    f2_mean = call(formant, "Get mean", 2, 0, 0, "Hertz")
    f3_mean = call(formant, "Get mean", 3, 0, 0, "Hertz")
    f4_mean = call(formant, "Get mean", 4, 0, 0, "Hertz")

    return {
        # F0 statistics
        "MDVP:Fo(Hz)": f0_mean,
        "MDVP:Fhi(Hz)": f0_max_val,
        "MDVP:Flo(Hz)": f0_min_val,
        "f0_std": f0_std,
        # Jitter variants
        "MDVP:Jitter(%)": jitter_local,
        "MDVP:Jitter(Abs)": jitter_local_abs,
        "MDVP:RAP": jitter_rap,
        "MDVP:PPQ": jitter_ppq5,
        "Jitter:DDP": jitter_ddp,
        # Shimmer variants
        "MDVP:Shimmer": shimmer_local,
        "MDVP:Shimmer(dB)": shimmer_local_dB,
        "Shimmer:APQ3": shimmer_apq3,
        "Shimmer:APQ5": shimmer_apq5,
        "MDVP:APQ": shimmer_apq11,
        "Shimmer:DDA": shimmer_dda,
        # Noise ratios
        "NHR": nhr,
        "HNR": hnr,
        # Formants
        "F1_mean": f1_mean,
        "F2_mean": f2_mean,
        "F3_mean": f3_mean,
        "F4_mean": f4_mean,
    }
