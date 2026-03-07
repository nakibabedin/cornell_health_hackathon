"""
Spectral feature extraction using librosa.

Extracts MFCCs (13 coefficients + deltas + delta-deltas), spectral
centroid/bandwidth/rolloff, zero-crossing rate, and RMS energy.
"""

import numpy as np
import librosa


def extract_spectral_features(
    wav_path: str,
    sr: int = 16000,
    n_mfcc: int = 13,
) -> dict:
    """
    Extract spectral features from a WAV file using librosa.

    Returns ~62 features:
        - 13 MFCCs (mean + std = 26)
        - 13 delta MFCCs (mean = 13)
        - 13 delta-delta MFCCs (mean = 13)
        - Spectral centroid (mean, std)
        - Spectral bandwidth (mean, std)
        - Spectral rolloff (mean, std)
        - Zero crossing rate (mean, std)
        - RMS energy (mean, std)
    """
    y, sr = librosa.load(wav_path, sr=sr, mono=True)

    features = {}

    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))

    # --- Delta MFCCs ---
    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(n_mfcc):
        features[f"delta_mfcc_{i+1}_mean"] = float(np.mean(delta_mfccs[i]))

    # --- Delta-delta MFCCs ---
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    for i in range(n_mfcc):
        features[f"delta2_mfcc_{i+1}_mean"] = float(np.mean(delta2_mfccs[i]))

    # --- Spectral Centroid ---
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(spec_cent))
    features["spectral_centroid_std"] = float(np.std(spec_cent))

    # --- Spectral Bandwidth ---
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(spec_bw))
    features["spectral_bandwidth_std"] = float(np.std(spec_bw))

    # --- Spectral Rolloff ---
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = float(np.mean(spec_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spec_rolloff))

    # --- Zero Crossing Rate ---
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # --- RMS Energy ---
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    return features
