"""
Combine acoustic + spectral features into ordered feature vectors.

Handles the column name mapping between live-extracted features and the
two UCI dataset schemas (Classification uses MDVP: prefixes, Telemonitoring
does not).
"""

import numpy as np

from ml.features_acoustic import extract_acoustic_features
from ml.features_spectral import extract_spectral_features


# The 16 features from UCI Classification that we can extract from audio.
# Excludes: RPDE, DFA, spread1, spread2, D2, PPE (nonlinear dynamics).
CLS_FEATURE_NAMES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
]

# The 13 features from UCI Telemonitoring that we can extract from audio.
# Excludes: RPDE, DFA, PPE.
REG_FEATURE_NAMES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR",
]

# Maps our extracted feature names → UCI Telemonitoring column names.
# (Our acoustic extractor uses UCI Classification names as keys.)
_LIVE_TO_TEL = {
    "MDVP:Jitter(%)": "Jitter(%)",
    "MDVP:Jitter(Abs)": "Jitter(Abs)",
    "MDVP:RAP": "Jitter:RAP",
    "MDVP:PPQ": "Jitter:PPQ5",
    "Jitter:DDP": "Jitter:DDP",
    "MDVP:Shimmer": "Shimmer",
    "MDVP:Shimmer(dB)": "Shimmer(dB)",
    "Shimmer:APQ3": "Shimmer:APQ3",
    "Shimmer:APQ5": "Shimmer:APQ5",
    "MDVP:APQ": "Shimmer:APQ11",
    "Shimmer:DDA": "Shimmer:DDA",
    "NHR": "NHR",
    "HNR": "HNR",
}

# Reverse: telemonitoring name → our extracted name
_TEL_TO_LIVE = {v: k for k, v in _LIVE_TO_TEL.items()}


def extract_all_features(wav_path: str) -> dict:
    """Extract all features from a WAV file (acoustic + spectral)."""
    acoustic = extract_acoustic_features(wav_path)
    spectral = extract_spectral_features(wav_path)
    return {**acoustic, **spectral}


def build_cls_vector(features: dict, fill_value: float = 0.0) -> np.ndarray:
    """
    Build feature vector aligned with classification model (16 features).

    Our acoustic extractor already uses UCI Classification names as keys,
    so this is a direct lookup.
    """
    vector = []
    for name in CLS_FEATURE_NAMES:
        val = features.get(name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            vector.append(fill_value)
        else:
            vector.append(float(val))
    return np.array(vector, dtype=np.float64)


def build_reg_vector(features: dict, fill_value: float = 0.0) -> np.ndarray:
    """
    Build feature vector aligned with regression model (13 telemonitoring features).

    Maps from our extracted names (MDVP: prefixed) to telemonitoring names.
    """
    vector = []
    for tel_name in REG_FEATURE_NAMES:
        our_name = _TEL_TO_LIVE.get(tel_name, tel_name)
        val = features.get(our_name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            vector.append(fill_value)
        else:
            vector.append(float(val))
    return np.array(vector, dtype=np.float64)
