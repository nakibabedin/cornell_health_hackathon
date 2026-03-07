"""
Model loading and inference pipeline.

Takes a WAV file path and returns PD probability, UPDRS estimate,
feature contributions, and audio quality info.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

import joblib

from ml.preprocess import load_and_preprocess, validate_audio
from ml.features_acoustic import extract_acoustic_features
from ml.features_spectral import extract_spectral_features
from ml.vectorize import CLS_FEATURE_NAMES, REG_FEATURE_NAMES, build_cls_vector, build_reg_vector


MODELS_DIR = Path(__file__).parent / "models"


class PDVoiceModel:
    """
    Wraps classification and regression models for inference.

    Usage:
        model = PDVoiceModel()
        result = model.predict("path/to/recording.wav")
    """

    def __init__(self, models_dir: Optional[str] = None):
        models_path = Path(models_dir) if models_dir else MODELS_DIR

        # Classification
        self.cls_model = joblib.load(models_path / "cls_xgb_model.joblib")
        self.cls_scaler = joblib.load(models_path / "cls_scaler.joblib")
        with open(models_path / "cls_feature_names.json") as f:
            self.cls_feature_names = json.load(f)
        with open(models_path / "cls_feature_importances.json") as f:
            self.cls_feature_importances = json.load(f)

        # Regression
        self.reg_model = joblib.load(models_path / "reg_xgb_model.joblib")
        self.reg_scaler = joblib.load(models_path / "reg_scaler.joblib")
        with open(models_path / "reg_feature_names.json") as f:
            self.reg_feature_names = json.load(f)
        with open(models_path / "reg_feature_importances.json") as f:
            self.reg_feature_importances = json.load(f)

    def predict(self, wav_path: str, explain: bool = False) -> dict:
        """
        Full inference: WAV → features → prediction.

        Args:
            wav_path: Path to a WAV file (already converted / preprocessed).
            explain: If True, include per-feature SHAP contributions (slower).

        Returns:
            dict with keys:
                pd_probability  (float 0-1)
                pd_label        ("positive" | "negative")
                updrs_estimate  (float >= 0)
                confidence      (float 0-1)
                features        (dict of all extracted features)
                audio_quality   (dict with valid, duration, rms, errors)
                feature_contributions  (dict, only if explain=True)
        """
        # 1. Validate audio
        y, sr = load_and_preprocess(wav_path)
        quality = validate_audio(y, sr, min_duration=2.0)

        # 2. Extract features
        acoustic = extract_acoustic_features(wav_path)
        spectral = extract_spectral_features(wav_path)
        all_features = {**acoustic, **spectral}

        # 3. Classification
        cls_vec = build_cls_vector(all_features)
        cls_vec_scaled = self.cls_scaler.transform(cls_vec.reshape(1, -1))
        pd_prob = float(self.cls_model.predict_proba(cls_vec_scaled)[0, 1])
        pd_label = "positive" if pd_prob >= 0.5 else "negative"

        # 4. Regression (UPDRS)
        reg_vec = build_reg_vector(all_features)
        reg_vec_scaled = self.reg_scaler.transform(reg_vec.reshape(1, -1))
        updrs = float(self.reg_model.predict(reg_vec_scaled)[0])
        updrs = max(0.0, updrs)

        # 5. Confidence based on audio quality + NaN count
        nan_count = sum(
            1 for v in all_features.values()
            if v is None or (isinstance(v, float) and np.isnan(v))
        )
        feature_conf = max(0.0, 1.0 - (nan_count / max(len(all_features), 1)))
        quality_conf = 1.0 if quality["valid"] else 0.5
        confidence = round(feature_conf * quality_conf, 3)

        result = {
            "pd_probability": round(pd_prob, 4),
            "pd_label": pd_label,
            "updrs_estimate": round(updrs, 2),
            "confidence": confidence,
            "features": all_features,
            "audio_quality": quality,
        }

        # 6. Optional SHAP explanations
        if explain:
            result["feature_contributions"] = self._explain(
                cls_vec_scaled[0], all_features
            )

        return result

    def _explain(self, scaled_vector: np.ndarray, all_features: dict) -> dict:
        """Compute per-feature SHAP contributions for the classification."""
        import shap

        explainer = shap.TreeExplainer(self.cls_model)
        shap_vals = explainer.shap_values(scaled_vector.reshape(1, -1))[0]

        contributions = {}
        for i, name in enumerate(self.cls_feature_names):
            raw_val = all_features.get(name, 0.0)
            if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                raw_val = 0.0
            contributions[name] = {
                "value": float(raw_val),
                "shap_value": float(shap_vals[i]),
                "direction": (
                    "increases_pd_risk" if shap_vals[i] > 0 else "decreases_pd_risk"
                ),
            }
        return contributions
