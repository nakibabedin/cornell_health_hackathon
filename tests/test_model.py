"""Tests for model loading and artifact integrity."""

import json
from pathlib import Path

import joblib
import numpy as np


MODELS_DIR = Path(__file__).parent.parent / "backend" / "models"


class TestModelArtifacts:
    """Verify all model artifacts were saved correctly."""

    def test_cls_model_exists(self):
        assert (MODELS_DIR / "cls_xgb_model.joblib").exists()

    def test_reg_model_exists(self):
        assert (MODELS_DIR / "reg_xgb_model.joblib").exists()

    def test_cls_scaler_exists(self):
        assert (MODELS_DIR / "cls_scaler.joblib").exists()

    def test_reg_scaler_exists(self):
        assert (MODELS_DIR / "reg_scaler.joblib").exists()

    def test_cls_feature_names(self):
        with open(MODELS_DIR / "cls_feature_names.json") as f:
            names = json.load(f)
        assert len(names) == 16
        assert "HNR" in names

    def test_reg_feature_names(self):
        with open(MODELS_DIR / "reg_feature_names.json") as f:
            names = json.load(f)
        assert len(names) == 13
        assert "HNR" in names

    def test_cls_importances_sum_to_one(self):
        with open(MODELS_DIR / "cls_feature_importances.json") as f:
            weights = json.load(f)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_reg_importances_sum_to_one(self):
        with open(MODELS_DIR / "reg_feature_importances.json") as f:
            weights = json.load(f)
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestClassificationModel:
    """Test that the classification model produces valid outputs."""

    def test_predict_proba_shape(self):
        model = joblib.load(MODELS_DIR / "cls_xgb_model.joblib")
        scaler = joblib.load(MODELS_DIR / "cls_scaler.joblib")
        # Create a dummy input (16 features)
        dummy = np.zeros((1, 16))
        dummy_scaled = scaler.transform(dummy)
        probs = model.predict_proba(dummy_scaled)
        assert probs.shape == (1, 2)
        assert 0.0 <= probs[0, 1] <= 1.0

    def test_predict_binary(self):
        model = joblib.load(MODELS_DIR / "cls_xgb_model.joblib")
        scaler = joblib.load(MODELS_DIR / "cls_scaler.joblib")
        dummy = np.zeros((1, 16))
        pred = model.predict(scaler.transform(dummy))
        assert pred[0] in (0, 1)


class TestRegressionModel:
    """Test that the regression model produces valid outputs."""

    def test_predict_returns_scalar(self):
        model = joblib.load(MODELS_DIR / "reg_xgb_model.joblib")
        scaler = joblib.load(MODELS_DIR / "reg_scaler.joblib")
        dummy = np.zeros((1, 13))
        pred = model.predict(scaler.transform(dummy))
        assert pred.shape == (1,)
        # UPDRS should be a reasonable number (not NaN/inf)
        assert np.isfinite(pred[0])
