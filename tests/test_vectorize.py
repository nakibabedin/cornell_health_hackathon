"""Tests for feature vector assembly and column mapping."""

import numpy as np

from ml.vectorize import (
    CLS_FEATURE_NAMES,
    REG_FEATURE_NAMES,
    build_cls_vector,
    build_reg_vector,
)


class TestConstants:
    def test_cls_feature_count(self):
        assert len(CLS_FEATURE_NAMES) == 16

    def test_reg_feature_count(self):
        assert len(REG_FEATURE_NAMES) == 13


class TestBuildClsVector:
    def test_correct_length(self):
        features = {name: 1.0 for name in CLS_FEATURE_NAMES}
        vec = build_cls_vector(features)
        assert vec.shape == (16,)

    def test_missing_features_filled(self):
        vec = build_cls_vector({}, fill_value=-1.0)
        assert vec.shape == (16,)
        assert all(v == -1.0 for v in vec)

    def test_nan_replaced_with_fill(self):
        features = {"MDVP:Fo(Hz)": float("nan"), "HNR": 25.0}
        vec = build_cls_vector(features, fill_value=0.0)
        assert vec[0] == 0.0  # MDVP:Fo(Hz) is first
        assert vec[-1] == 25.0  # HNR is last

    def test_preserves_order(self):
        features = {name: float(i) for i, name in enumerate(CLS_FEATURE_NAMES)}
        vec = build_cls_vector(features)
        for i, name in enumerate(CLS_FEATURE_NAMES):
            assert vec[i] == float(i)


class TestBuildRegVector:
    def test_correct_length(self):
        # Provide features using our extracted names (MDVP: prefixed)
        features = {
            "MDVP:Jitter(%)": 0.01, "MDVP:Jitter(Abs)": 0.0001,
            "MDVP:RAP": 0.005, "MDVP:PPQ": 0.005, "Jitter:DDP": 0.01,
            "MDVP:Shimmer": 0.03, "MDVP:Shimmer(dB)": 0.3,
            "Shimmer:APQ3": 0.02, "Shimmer:APQ5": 0.03,
            "MDVP:APQ": 0.04, "Shimmer:DDA": 0.05,
            "NHR": 0.02, "HNR": 22.0,
        }
        vec = build_reg_vector(features)
        assert vec.shape == (13,)

    def test_maps_names_correctly(self):
        # MDVP:Jitter(%) should map to Jitter(%) (first in REG_FEATURE_NAMES)
        features = {"MDVP:Jitter(%)": 0.42}
        vec = build_reg_vector(features, fill_value=0.0)
        assert vec[0] == 0.42  # Jitter(%) is first in REG list
