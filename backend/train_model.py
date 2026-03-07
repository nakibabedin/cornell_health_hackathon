"""
Train XGBoost models for PD detection and UPDRS regression.

Usage:
    python -m backend.train_model

Outputs (in backend/models/):
    cls_xgb_model.joblib          Binary classification model
    reg_xgb_model.joblib          UPDRS regression model
    cls_scaler.joblib             StandardScaler for classification
    reg_scaler.joblib             StandardScaler for regression
    cls_feature_names.json        Feature column order for classification
    reg_feature_names.json        Feature column order for regression
    cls_feature_importances.json  SHAP-based importance weights (for scoring)
    reg_feature_importances.json  SHAP-based importance weights
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
import shap
import joblib


MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# ── Features extractable from audio (no nonlinear dynamics) ─────────────

CLS_EXTRACTABLE = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
]

REG_EXTRACTABLE = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR",
]


# ── Data loading ────────────────────────────────────────────────────────

def load_classification_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load UCI Parkinsons Classification dataset (id=174).

    Returns (X, y, groups) where groups are subject IDs extracted
    from the ``name`` column (format ``phon_R01_S01_1`` → ``S01``).
    """
    cached = DATA_DIR / "parkinsons_classification.csv"
    if cached.exists():
        raw = pd.read_csv(cached)
    else:
        # Try ucimlrepo first, fall back to direct URL
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=174)
            # ucimlrepo may separate identifiers; rebuild from original URL
            raw = pd.read_csv(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "parkinsons/parkinsons.data"
            )
        except Exception:
            raw = pd.read_csv(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "parkinsons/parkinsons.data"
            )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cached, index=False)

    groups = raw["name"].str.extract(r"(S\d+)")[0].values
    y = raw["status"].values
    X = raw.drop(columns=["name", "status"])
    return X, y, groups


def load_telemonitoring_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load UCI Parkinsons Telemonitoring dataset (id=189).

    Returns (X_voice_features, motor_updrs, total_updrs, groups).
    """
    cached = DATA_DIR / "parkinsons_telemonitoring.csv"
    if cached.exists():
        raw = pd.read_csv(cached)
    else:
        try:
            raw = pd.read_csv(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "parkinsons/telemonitoring/parkinsons_updrs.data"
            )
        except Exception:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=189)
            raw = pd.concat(
                [dataset.data.features, dataset.data.targets], axis=1
            )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cached, index=False)

    groups = raw["subject#"].values
    motor_updrs = raw["motor_UPDRS"].values
    total_updrs = raw["total_UPDRS"].values
    X = raw.drop(
        columns=["subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"]
    )
    return X, motor_updrs, total_updrs, groups


# ── Training ────────────────────────────────────────────────────────────

def _compute_shap_weights(
    model, X_scaled: np.ndarray, feature_names: list[str]
) -> dict:
    """Compute normalized SHAP importance weights (sum to 1)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    raw = {name: float(imp) for name, imp in zip(feature_names, mean_abs)}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()} if total > 0 else raw


def train_classification_model():
    """
    Train binary PD classification on UCI Parkinsons (16 extractable features).

    Uses GroupKFold(5) so no subject appears in both train and test.
    """
    print("=" * 60)
    print("TRAINING CLASSIFICATION MODEL")
    print("=" * 60)

    X_full, y, groups = load_classification_data()

    # Use only features we can extract from live audio
    X = X_full[CLS_EXTRACTABLE].copy()
    feature_names = list(X.columns)

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  PD: {np.sum(y == 1)}, Healthy: {np.sum(y == 0)}")
    print(f"  Subjects: {len(np.unique(groups))}")

    # Handle class imbalance via scale_pos_weight
    neg, pos = np.sum(y == 0), np.sum(y == 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42,
    )

    gkf = GroupKFold(n_splits=5)
    all_true, all_pred, all_prob = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Verify no subject leakage
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)
        probs = model.predict_proba(X_te_s)[:, 1]

        all_true.extend(y_te)
        all_pred.extend(preds)
        all_prob.extend(probs)

        print(f"  Fold {fold + 1}: acc={accuracy_score(y_te, preds):.3f}")

    acc = accuracy_score(all_true, all_pred)
    auc = roc_auc_score(all_true, all_prob)
    print(f"\n  GroupKFold CV — Accuracy: {acc:.3f}, AUC: {auc:.3f}")
    print(classification_report(all_true, all_pred, target_names=["Healthy", "PD"]))

    # Train final model on ALL data
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(X)
    model.fit(X_all, y)

    # SHAP importance weights
    weights = _compute_shap_weights(model, X_all, feature_names)
    print("  Feature importances (SHAP):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {name}: {w:.4f}")

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "cls_xgb_model.joblib")
    joblib.dump(scaler_final, MODELS_DIR / "cls_scaler.joblib")
    with open(MODELS_DIR / "cls_feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open(MODELS_DIR / "cls_feature_importances.json", "w") as f:
        json.dump(weights, f)

    print(f"\n  Saved to {MODELS_DIR}/cls_xgb_model.joblib")
    return model, scaler_final, feature_names, weights


def train_regression_model():
    """
    Train UPDRS severity regression on UCI Telemonitoring (13 extractable features).

    Uses GroupKFold(5) grouped by subject#.
    """
    print("\n" + "=" * 60)
    print("TRAINING REGRESSION MODEL (UPDRS)")
    print("=" * 60)

    X_full, motor_updrs, total_updrs, groups = load_telemonitoring_data()

    X = X_full[REG_EXTRACTABLE].copy()
    feature_names = list(X.columns)
    y = total_updrs  # primary target

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  UPDRS range: [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.1f}")
    print(f"  Subjects: {len(np.unique(groups))}")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )

    gkf = GroupKFold(n_splits=5)
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)

        all_true.extend(y_te)
        all_pred.extend(preds)

        fold_mae = mean_absolute_error(y_te, preds)
        print(f"  Fold {fold + 1}: MAE={fold_mae:.2f}")

    mae = mean_absolute_error(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    r2 = r2_score(all_true, all_pred)
    print(f"\n  GroupKFold CV — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

    # Train final model on ALL data
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(X)
    model.fit(X_all, y)

    weights = _compute_shap_weights(model, X_all, feature_names)

    # Save
    joblib.dump(model, MODELS_DIR / "reg_xgb_model.joblib")
    joblib.dump(scaler_final, MODELS_DIR / "reg_scaler.joblib")
    with open(MODELS_DIR / "reg_feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open(MODELS_DIR / "reg_feature_importances.json", "w") as f:
        json.dump(weights, f)

    print(f"\n  Saved to {MODELS_DIR}/reg_xgb_model.joblib")
    return model, scaler_final, feature_names, weights


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_classification_model()
    train_regression_model()
    print("\nAll models trained and saved.")
