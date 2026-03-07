"""
Progression scoring: personal baseline comparison + PD Voice Index.

Pipeline:
    1. First 3 sessions establish a personal baseline (mean per feature)
    2. Each new session: compute % deviation from baseline per feature
    3. PD Voice Index (0-100): weighted average of deviations
       (weights from SHAP feature importance)
    4. Trend detection: 3+ consecutive declining sessions = "worsening"
"""

import numpy as np


# Features where INCREASE indicates worsening
HIGHER_IS_WORSE = {
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR",
    # Telemonitoring column names (same features, different naming)
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ11",
}

# Features where DECREASE indicates worsening
LOWER_IS_WORSE = {
    "HNR",
    "f0_std",  # Reduced F0 variability = monotone speech = PD marker
}

BASELINE_SESSION_COUNT = 3
WORSENING_CONSECUTIVE = 3
WORSENING_THRESHOLD = 3.0  # minimum score-point change per session to count


def compute_baseline(
    session_features: list[dict],
    max_sessions: int = BASELINE_SESSION_COUNT,
) -> dict:
    """
    Compute personal baseline from the first N sessions.

    Returns dict of feature_name -> {"mean": float, "std": float}.
    """
    baseline_sessions = session_features[:max_sessions]
    if not baseline_sessions:
        return {}

    all_names: set[str] = set()
    for s in baseline_sessions:
        all_names.update(s.keys())

    baseline = {}
    for name in all_names:
        values = [
            s[name] for s in baseline_sessions
            if name in s and s[name] is not None and not np.isnan(s[name])
        ]
        if values:
            baseline[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)) if len(values) > 1 else 0.0,
            }

    return baseline


def compute_deviation(current_features: dict, baseline: dict) -> dict:
    """
    Compute per-feature % deviation from personal baseline.

    Sign is normalized so positive always means WORSENING.
    """
    deviations = {}

    for name, bl in baseline.items():
        if name not in current_features:
            continue
        val = current_features[name]
        if val is None or np.isnan(val) or bl["mean"] == 0:
            continue

        raw_deviation = (val - bl["mean"]) / abs(bl["mean"])

        # Normalize sign: positive = worsening
        if name in HIGHER_IS_WORSE:
            signed = raw_deviation
        elif name in LOWER_IS_WORSE:
            signed = -raw_deviation
        else:
            signed = abs(raw_deviation)

        deviations[name] = {
            "raw_value": float(val),
            "baseline_mean": float(bl["mean"]),
            "percent_deviation": float(signed),
            "abs_deviation": float(abs(signed)),
        }

    return deviations


def compute_pd_voice_index(
    deviations: dict,
    feature_weights: dict,
    pd_probability: float = 0.5,
    updrs_estimate: float = 20.0,
) -> float:
    """
    Compute composite PD Voice Index (0-100).

    Combines:
        - Weighted deviation from baseline (40%) — if baseline exists
        - PD probability from classification model (30%)
        - Normalized UPDRS estimate (30%)

    Falls back to 50/50 model-only when no baseline exists.
    """
    # Component 1: Weighted deviation score (0-100)
    if deviations:
        weighted_sum = 0.0
        total_weight = 0.0
        for name, dev in deviations.items():
            weight = feature_weights.get(name, 0.01)
            # Clamp to [-2, +2] to avoid outlier domination
            clamped = max(-2.0, min(2.0, dev["percent_deviation"]))
            # Map [-2, +2] → [0, 1] where 0 = improved, 1 = severely worsened
            normalized = (clamped + 2.0) / 4.0
            weighted_sum += normalized * weight
            total_weight += weight

        deviation_score = (weighted_sum / total_weight * 100) if total_weight > 0 else 50.0
    else:
        deviation_score = 50.0

    # Component 2: PD probability (0-100)
    pd_score = pd_probability * 100

    # Component 3: UPDRS normalized (typical range 0-176, 80 ≈ moderate)
    updrs_score = min(100.0, (updrs_estimate / 80.0) * 100)

    # Weighted combination
    if deviations:
        composite = 0.40 * deviation_score + 0.30 * pd_score + 0.30 * updrs_score
    else:
        composite = 0.50 * pd_score + 0.50 * updrs_score

    return round(max(0.0, min(100.0, composite)), 1)


def detect_trend(
    session_scores: list[float],
    threshold: float = WORSENING_THRESHOLD,
    consecutive: int = WORSENING_CONSECUTIVE,
) -> str:
    """
    Detect trend from a series of PD Voice Index scores.

    Returns one of: "improving", "stable", "worsening", "insufficient_data".
    """
    if len(session_scores) < 3:
        return "insufficient_data"

    changes = [
        session_scores[i] - session_scores[i - 1]
        for i in range(1, len(session_scores))
    ]

    # Count consecutive worsening at the tail
    consecutive_worse = 0
    for change in reversed(changes):
        if change > threshold:
            consecutive_worse += 1
        else:
            break

    # Count consecutive improving at the tail
    consecutive_better = 0
    for change in reversed(changes):
        if change < -threshold:
            consecutive_better += 1
        else:
            break

    if consecutive_worse >= consecutive:
        return "worsening"
    elif consecutive_better >= consecutive:
        return "improving"
    else:
        return "stable"


def score_session(
    current_features: dict,
    past_sessions: list[dict],
    past_scores: list[float],
    feature_weights: dict,
    pd_probability: float,
    updrs_estimate: float,
) -> dict:
    """
    Top-level scoring function for a single session.

    Args:
        current_features: Feature dict for this session.
        past_sessions: Feature dicts from all prior sessions (ordered by time).
        past_scores: PD Voice Index values from prior sessions.
        feature_weights: SHAP importance weights from model training.
        pd_probability: Model's PD probability for this session.
        updrs_estimate: Model's UPDRS estimate for this session.

    Returns:
        dict with: score, trend, label, baseline_established,
                   deviations, top_changed_features.
    """
    baseline = compute_baseline(past_sessions, max_sessions=BASELINE_SESSION_COUNT)
    baseline_established = len(past_sessions) >= BASELINE_SESSION_COUNT

    deviations = compute_deviation(current_features, baseline) if baseline else {}

    score = compute_pd_voice_index(
        deviations, feature_weights, pd_probability, updrs_estimate
    )

    all_scores = past_scores + [score]
    trend = detect_trend(all_scores)

    if score < 30:
        label = "low_concern"
    elif score < 60:
        label = "moderate_concern"
    else:
        label = "high_concern"

    # Top 5 features with largest deviation
    top_features = sorted(
        deviations.items(),
        key=lambda x: abs(x[1]["percent_deviation"]),
        reverse=True,
    )[:5]
    top_changed = [
        {
            "feature": name,
            "deviation_pct": round(dev["percent_deviation"] * 100, 1),
            "direction": "worse" if dev["percent_deviation"] > 0 else "better",
            "current": dev["raw_value"],
            "baseline": dev["baseline_mean"],
        }
        for name, dev in top_features
    ]

    return {
        "score": score,
        "trend": trend,
        "label": label,
        "baseline_established": baseline_established,
        "deviations": deviations,
        "top_changed_features": top_changed,
    }
