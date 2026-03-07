"""Tests for progression scoring logic."""

from ml.scoring import (
    compute_baseline,
    compute_deviation,
    compute_pd_voice_index,
    detect_trend,
    score_session,
)


class TestBaseline:
    def test_baseline_from_three_sessions(self):
        sessions = [
            {"HNR": 25.0, "MDVP:Jitter(%)": 0.004},
            {"HNR": 24.0, "MDVP:Jitter(%)": 0.005},
            {"HNR": 26.0, "MDVP:Jitter(%)": 0.003},
        ]
        baseline = compute_baseline(sessions)
        assert abs(baseline["HNR"]["mean"] - 25.0) < 0.01
        assert abs(baseline["MDVP:Jitter(%)"]["mean"] - 0.004) < 0.001

    def test_empty_sessions(self):
        assert compute_baseline([]) == {}

    def test_uses_only_first_n_sessions(self):
        sessions = [
            {"HNR": 25.0},
            {"HNR": 24.0},
            {"HNR": 26.0},
            {"HNR": 10.0},  # should be ignored (4th session)
        ]
        baseline = compute_baseline(sessions, max_sessions=3)
        assert abs(baseline["HNR"]["mean"] - 25.0) < 0.01


class TestDeviation:
    def test_worsening_jitter(self):
        baseline = {"MDVP:Jitter(%)": {"mean": 0.004, "std": 0.001}}
        current = {"MDVP:Jitter(%)": 0.008}
        dev = compute_deviation(current, baseline)
        # Jitter increase = worsening → positive deviation
        assert dev["MDVP:Jitter(%)"]["percent_deviation"] > 0

    def test_worsening_hnr(self):
        baseline = {"HNR": {"mean": 25.0, "std": 1.0}}
        current = {"HNR": 20.0}
        dev = compute_deviation(current, baseline)
        # HNR decrease = worsening → positive deviation (flipped)
        assert dev["HNR"]["percent_deviation"] > 0

    def test_improving_jitter(self):
        baseline = {"MDVP:Jitter(%)": {"mean": 0.008, "std": 0.001}}
        current = {"MDVP:Jitter(%)": 0.004}
        dev = compute_deviation(current, baseline)
        # Jitter decrease = improving → negative deviation
        assert dev["MDVP:Jitter(%)"]["percent_deviation"] < 0

    def test_missing_feature_skipped(self):
        baseline = {"HNR": {"mean": 25.0, "std": 1.0}}
        current = {"something_else": 10.0}
        dev = compute_deviation(current, baseline)
        assert len(dev) == 0


class TestTrend:
    def test_worsening_trend(self):
        scores = [30, 35, 42, 50, 58]
        assert detect_trend(scores) == "worsening"

    def test_improving_trend(self):
        scores = [58, 50, 42, 35, 30]
        assert detect_trend(scores) == "improving"

    def test_stable_trend(self):
        scores = [35, 36, 34, 37, 35]
        assert detect_trend(scores) == "stable"

    def test_insufficient_data(self):
        assert detect_trend([35, 36]) == "insufficient_data"

    def test_single_score(self):
        assert detect_trend([35]) == "insufficient_data"


class TestPDVoiceIndex:
    def test_score_in_valid_range(self):
        score = compute_pd_voice_index(
            deviations={},
            feature_weights={},
            pd_probability=0.7,
            updrs_estimate=40.0,
        )
        assert 0.0 <= score <= 100.0

    def test_high_probability_gives_high_score(self):
        high = compute_pd_voice_index({}, {}, pd_probability=0.95, updrs_estimate=60.0)
        low = compute_pd_voice_index({}, {}, pd_probability=0.1, updrs_estimate=10.0)
        assert high > low

    def test_with_deviations(self):
        deviations = {
            "MDVP:Jitter(%)": {
                "raw_value": 0.008,
                "baseline_mean": 0.004,
                "percent_deviation": 1.0,  # 100% worse
                "abs_deviation": 1.0,
            }
        }
        weights = {"MDVP:Jitter(%)": 1.0}
        score = compute_pd_voice_index(
            deviations, weights, pd_probability=0.5, updrs_estimate=20.0
        )
        assert 0.0 <= score <= 100.0


class TestScoreSession:
    def test_first_session_no_baseline(self):
        result = score_session(
            current_features={"HNR": 25.0, "MDVP:Jitter(%)": 0.004},
            past_sessions=[],
            past_scores=[],
            feature_weights={"HNR": 0.5, "MDVP:Jitter(%)": 0.5},
            pd_probability=0.3,
            updrs_estimate=15.0,
        )
        assert not result["baseline_established"]
        assert 0.0 <= result["score"] <= 100.0
        assert result["trend"] == "insufficient_data"
        assert result["label"] in ("low_concern", "moderate_concern", "high_concern")

    def test_with_established_baseline(self):
        past = [
            {"HNR": 25.0, "MDVP:Jitter(%)": 0.004},
            {"HNR": 24.5, "MDVP:Jitter(%)": 0.0042},
            {"HNR": 25.5, "MDVP:Jitter(%)": 0.0038},
        ]
        result = score_session(
            current_features={"HNR": 20.0, "MDVP:Jitter(%)": 0.008},
            past_sessions=past,
            past_scores=[35.0, 36.0, 34.0],
            feature_weights={"HNR": 0.5, "MDVP:Jitter(%)": 0.5},
            pd_probability=0.6,
            updrs_estimate=30.0,
        )
        assert result["baseline_established"]
        assert len(result["top_changed_features"]) <= 5
