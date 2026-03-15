"""
ReasonForge - Security Tests

Tests for input sanitization, rate guarding, and anomaly detection.
"""

import base64

import pytest

from reasonforge.security.anomaly import AnomalyDetector
from reasonforge.security.rate_guard import RateGuard
from reasonforge.security.sanitizer import InputSanitizer


class TestInputSanitizer:
    """Test input sanitization."""

    def test_sanitize_oversized_steps(self):
        class FakeResponse:
            reasoning_steps = [{"reasoning": "x" * 20_000, "confidence": 0.5}] * 60
            final_answer = "answer"
            proof_artifact = None
            code_artifact = None

        resp = FakeResponse()
        InputSanitizer.sanitize_submission(resp)
        assert len(resp.reasoning_steps) <= 50
        assert len(resp.reasoning_steps[0]["reasoning"]) <= 10_000

    def test_sanitize_confidence_range(self):
        class FakeResponse:
            reasoning_steps = [
                {"reasoning": "test", "confidence": 5.0},
                {"reasoning": "test", "confidence": -1.0},
            ]
            final_answer = "answer"
            proof_artifact = None
            code_artifact = None

        resp = FakeResponse()
        InputSanitizer.sanitize_submission(resp)
        assert resp.reasoning_steps[0]["confidence"] == 1.0
        assert resp.reasoning_steps[1]["confidence"] == 0.0

    def test_sanitize_oversized_answer(self):
        class FakeResponse:
            reasoning_steps = []
            final_answer = "x" * 100_000
            proof_artifact = None
            code_artifact = None

        resp = FakeResponse()
        InputSanitizer.sanitize_submission(resp)
        assert len(resp.final_answer) <= 50_000

    def test_sanitize_oversized_proof(self):
        big_proof = base64.b64encode(b"x" * 2_000_000).decode()

        class FakeResponse:
            reasoning_steps = []
            final_answer = "answer"
            proof_artifact = big_proof
            code_artifact = None

        resp = FakeResponse()
        InputSanitizer.sanitize_submission(resp)
        assert resp.proof_artifact is None

    def test_sanitize_problem_xss(self):
        problem = '<script>alert("xss")</script>Real problem here.'
        sanitized = InputSanitizer.sanitize_problem(problem)
        assert "<script>" not in sanitized
        assert "Real problem here." in sanitized

    def test_sanitize_problem_html(self):
        problem = "<b>Bold</b> and <i>italic</i> problem"
        sanitized = InputSanitizer.sanitize_problem(problem)
        assert "<b>" not in sanitized
        assert "Bold" in sanitized

    def test_validate_domain(self):
        assert InputSanitizer.validate_domain("mathematics") is True
        assert InputSanitizer.validate_domain("code") is True
        assert InputSanitizer.validate_domain("invalid") is False

    def test_validate_difficulty(self):
        assert InputSanitizer.validate_difficulty(1) is True
        assert InputSanitizer.validate_difficulty(10) is True
        assert InputSanitizer.validate_difficulty(0) is False
        assert InputSanitizer.validate_difficulty(11) is False


class TestRateGuard:
    """Test per-UID rate limiting."""

    def test_allow_within_limit(self):
        guard = RateGuard(max_requests_per_minute=5)
        for _ in range(5):
            assert guard.check(1) is True

    def test_block_over_limit(self):
        guard = RateGuard(max_requests_per_minute=3)
        for _ in range(3):
            guard.check(1)
        assert guard.check(1) is False

    def test_different_uids(self):
        guard = RateGuard(max_requests_per_minute=1)
        assert guard.check(1) is True
        assert guard.check(2) is True  # Different UID

    def test_remaining(self):
        guard = RateGuard(max_requests_per_minute=5)
        guard.check(1)
        guard.check(1)
        assert guard.get_remaining(1) == 3

    def test_reset(self):
        guard = RateGuard(max_requests_per_minute=1)
        guard.check(1)
        assert guard.check(1) is False
        guard.reset(1)
        assert guard.check(1) is True


class TestAnomalyDetector:
    """Test anomaly detection."""

    @pytest.fixture
    def detector(self):
        return AnomalyDetector()

    def test_timing_anomaly_fast(self, detector):
        # Difficulty 5, solved in 100ms -> anomalous
        assert detector.check_timing_anomaly(100, 5) is True

    def test_timing_anomaly_normal(self, detector):
        # Difficulty 5, solved in 10000ms -> normal
        assert detector.check_timing_anomaly(10000, 5) is False

    def test_score_manipulation_consistent(self, detector):
        # Nearly identical scores -> suspicious
        scores = [0.800, 0.801, 0.800, 0.801, 0.800]
        assert detector.check_score_manipulation(scores) is True

    def test_score_manipulation_normal(self, detector):
        # Normal variance -> not suspicious
        scores = [0.8, 0.6, 0.9, 0.7, 0.5]
        assert detector.check_score_manipulation(scores) is False

    def test_score_manipulation_too_few(self, detector):
        scores = [0.8, 0.8]
        assert detector.check_score_manipulation(scores) is False

    def test_sudden_improvement(self, detector):
        assert detector.check_sudden_improvement([0.9, 0.95], 0.3) is True
        assert detector.check_sudden_improvement([0.35, 0.4], 0.3) is False

    def test_collusion_detection(self, detector):
        subs = [
            {"uid": 0, "text": "the quick brown fox jumps over the lazy dog"},
            {"uid": 1, "text": "the quick brown fox jumps over the lazy dog"},
        ]
        flagged = detector.check_collusion(subs)
        assert len(flagged) == 1
        assert flagged[0][2] > 0.9  # High similarity

    def test_no_collusion(self, detector):
        subs = [
            {"uid": 0, "text": "mathematical proof using induction on natural numbers"},
            {"uid": 1, "text": "computational approach with dynamic programming algorithm"},
        ]
        flagged = detector.check_collusion(subs)
        assert len(flagged) == 0

    def test_anomaly_report(self, detector):
        report = detector.get_anomaly_report(
            uid=1,
            time_ms=100,
            difficulty=5,
            cms_history=[0.8, 0.801, 0.800, 0.801, 0.800],
        )
        assert report["uid"] == 1
        assert report["timing_anomaly"] is True
        assert report["score_manipulation"] is True
        assert report["flags_count"] == 2
