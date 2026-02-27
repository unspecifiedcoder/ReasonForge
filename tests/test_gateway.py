"""
ReasonForge - Gateway Tests

Tests for API gateway endpoints, authentication, and rate limiting.
"""

import pytest
from reasonforge.gateway.auth import APIKeyManager, TIER_LIMITS
from reasonforge.gateway.rate_limiter import TokenBucketRateLimiter, PerIPRateLimiter
from reasonforge.gateway.billing import BillingTracker
from reasonforge.gateway.schemas import TaskSubmissionRequest, HealthResponse


class TestAPIKeyManager:
    """Test API key management."""

    def test_create_key(self):
        mgr = APIKeyManager()
        key = mgr.create_key("test_user", "free")
        assert key.startswith("rf_")
        assert len(key) > 10

    def test_verify_key_no_db(self):
        mgr = APIKeyManager()
        info = mgr.verify_key("rf_some_key_value")
        assert info is not None
        assert info.tier == "free"

    def test_verify_invalid_key(self):
        mgr = APIKeyManager()
        info = mgr.verify_key("invalid_key")
        assert info is None

    def test_tier_limits(self):
        assert TIER_LIMITS["free"] == 100
        assert TIER_LIMITS["pro"] == 10_000
        assert TIER_LIMITS["enterprise"] == 1_000_000


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiting."""

    def test_allow_within_limit(self):
        limiter = TokenBucketRateLimiter(rate=100, capacity=100)
        assert limiter.allow("test") is True

    def test_burst_capacity(self):
        limiter = TokenBucketRateLimiter(rate=1, capacity=5)
        # Should allow burst of 5
        for _ in range(5):
            assert limiter.allow("test") is True
        # 6th should fail
        assert limiter.allow("test") is False

    def test_retry_after(self):
        limiter = TokenBucketRateLimiter(rate=1, capacity=1)
        limiter.allow("test")  # Use the token
        retry = limiter.get_retry_after("test")
        assert retry >= 0.0

    def test_different_keys(self):
        limiter = TokenBucketRateLimiter(rate=1, capacity=1)
        assert limiter.allow("key1") is True
        assert limiter.allow("key2") is True  # Different key, separate bucket


class TestPerIPRateLimiter:
    """Test per-IP rate limiting."""

    def test_allow(self):
        limiter = PerIPRateLimiter(requests_per_minute=60)
        assert limiter.allow("127.0.0.1") is True

    def test_exceed_limit(self):
        limiter = PerIPRateLimiter(requests_per_minute=3)
        for _ in range(3):
            limiter.allow("1.2.3.4")
        assert limiter.allow("1.2.3.4") is False

    def test_different_ips(self):
        limiter = PerIPRateLimiter(requests_per_minute=1)
        assert limiter.allow("1.1.1.1") is True
        assert limiter.allow("2.2.2.2") is True  # Different IP


class TestBillingTracker:
    """Test billing/usage tracking."""

    def test_record_usage(self):
        tracker = BillingTracker()
        tracker.record_usage("key1", "task1", "mathematics")
        assert tracker.get_usage_count("key1") == 1

    def test_usage_summary(self):
        tracker = BillingTracker()
        tracker.record_usage("key1", "t1", "mathematics")
        tracker.record_usage("key1", "t2", "code")
        tracker.record_usage("key1", "t3", "mathematics")

        summary = tracker.get_usage_summary("key1")
        assert summary["total_requests"] == 3
        assert summary["by_domain"]["mathematics"] == 2
        assert summary["by_domain"]["code"] == 1


class TestSchemas:
    """Test Pydantic schemas."""

    def test_task_submission_valid(self):
        req = TaskSubmissionRequest(
            problem="This is a valid problem statement for testing.",
            domain="mathematics",
            difficulty=5,
        )
        assert req.problem.startswith("This is")

    def test_task_submission_min_length(self):
        with pytest.raises(Exception):
            TaskSubmissionRequest(problem="short")

    def test_health_response(self):
        resp = HealthResponse(
            status="healthy",
            version="0.1.0",
            uptime_seconds=100.0,
            epoch=5,
            db_connected=True,
        )
        assert resp.status == "healthy"
