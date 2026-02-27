"""
ReasonForge - Rate Limiter

Token-bucket rate limiting for the API gateway.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List


class TokenBucketRateLimiter:
    """Token-bucket rate limiter for API requests."""

    def __init__(
        self,
        rate: float = 10.0,      # Tokens per second
        capacity: int = 100,     # Max burst size
    ):
        self.rate = rate
        self.capacity = capacity
        self._buckets: Dict[str, Dict] = defaultdict(
            lambda: {"tokens": capacity, "last_time": time.time()}
        )

    def allow(self, key: str) -> bool:
        """Check if a request is allowed under rate limits."""
        bucket = self._buckets[key]
        now = time.time()
        elapsed = now - bucket["last_time"]

        # Refill tokens
        bucket["tokens"] = min(
            self.capacity,
            bucket["tokens"] + elapsed * self.rate,
        )
        bucket["last_time"] = now

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True

        return False

    def get_retry_after(self, key: str) -> float:
        """Get seconds until the next token is available."""
        bucket = self._buckets[key]
        if bucket["tokens"] >= 1.0:
            return 0.0
        deficit = 1.0 - bucket["tokens"]
        return deficit / self.rate

    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        if key in self._buckets:
            del self._buckets[key]


class PerIPRateLimiter:
    """Per-IP address rate limiting."""

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def allow(self, ip: str) -> bool:
        now = time.time()
        # Clean old entries
        self._requests[ip] = [
            t for t in self._requests[ip] if now - t < 60
        ]

        if len(self._requests[ip]) >= self.rpm:
            return False

        self._requests[ip].append(now)
        return True
