"""
ReasonForge - Rate Guard

Per-UID rate limiting to prevent DoS attacks from miners/validators.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger("reasonforge.security.rate_guard")


class RateGuard:
    """Per-UID rate limiting to prevent DoS."""

    def __init__(self, max_requests_per_minute: int = 10):
        self.max_rpm = max_requests_per_minute
        self._requests: Dict[int, List[float]] = defaultdict(list)

    def check(self, uid: int) -> bool:
        """
        Check if a request from this UID is allowed.

        Returns:
            True if request is allowed, False if rate limited.
        """
        now = time.time()
        # Clean old entries
        self._requests[uid] = [t for t in self._requests[uid] if now - t < 60]

        if len(self._requests[uid]) >= self.max_rpm:
            logger.warning("Rate limited UID %d (%d requests/min)", uid, len(self._requests[uid]))
            return False

        self._requests[uid].append(now)
        return True

    def get_remaining(self, uid: int) -> int:
        """Get remaining requests for a UID in the current window."""
        now = time.time()
        recent = [t for t in self._requests.get(uid, []) if now - t < 60]
        return max(0, self.max_rpm - len(recent))

    def reset(self, uid: int) -> None:
        """Reset rate limit for a specific UID."""
        if uid in self._requests:
            del self._requests[uid]

    def reset_all(self) -> None:
        """Reset all rate limits."""
        self._requests.clear()
