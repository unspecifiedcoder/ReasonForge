"""
ReasonForge - API Authentication

API key management with usage tracking and rate limits.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from typing import Optional

from .schemas import APIKeyInfo

logger = logging.getLogger("reasonforge.gateway.auth")

# Tier limits: requests per month
TIER_LIMITS = {
    "free": 100,
    "pro": 10_000,
    "enterprise": 1_000_000,
}


class APIKeyManager:
    """API key management with usage tracking and rate limits."""

    def __init__(self, db=None):
        self.db = db

    def create_key(self, owner: str, tier: str = "free") -> str:
        """Generate a new API key."""
        key_id = str(uuid.uuid4())
        api_key = f"rf_{secrets.token_urlsafe(32)}"
        limit = TIER_LIMITS.get(tier, 100)

        if self.db:
            self.db.save_api_key(
                key_id=key_id,
                api_key=api_key,
                owner=owner,
                tier=tier,
                limit=limit,
            )

        logger.info("Created API key for %s (tier=%s)", owner, tier)
        return api_key

    def verify_key(self, key: str) -> Optional[APIKeyInfo]:
        """Validate key and check rate limits."""
        if not self.db:
            # If no DB, accept any key starting with rf_
            if key.startswith("rf_"):
                return APIKeyInfo(
                    key_id="unknown",
                    owner="unknown",
                    tier="free",
                    requests_used=0,
                    requests_limit=100,
                )
            return None

        key_data = self.db.get_api_key(key)
        if not key_data:
            return None

        info = APIKeyInfo(
            key_id=key_data["key_id"],
            owner=key_data["owner"],
            tier=key_data["tier"],
            requests_used=key_data["requests_used"],
            requests_limit=key_data["requests_limit"],
        )

        # Check quota
        if info.requests_used >= info.requests_limit:
            logger.warning(
                "API key %s exceeded quota (%d/%d)",
                info.key_id, info.requests_used, info.requests_limit,
            )
            return None

        return info

    def track_usage(self, key: str) -> None:
        """Record API usage."""
        if self.db:
            self.db.increment_api_usage(key)
