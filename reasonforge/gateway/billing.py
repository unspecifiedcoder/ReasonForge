"""
ReasonForge - Billing Module

Usage tracking and quotas for API consumers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger("reasonforge.gateway.billing")


@dataclass
class UsageRecord:
    """Single API usage record."""
    key_id: str
    task_id: str
    domain: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tokens_used: int = 0


class BillingTracker:
    """Track API usage for billing and quota enforcement."""

    def __init__(self, db=None):
        self.db = db
        self._usage: Dict[str, List[UsageRecord]] = {}

    def record_usage(
        self,
        key_id: str,
        task_id: str,
        domain: str = "",
        tokens_used: int = 0,
    ) -> None:
        """Record an API usage event."""
        record = UsageRecord(
            key_id=key_id,
            task_id=task_id,
            domain=domain,
            tokens_used=tokens_used,
        )
        if key_id not in self._usage:
            self._usage[key_id] = []
        self._usage[key_id].append(record)

    def get_usage_count(self, key_id: str) -> int:
        """Get total usage count for a key."""
        return len(self._usage.get(key_id, []))

    def get_usage_summary(self, key_id: str) -> Dict:
        """Get usage summary for a key."""
        records = self._usage.get(key_id, [])
        domains: Dict[str, int] = {}
        for r in records:
            domains[r.domain] = domains.get(r.domain, 0) + 1

        return {
            "key_id": key_id,
            "total_requests": len(records),
            "total_tokens": sum(r.tokens_used for r in records),
            "by_domain": domains,
        }
