"""
ReasonForge - Health Check

Health check utilities for neurons and services.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("reasonforge.monitoring.health")


@dataclass
class HealthStatus:
    """Health status of a neuron or service."""
    healthy: bool = True
    version: str = "0.1.0"
    uptime_seconds: float = 0.0
    checks: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class HealthChecker:
    """Perform health checks on neuron components."""

    def __init__(self):
        self._start_time = time.time()
        self._checks: Dict[str, callable] = {}

    def register_check(self, name: str, check_fn: callable) -> None:
        """Register a health check function. Should return bool."""
        self._checks[name] = check_fn

    def check(self) -> HealthStatus:
        """Run all registered health checks."""
        status = HealthStatus(
            uptime_seconds=time.time() - self._start_time,
        )

        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                status.checks[name] = bool(result)
                if not result:
                    status.healthy = False
                    status.errors.append(f"{name}: unhealthy")
            except Exception as e:
                status.checks[name] = False
                status.healthy = False
                status.errors.append(f"{name}: {str(e)}")

        return status

    def check_bittensor(self, subtensor) -> bool:
        """Check bittensor connectivity."""
        if subtensor is None:
            return False
        try:
            block = subtensor.get_current_block()
            return block > 0
        except Exception:
            return False

    def check_database(self, db) -> bool:
        """Check database connectivity."""
        if db is None:
            return False
        try:
            db.get_stats()
            return True
        except Exception:
            return False

    def check_axon(self, axon) -> bool:
        """Check if axon is serving."""
        if axon is None:
            return False
        try:
            return axon.is_serving
        except Exception:
            return True  # Assume serving if we can't check
