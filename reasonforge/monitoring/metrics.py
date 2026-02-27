"""
ReasonForge - Prometheus Metrics

Counters, histograms, and gauges for subnet monitoring.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("reasonforge.monitoring.metrics")

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class MetricsCollector:
    """Prometheus metrics for subnet monitoring."""

    def __init__(self, neuron_type: str = "validator", uid: int = 0):
        self.neuron_type = neuron_type
        self.uid = uid
        self._initialized = False

        if not HAS_PROMETHEUS:
            logger.debug("prometheus_client not available, metrics disabled")
            return

        prefix = f"reasonforge_{neuron_type}"

        # Counters
        self.tasks_processed = Counter(
            f"{prefix}_tasks_total",
            "Total tasks processed",
            ["domain", "difficulty"],
        )
        self.epochs_completed = Counter(
            f"{prefix}_epochs_total",
            "Epochs completed",
        )
        self.traps_injected = Counter(
            f"{prefix}_traps_total",
            "Trap problems injected",
        )
        self.breakthroughs = Counter(
            f"{prefix}_breakthroughs_total",
            "Breakthrough solutions",
        )
        self.plagiarism_detected = Counter(
            f"{prefix}_plagiarism_total",
            "Plagiarism detections",
        )
        self.weight_set_failures = Counter(
            f"{prefix}_weight_failures_total",
            "Weight setting failures",
        )

        # Histograms
        self.task_latency = Histogram(
            f"{prefix}_task_latency_seconds",
            "Task processing time",
            ["domain"],
        )
        self.cms_distribution = Histogram(
            f"{prefix}_cms_score",
            "CMS score distribution",
            buckets=[0.1 * i for i in range(11)],
        )

        # Gauges
        self.current_epoch = Gauge(
            f"{prefix}_current_epoch",
            "Current epoch number",
        )
        self.active_miners = Gauge(
            f"{prefix}_active_miners",
            "Number of active miners",
        )
        self.avg_cms = Gauge(
            f"{prefix}_avg_cms",
            "Average CMS this epoch",
        )
        self.total_emission = Gauge(
            f"{prefix}_total_emission_tao",
            "Total TAO emitted",
        )
        self.top_miner_score = Gauge(
            f"{prefix}_top_miner_score",
            "Highest S_epoch",
        )

        self._initialized = True

    def start_server(self, port: Optional[int] = None) -> None:
        """Start Prometheus metrics HTTP server."""
        if not HAS_PROMETHEUS or not self._initialized:
            return
        port = port or (9090 + self.uid)
        try:
            start_http_server(port)
            logger.info("Metrics server started on port %d", port)
        except Exception as e:
            logger.warning("Failed to start metrics server: %s", e)

    def record_task(self, domain: str, difficulty: int, latency_s: float) -> None:
        if not self._initialized:
            return
        self.tasks_processed.labels(domain=domain, difficulty=str(difficulty)).inc()
        self.task_latency.labels(domain=domain).observe(latency_s)

    def record_epoch(
        self, epoch_id: int, n_miners: int, avg_score: float, top_score: float
    ) -> None:
        if not self._initialized:
            return
        self.epochs_completed.inc()
        self.current_epoch.set(epoch_id)
        self.active_miners.set(n_miners)
        self.avg_cms.set(avg_score)
        self.top_miner_score.set(top_score)

    def record_cms(self, score: float) -> None:
        if not self._initialized:
            return
        self.cms_distribution.observe(score)

    def record_plagiarism(self) -> None:
        if not self._initialized:
            return
        self.plagiarism_detected.inc()

    def record_weight_failure(self) -> None:
        if not self._initialized:
            return
        self.weight_set_failures.inc()
