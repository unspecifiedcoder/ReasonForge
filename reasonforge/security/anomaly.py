"""
ReasonForge - Anomaly Detection

Detect suspicious miner behavior patterns including timing anomalies,
score manipulation, and collusion.
"""

from __future__ import annotations

import logging
import statistics
from typing import List, Optional, Tuple

logger = logging.getLogger("reasonforge.security.anomaly")


class AnomalyDetector:
    """Detect suspicious miner behavior patterns."""

    def __init__(
        self,
        min_solve_time_per_difficulty: int = 500,  # ms per difficulty level
        score_variance_threshold: float = 0.001,
        collusion_similarity_threshold: float = 0.98,
    ):
        self.min_solve_time_per_difficulty = min_solve_time_per_difficulty
        self.score_variance_threshold = score_variance_threshold
        self.collusion_threshold = collusion_similarity_threshold

    def check_timing_anomaly(self, time_ms: int, difficulty: int) -> bool:
        """
        Flag if solve time is unrealistically fast for difficulty.

        Returns:
            True if anomalous, False if normal.
        """
        min_expected = difficulty * self.min_solve_time_per_difficulty
        if time_ms < min_expected:
            logger.warning(
                "Timing anomaly: %dms for difficulty %d (min expected: %dms)",
                time_ms, difficulty, min_expected,
            )
            return True
        return False

    def check_score_manipulation(self, cms_history: List[float]) -> bool:
        """
        Flag if CMS scores are suspiciously consistent (potential gaming).

        Returns:
            True if suspicious, False if normal.
        """
        if len(cms_history) < 5:
            return False

        try:
            variance = statistics.variance(cms_history)
            if variance < self.score_variance_threshold:
                logger.warning(
                    "Score manipulation suspected: variance=%.6f (threshold=%.6f)",
                    variance, self.score_variance_threshold,
                )
                return True
        except statistics.StatisticsError:
            pass

        return False

    def check_sudden_improvement(
        self, recent_scores: List[float], historical_avg: float
    ) -> bool:
        """
        Flag if a miner suddenly improves dramatically (potential identity swap).

        Returns:
            True if suspicious, False if normal.
        """
        if not recent_scores or historical_avg <= 0:
            return False

        recent_avg = sum(recent_scores) / len(recent_scores)
        improvement_ratio = recent_avg / historical_avg

        if improvement_ratio > 2.0:
            logger.warning(
                "Sudden improvement detected: %.2f -> %.2f (%.1fx)",
                historical_avg, recent_avg, improvement_ratio,
            )
            return True

        return False

    def check_collusion(
        self,
        submissions: List[dict],
        similarity_fn=None,
    ) -> List[Tuple[int, int, float]]:
        """
        Detect colluding miners with near-identical submissions.

        Args:
            submissions: List of {uid, text} dicts.
            similarity_fn: Function(text_a, text_b) -> float.

        Returns:
            List of (uid_a, uid_b, similarity) tuples for flagged pairs.
        """
        flagged = []

        if similarity_fn is None:
            # Basic text similarity fallback
            similarity_fn = self._jaccard_similarity

        for i in range(len(submissions)):
            for j in range(i + 1, len(submissions)):
                text_a = submissions[i].get("text", "")
                text_b = submissions[j].get("text", "")

                if not text_a or not text_b:
                    continue

                sim = similarity_fn(text_a, text_b)
                if sim > self.collusion_threshold:
                    uid_a = submissions[i].get("uid", i)
                    uid_b = submissions[j].get("uid", j)
                    flagged.append((uid_a, uid_b, sim))
                    logger.warning(
                        "Collusion detected: UID %d and UID %d (similarity=%.4f)",
                        uid_a, uid_b, sim,
                    )

        return flagged

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Simple Jaccard similarity as fallback."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a and not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0

    def get_anomaly_report(
        self,
        uid: int,
        time_ms: int,
        difficulty: int,
        cms_history: List[float],
    ) -> dict:
        """Generate a full anomaly report for a miner."""
        return {
            "uid": uid,
            "timing_anomaly": self.check_timing_anomaly(time_ms, difficulty),
            "score_manipulation": self.check_score_manipulation(cms_history),
            "flags_count": sum([
                self.check_timing_anomaly(time_ms, difficulty),
                self.check_score_manipulation(cms_history),
            ]),
        }
