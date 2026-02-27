"""
ReasonForge - Validator Scoring Pipeline

Orchestrates the full scoring pipeline for miner submissions.
Wraps the MVP ScoringEngine and adds production verification backends.
"""

from __future__ import annotations

import logging

from ..engine import ScoringEngine
from ..types import DimensionScores, Task
from .objective_scorer import ObjectiveScorer

logger = logging.getLogger("reasonforge.validator.scoring")


class ValidatorScorer:
    """
    Orchestrates the scoring pipeline using the MVP's ScoringEngine.
    Adds objective verification backends on top of the formula layer.
    """

    def __init__(
        self,
        lean4_enabled: bool = False,
        sandbox_enabled: bool = False,
    ):
        self.engine = ScoringEngine()

        # Initialize verification backends
        lean4_checker = None
        code_sandbox = None
        math_checker = None
        fact_checker = None

        if lean4_enabled:
            try:
                from ..verification.lean4_checker import Lean4Checker
                lean4_checker = Lean4Checker()
            except ImportError:
                logger.warning("Lean4 checker not available")

        if sandbox_enabled:
            try:
                from ..verification.code_sandbox import CodeSandbox
                code_sandbox = CodeSandbox()
            except ImportError:
                logger.warning("Code sandbox not available")

        try:
            from ..verification.math_checker import MathChecker
            math_checker = MathChecker()
        except ImportError:
            logger.debug("Math checker not available")

        try:
            from ..verification.fact_checker import FactChecker
            fact_checker = FactChecker()
        except ImportError:
            logger.debug("Fact checker not available")

        self.objective_scorer = ObjectiveScorer(
            lean4_checker=lean4_checker,
            code_sandbox=code_sandbox,
            math_checker=math_checker,
            fact_checker=fact_checker,
        )

    async def compute_dimensions(
        self, task: Task, response: dict
    ) -> DimensionScores:
        """
        Compute all 4 dimension scores for a miner's response.
        Maps to Quality, Accuracy, Novelty, Efficiency in the whitepaper.
        """
        quality = self._score_quality(task, response)
        accuracy = await self._score_accuracy(task, response)
        novelty = self._score_novelty(task, response)
        efficiency = self._score_efficiency(task, response)

        return DimensionScores(
            quality=quality,
            accuracy=accuracy,
            novelty=novelty,
            efficiency=efficiency,
        )

    def _score_quality(self, task: Task, response: dict) -> float:
        """
        Quality (40% of CMS):
        - Step coherence, completeness, depth
        - Formal proof fragments (bonus)
        """
        steps = response.get("steps", [])
        if not steps:
            return 0.0

        # Step count vs difficulty expectation
        expected_steps = max(3, task.difficulty)
        step_ratio = min(1.0, len(steps) / expected_steps)

        # Average confidence
        confidences = [s.get("confidence", 0) for s in steps]
        avg_confidence = sum(confidences) / len(confidences)

        # Evidence presence
        evidence_count = sum(1 for s in steps if s.get("evidence"))
        evidence_ratio = evidence_count / len(steps)

        # Proof fragment bonus
        proof_bonus = 0.1 if any(s.get("formal_proof_fragment") for s in steps) else 0.0

        return min(1.0,
                   (0.3 * step_ratio)
                   + (0.3 * avg_confidence)
                   + (0.2 * evidence_ratio)
                   + (0.2 + proof_bonus))

    async def _score_accuracy(self, task: Task, response: dict) -> float:
        """
        Accuracy (30% of CMS):
        Domain-specific automated checks via ObjectiveScorer.
        """
        try:
            return await self.objective_scorer.compute_objective_score(task, response)
        except Exception as e:
            logger.warning("Accuracy scoring failed: %s", e)
            return 0.3  # Default fallback

    def _score_novelty(self, task: Task, response: dict) -> float:
        """
        Novelty (15% of CMS):
        - Unique approach vs common solutions
        - Creative reasoning paths
        """
        steps = response.get("steps", [])
        if not steps:
            return 0.0

        # Step text length as proxy for reasoning depth
        avg_step_length = sum(
            len(s.get("reasoning", "")) for s in steps
        ) / len(steps)
        length_score = min(1.0, avg_step_length / 500)

        # Vocabulary diversity
        all_words = " ".join(
            s.get("reasoning", "") for s in steps
        ).lower().split()
        if all_words:
            diversity = len(set(all_words)) / len(all_words)
        else:
            diversity = 0.0

        return min(1.0, 0.5 * length_score + 0.5 * diversity)

    def _score_efficiency(self, task: Task, response: dict) -> float:
        """
        Efficiency (15% of CMS):
        - Solve time relative to timeout
        - Conciseness
        """
        time_ms = response.get("time_taken_ms") or (task.timeout_seconds * 1000)
        timeout_ms = task.timeout_seconds * 1000

        time_ratio = time_ms / timeout_ms
        if time_ratio < 0.01:
            # Suspiciously fast
            time_score = 0.2
        elif time_ratio > 1.0:
            # Timed out
            time_score = 0.0
        else:
            time_score = 1.0 - (time_ratio * 0.5)

        return min(1.0, time_score)
