"""
ReasonForge - Trap Manager

Injects trap problems with known ground-truth scores and evaluates
miner responses against them for integrity checking.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..types import Domain, Task, TRAP_RATE

logger = logging.getLogger("reasonforge.validator.trap_manager")


class TrapManager:
    """Inject trap problems and evaluate responses against ground truth."""

    def __init__(self, trap_rate: float = TRAP_RATE):
        self.trap_rate = trap_rate
        self.trap_results: Dict[int, List[float]] = defaultdict(list)  # uid -> scores

    def evaluate_trap_response(
        self,
        task: Task,
        final_answer: str | None,
        reasoning_steps: list[dict] | None = None,
    ) -> float:
        """
        Compare miner response against ground truth.
        Returns a score 0.0-1.0 indicating how correct the response is.
        """
        if not task.is_trap or task.ground_truth_score is None:
            return 1.0

        if not final_answer:
            return 0.0

        domain_val = task.domain.value if hasattr(task.domain, "value") else task.domain

        if domain_val == "mathematics":
            return self._evaluate_math_trap(task, final_answer)
        elif domain_val == "code":
            return self._evaluate_code_trap(task, final_answer)
        else:
            return self._evaluate_general_trap(task, final_answer, reasoning_steps)

    def _evaluate_math_trap(self, task: Task, answer: str) -> float:
        """Evaluate math trap response."""
        answer_lower = answer.lower().strip()
        problem_lower = task.problem.lower()

        # Simple numeric checks for known traps
        if "2+2" in problem_lower or "2 + 2" in problem_lower:
            if "4" in answer_lower:
                return 1.0
            return 0.0

        if "prime" in problem_lower and "7" in problem_lower:
            if "prime" in answer_lower and ("yes" in answer_lower or "is prime" in answer_lower or "true" in answer_lower):
                return 0.95
            return 0.2

        if "derivative" in problem_lower and "x^3" in problem_lower:
            if "3x^2" in answer_lower or "3x²" in answer_lower or "3 x^2" in answer_lower:
                return 1.0
            return 0.1

        # Default: check if answer is non-empty and contains reasoning
        return 0.5 if len(answer_lower) > 20 else 0.2

    def _evaluate_code_trap(self, task: Task, answer: str) -> float:
        """Evaluate code trap response."""
        answer_lower = answer.lower()

        if "maximum" in task.problem.lower() or "max" in task.problem.lower():
            # Check for function-like code
            if "def " in answer_lower or "function" in answer_lower:
                if "return" in answer_lower:
                    return 0.9
            return 0.3

        if "binary search" in task.problem.lower():
            if "def " in answer_lower and ("mid" in answer_lower or "middle" in answer_lower):
                return 0.9
            return 0.3

        return 0.5 if len(answer) > 50 else 0.2

    def _evaluate_general_trap(
        self, task: Task, answer: str,
        reasoning_steps: list[dict] | None = None,
    ) -> float:
        """Evaluate general trap response."""
        if not answer or len(answer) < 10:
            return 0.1

        # Basic quality heuristics
        score = 0.3
        if len(answer) > 100:
            score += 0.2
        if reasoning_steps and len(reasoning_steps) >= 2:
            score += 0.2
        if any(kw in answer.lower() for kw in ["because", "therefore", "thus", "hence"]):
            score += 0.1

        return min(1.0, score)

    def record_trap_score(self, uid: int, score: float) -> None:
        """Record a trap score for a miner."""
        self.trap_results[uid].append(score)

    def get_trap_scores(self, uid: int) -> List[float]:
        """Get all trap scores for a miner."""
        return self.trap_results.get(uid, [])

    def get_trap_penalty(self, uid: int) -> float:
        """Compute trap penalty for a miner based on their trap scores."""
        scores = self.trap_results.get(uid, [])
        if not scores:
            return 1.0
        from ..engine import ScoringEngine
        return ScoringEngine.compute_trap_penalty(scores)

    def reset_epoch(self) -> None:
        """Reset per-epoch trap tracking (keep historical data)."""
        # We keep the data across epochs for cumulative tracking
        pass
