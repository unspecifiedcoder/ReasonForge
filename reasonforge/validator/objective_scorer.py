"""
ReasonForge - Objective Scorer

Domain-specific automated checks for scoring miner submissions.
Provides the objective component of the scoring pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict

from ..types import DOMAIN_CHECK_WEIGHTS, Domain, Task

logger = logging.getLogger("reasonforge.validator.objective_scorer")


class ObjectiveScorer:
    """Domain-specific automated checks for miner submissions."""

    def __init__(
        self,
        lean4_checker=None,
        code_sandbox=None,
        math_checker=None,
        fact_checker=None,
    ):
        self.lean4 = lean4_checker
        self.sandbox = code_sandbox
        self.math_checker = math_checker
        self.fact_checker = fact_checker

    async def compute_objective_score(
        self, task: Task, response_data: dict
    ) -> float:
        """
        Compute objective score using domain-specific automated checks.
        Maps to Eq. 11 in the whitepaper.
        """
        domain = task.domain if isinstance(task.domain, Domain) else Domain(task.domain)
        weights = DOMAIN_CHECK_WEIGHTS.get(domain, {})

        checks = await self._run_domain_checks(domain, task, response_data)

        # Use engine's objective score formula
        from ..engine import ScoringEngine
        return ScoringEngine.compute_objective_score(checks, weights)

    async def _run_domain_checks(
        self, domain: Domain, task: Task, response: dict
    ) -> Dict[str, float]:
        """Run domain-specific verification checks."""
        if domain == Domain.MATHEMATICS:
            return await self._check_mathematics(task, response)
        elif domain == Domain.CODE:
            return await self._check_code(task, response)
        elif domain == Domain.SCIENTIFIC:
            return await self._check_scientific(task, response)
        elif domain == Domain.STRATEGIC:
            return await self._check_strategic(task, response)
        elif domain == Domain.CAUSAL:
            return await self._check_causal(task, response)
        elif domain == Domain.ETHICAL:
            return await self._check_ethical(task, response)
        return {}

    async def _check_mathematics(self, task: Task, response: dict) -> Dict[str, float]:
        checks = {}

        # Lean4 proof verification
        if self.lean4 and response.get("proof_artifact"):
            try:
                checks["proof"] = await self.lean4.verify(response["proof_artifact"])
            except Exception as e:
                logger.warning("Lean4 verification failed: %s", e)
                checks["proof"] = 0.0
        else:
            # No proof artifact: partial credit for reasoning
            checks["proof"] = 0.3 if response.get("steps") else 0.0

        # Numerical verification
        if self.math_checker:
            try:
                checks["numerical"] = self.math_checker.verify(
                    task.problem, response.get("final_answer", "")
                )
            except Exception:
                checks["numerical"] = 0.5
        else:
            checks["numerical"] = 0.5  # Default when no checker available

        # Step verification
        checks["steps"] = self._verify_reasoning_steps(response.get("steps", []))

        return checks

    async def _check_code(self, task: Task, response: dict) -> Dict[str, float]:
        checks = {}

        if self.sandbox and response.get("code_artifact"):
            try:
                checks["tests"] = await self.sandbox.run_tests(response["code_artifact"])
                checks["static_analysis"] = await self.sandbox.lint(response["code_artifact"])
            except Exception as e:
                logger.warning("Code sandbox check failed: %s", e)
                checks["tests"] = 0.0
                checks["static_analysis"] = 0.0
        else:
            checks["tests"] = 0.3 if response.get("code_artifact") else 0.0
            checks["static_analysis"] = 0.5

        checks["formal"] = 0.5 if response.get("steps") else 0.0

        return checks

    async def _check_scientific(self, task: Task, response: dict) -> Dict[str, float]:
        steps = response.get("steps", [])
        checks = {
            "simulation": self._check_quantitative_content(steps),
            "statistics": self._check_statistical_content(steps),
            "citations": self._check_citation_content(steps),
        }
        return checks

    async def _check_strategic(self, task: Task, response: dict) -> Dict[str, float]:
        steps = response.get("steps", [])
        checks = {
            "solver": self._check_formal_solution(steps),
            "constraints": self._check_constraint_handling(steps),
            "equilibrium": self._check_equilibrium_analysis(steps),
        }
        return checks

    async def _check_causal(self, task: Task, response: dict) -> Dict[str, float]:
        _steps = response.get("steps", [])  # noqa: F841 — reserved for future causal DAG checks
        answer = response.get("final_answer", "").lower()
        checks = {
            "docalculus": 0.7 if any(kw in answer for kw in ["do(", "intervention", "do-calculus"]) else 0.3,
            "bootstrap": 0.5 if "confidence" in answer or "interval" in answer else 0.3,
            "dag": 0.7 if any(kw in answer for kw in ["dag", "graph", "node", "edge", "path"]) else 0.3,
        }
        return checks

    async def _check_ethical(self, task: Task, response: dict) -> Dict[str, float]:
        steps = response.get("steps", [])
        answer = response.get("final_answer", "").lower()
        all_text = " ".join([s.get("reasoning", "") for s in steps]) + " " + answer
        all_text_lower = all_text.lower()

        frameworks = ["utilitarian", "deontolog", "virtue", "consequential", "kantian", "rawls"]
        framework_count = sum(1 for f in frameworks if f in all_text_lower)

        checks = {
            "coverage": min(1.0, framework_count / 3.0),
            "logic": self._check_logical_structure(steps),
        }
        return checks

    def _verify_reasoning_steps(self, steps: list) -> float:
        """Score the quality of reasoning steps."""
        if not steps:
            return 0.0

        score = 0.0
        for step in steps:
            reasoning = step.get("reasoning", "")
            # Check for substantive content
            if len(reasoning) > 50:
                score += 0.3
            if step.get("evidence"):
                score += 0.2
            if step.get("confidence", 0) > 0.5:
                score += 0.1

        return min(1.0, score / max(1, len(steps)))

    def _check_quantitative_content(self, steps: list) -> float:
        """Check for quantitative/numerical content in steps."""
        import re
        all_text = " ".join(s.get("reasoning", "") for s in steps)
        numbers = re.findall(r"\d+\.?\d*", all_text)
        return min(1.0, len(numbers) / 10.0) if numbers else 0.2

    def _check_statistical_content(self, steps: list) -> float:
        all_text = " ".join(s.get("reasoning", "") for s in steps).lower()
        keywords = ["mean", "variance", "standard deviation", "p-value",
                    "confidence interval", "regression", "correlation", "hypothesis"]
        found = sum(1 for kw in keywords if kw in all_text)
        return min(1.0, found / 3.0)

    def _check_citation_content(self, steps: list) -> float:
        all_text = " ".join(s.get("reasoning", "") for s in steps)
        indicators = ["et al", "according to", "study", "research", "paper", "["]
        found = sum(1 for ind in indicators if ind.lower() in all_text.lower())
        return min(1.0, found / 2.0)

    def _check_formal_solution(self, steps: list) -> float:
        all_text = " ".join(s.get("reasoning", "") for s in steps).lower()
        keywords = ["optimal", "maximize", "minimize", "equilibrium", "solution",
                    "payoff", "strategy", "constraint"]
        found = sum(1 for kw in keywords if kw in all_text)
        return min(1.0, found / 3.0)

    def _check_constraint_handling(self, steps: list) -> float:
        all_text = " ".join(s.get("reasoning", "") for s in steps).lower()
        if "constraint" in all_text or "subject to" in all_text or "bound" in all_text:
            return 0.7
        return 0.3

    def _check_equilibrium_analysis(self, steps: list) -> float:
        all_text = " ".join(s.get("reasoning", "") for s in steps).lower()
        keywords = ["nash", "equilibrium", "dominant", "pareto", "mixed strategy"]
        found = sum(1 for kw in keywords if kw in all_text)
        return min(1.0, found / 2.0)

    def _check_logical_structure(self, steps: list) -> float:
        """Check for logical reasoning structure."""
        if not steps:
            return 0.0
        all_text = " ".join(s.get("reasoning", "") for s in steps).lower()
        connectors = ["therefore", "because", "however", "moreover", "furthermore",
                      "on the other hand", "in contrast", "consequently"]
        found = sum(1 for c in connectors if c in all_text)
        return min(1.0, 0.3 + found * 0.15)
