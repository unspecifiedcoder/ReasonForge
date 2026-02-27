"""
ReasonForge - Input Sanitizer

Validate and sanitize all inputs from miners and external API.
"""

from __future__ import annotations

import base64
import logging
import re

logger = logging.getLogger("reasonforge.security.sanitizer")


class InputSanitizer:
    """Validate and sanitize all inputs from miners and external API."""

    MAX_STEP_LENGTH = 10_000     # chars per reasoning step
    MAX_STEPS = 50               # max steps per submission
    MAX_ANSWER_LENGTH = 50_000   # chars
    MAX_PROOF_SIZE = 1_000_000   # bytes (1MB)
    MAX_CODE_SIZE = 500_000      # bytes (500KB)
    MAX_PROBLEM_LENGTH = 10_000  # chars

    @staticmethod
    def sanitize_submission(response) -> None:
        """
        Validate and sanitize all miner-provided fields in-place.
        Truncates oversized fields, strips injection attempts.
        """
        # 1. Truncate reasoning steps
        if hasattr(response, "reasoning_steps") and response.reasoning_steps:
            if len(response.reasoning_steps) > InputSanitizer.MAX_STEPS:
                response.reasoning_steps = response.reasoning_steps[:InputSanitizer.MAX_STEPS]
                logger.warning("Truncated steps from submission")

            for step in response.reasoning_steps:
                if isinstance(step, dict):
                    reasoning = step.get("reasoning", "")
                    if len(reasoning) > InputSanitizer.MAX_STEP_LENGTH:
                        step["reasoning"] = reasoning[:InputSanitizer.MAX_STEP_LENGTH]

                    evidence = step.get("evidence", "")
                    if len(evidence) > InputSanitizer.MAX_STEP_LENGTH:
                        step["evidence"] = evidence[:InputSanitizer.MAX_STEP_LENGTH]

                    # Sanitize confidence to valid range
                    conf = step.get("confidence", 0.0)
                    step["confidence"] = max(0.0, min(1.0, float(conf)))

        # 2. Truncate final answer
        if hasattr(response, "final_answer") and response.final_answer:
            if len(response.final_answer) > InputSanitizer.MAX_ANSWER_LENGTH:
                response.final_answer = response.final_answer[:InputSanitizer.MAX_ANSWER_LENGTH]

        # 3. Validate proof artifact size
        if hasattr(response, "proof_artifact") and response.proof_artifact:
            try:
                decoded = base64.b64decode(response.proof_artifact)
                if len(decoded) > InputSanitizer.MAX_PROOF_SIZE:
                    response.proof_artifact = None
                    logger.warning("Removed oversized proof artifact")
            except Exception:
                response.proof_artifact = None

        # 4. Validate code artifact size
        if hasattr(response, "code_artifact") and response.code_artifact:
            try:
                decoded = base64.b64decode(response.code_artifact)
                if len(decoded) > InputSanitizer.MAX_CODE_SIZE:
                    response.code_artifact = None
                    logger.warning("Removed oversized code artifact")
            except Exception:
                response.code_artifact = None

    @staticmethod
    def sanitize_problem(problem: str) -> str:
        """Sanitize a problem statement from external API."""
        if not problem:
            return ""

        # Truncate
        problem = problem[:InputSanitizer.MAX_PROBLEM_LENGTH]

        # Remove potential injection patterns
        # Remove script tags
        problem = re.sub(r"<script[^>]*>.*?</script>", "", problem, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        problem = re.sub(r"<[^>]+>", "", problem)

        return problem.strip()

    @staticmethod
    def validate_domain(domain: str) -> bool:
        """Validate that a domain is one of the allowed values."""
        valid_domains = {"mathematics", "code", "scientific", "strategic", "causal", "ethical"}
        return domain.lower() in valid_domains

    @staticmethod
    def validate_difficulty(difficulty: int) -> bool:
        """Validate difficulty is in range."""
        return 1 <= difficulty <= 10
