"""Python code-claim verifier for ReasonForge."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from reasonforge.translation.types import StepTranslation
from .verdict import StepVerdict, VerificationVerdict

logger = logging.getLogger(__name__)

# Patterns considered dangerous in untrusted code snippets.
_DANGEROUS_PATTERNS: List[str] = [
    r"\bimport\s+os\b",
    r"\bimport\s+subprocess\b",
    r"\bfrom\s+os\b",
    r"\bfrom\s+subprocess\b",
    r"\b__import__\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
]


class CodeVerifier:
    """Verify Python code claims via syntax & safety checks."""

    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify_chain(
        self,
        task_id: str,
        translations: List[StepTranslation],
        original_code_claim: str = "",
    ) -> VerificationVerdict:
        """Verify each translated step that contains Python code.

        A step is marked *verified* when:
        1. Its ``formal_representation`` compiles without ``SyntaxError``.
        2. It contains no dangerous patterns (OS access, eval, etc.).
        """
        step_verdicts: List[StepVerdict] = []

        for t in translations:
            code = t.formal_representation
            if not code.strip():
                step_verdicts.append(
                    StepVerdict(
                        step_id=t.step_id,
                        verified=False,
                        error_message="Empty formal representation",
                        formal_representation=code,
                    )
                )
                continue

            syntax_err = self._check_syntax(code)
            if syntax_err is not None:
                step_verdicts.append(
                    StepVerdict(
                        step_id=t.step_id,
                        verified=False,
                        error_message=f"SyntaxError: {syntax_err}",
                        formal_representation=code,
                    )
                )
                continue

            danger_warn = self._check_dangerous_patterns(code)
            if danger_warn is not None:
                step_verdicts.append(
                    StepVerdict(
                        step_id=t.step_id,
                        verified=False,
                        error_message=f"Dangerous pattern: {danger_warn}",
                        formal_representation=code,
                    )
                )
                continue

            step_verdicts.append(
                StepVerdict(
                    step_id=t.step_id,
                    verified=True,
                    formal_representation=code,
                )
            )

        verified_count = sum(1 for sv in step_verdicts if sv.verified)
        failure_points = [sv for sv in step_verdicts if not sv.verified]
        overall = (
            "PASSED"
            if verified_count == len(step_verdicts) and step_verdicts
            else "FAILED"
        )

        return VerificationVerdict(
            task_id=task_id,
            overall=overall,
            step_verdicts=step_verdicts,
            total_steps=len(step_verdicts),
            verified_steps=verified_count,
            failure_points=failure_points,
            domain="code",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_syntax(code: str) -> Optional[str]:
        """Return *None* if *code* compiles; otherwise the error message."""
        try:
            compile(code, "<step>", "exec")
        except SyntaxError as exc:
            return str(exc)
        return None

    @staticmethod
    def _check_dangerous_patterns(code: str) -> Optional[str]:
        """Return a warning string if *code* contains dangerous patterns."""
        for pattern in _DANGEROUS_PATTERNS:
            match = re.search(pattern, code)
            if match:
                return f"Matched forbidden pattern: {match.group()}"
        return None
