# reasonforge/codeverify/report.py
#
# Structured verification report for ReasonForge code verification.
#
# Combines syntax analysis, security scanning, test generation, and test
# execution results into a single certificate-ready output.  Every report
# carries a deterministic SHA-256 report ID derived from the code hash,
# verdict, and timestamp so that downstream consumers can verify provenance.

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Security risk → numeric score mapping used by confidence computation.
# ---------------------------------------------------------------------------

_SECURITY_SCORE: Dict[str, float] = {
    "safe": 1.0,
    "low": 0.8,
    "medium": 0.5,
    "high": 0.2,
    "critical": 0.0,
}

# Weights for the composite confidence formula.
_W_SYNTAX: float = 0.2
_W_SECURITY: float = 0.3
_W_TESTS: float = 0.5


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class CodeVerificationReport:
    """Aggregated result of a full ReasonForge verification pipeline run.

    Fields are grouped into four logical sections: identity, syntax analysis,
    security analysis, test execution, and overall verdict.
    """

    # -- identity -----------------------------------------------------------
    report_id: str = ""
    timestamp: str = ""
    code_hash: str = ""
    code_length_bytes: int = 0
    code_line_count: int = 0

    # -- syntax analysis ----------------------------------------------------
    syntax_valid: bool = True
    syntax_error: Optional[str] = None

    # -- security analysis --------------------------------------------------
    security_risk_level: str = "safe"
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    security_issue_count: int = 0

    # -- test execution -----------------------------------------------------
    tests_generated: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    test_pass_rate: float = 0.0
    test_execution_time_ms: int = 0
    test_output: str = ""

    # -- overall verdict ----------------------------------------------------
    verdict: str = "ERROR"
    confidence_score: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def compute_report_id(self) -> str:
        """Generate a deterministic SHA-256 report ID.

        The hash is derived from :attr:`code_hash`, :attr:`verdict`, and
        :attr:`timestamp` so that two reports for the same code/verdict/time
        always produce the same identifier.
        """
        payload = f"{self.code_hash}:{self.verdict}:{self.timestamp}"
        self.report_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.report_id

    def compute_confidence(self) -> float:
        """Calculate the composite confidence score.

        Formula::

            confidence = 0.2 * syntax_score
                       + 0.3 * security_score
                       + 0.5 * test_pass_rate

        Where:
        - ``syntax_score`` is 1.0 when :attr:`syntax_valid` is ``True``,
          0.0 otherwise.
        - ``security_score`` is looked up from the risk-level mapping
          (safe → 1.0, low → 0.8, medium → 0.5, high → 0.2, critical → 0.0).
        - ``test_pass_rate`` is :attr:`test_pass_rate` (already 0.0–1.0).
        """
        syntax_score: float = 1.0 if self.syntax_valid else 0.0
        security_score: float = _SECURITY_SCORE.get(self.security_risk_level, 0.0)

        self.confidence_score = (
            _W_SYNTAX * syntax_score
            + _W_SECURITY * security_score
            + _W_TESTS * self.test_pass_rate
        )
        return self.confidence_score

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report to a JSON-safe dictionary."""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "code_hash": self.code_hash,
            "code_length_bytes": self.code_length_bytes,
            "code_line_count": self.code_line_count,
            "syntax": {
                "valid": self.syntax_valid,
                "error": self.syntax_error,
            },
            "security": {
                "risk_level": self.security_risk_level,
                "issues": self.security_issues,
                "issue_count": self.security_issue_count,
            },
            "tests": {
                "generated": self.tests_generated,
                "passed": self.tests_passed,
                "failed": self.tests_failed,
                "pass_rate": self.test_pass_rate,
                "execution_time_ms": self.test_execution_time_ms,
                "output": self.test_output,
            },
            "verdict": {
                "result": self.verdict,
                "confidence_score": self.confidence_score,
                "failure_reasons": self.failure_reasons,
            },
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_timestamp() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def compute_code_hash(source_code: str) -> str:
    """Return the SHA-256 hex digest of *source_code* (UTF-8 encoded)."""
    return hashlib.sha256(source_code.encode("utf-8")).hexdigest()
