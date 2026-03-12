# reasonforge/codeverify/pipeline.py
#
# Verification pipeline orchestrator for ReasonForge code verification.
#
# Chains sandbox execution, security scanning, test generation, and test
# execution into a single ``verify()`` call that returns a fully populated
# ``CodeVerificationReport``.

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from reasonforge.codeverify.report import (
    CodeVerificationReport,
    compute_code_hash,
    make_timestamp,
)
from reasonforge.codeverify.sandbox import SandboxExecutor
from reasonforge.codeverify.security_scanner import SecurityScanner
from reasonforge.codeverify.test_generator import TestGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pytest output parsing helpers
# ---------------------------------------------------------------------------

_PASSED_RE = re.compile(r"(\d+)\s+passed")
_FAILED_RE = re.compile(r"(\d+)\s+failed")


def _parse_pytest_counts(output: str) -> Dict[str, int]:
    """Extract *passed* and *failed* counts from pytest summary output.

    Scans every line for patterns like ``5 passed`` or ``2 failed`` and
    returns the last match found for each (pytest prints its summary at the
    end).
    """
    passed = 0
    failed = 0

    for line in output.splitlines():
        m_passed = _PASSED_RE.search(line)
        if m_passed:
            passed = int(m_passed.group(1))

        m_failed = _FAILED_RE.search(line)
        if m_failed:
            failed = int(m_failed.group(1))

    return {"passed": passed, "failed": failed}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class VerificationPipeline:
    """Orchestrates the full ReasonForge code-verification workflow.

    Parameters
    ----------
    timeout:
        Maximum wall-clock seconds for sandbox subprocess execution.
    blocked_imports:
        Optional list of module names to block inside the sandbox.
        Defaults to the sandbox's built-in block list.
    """

    def __init__(
        self,
        timeout: int = 30,
        blocked_imports: Optional[List[str]] = None,
    ) -> None:
        self.sandbox = SandboxExecutor(timeout=timeout, blocked_imports=blocked_imports)
        self.scanner = SecurityScanner()
        self.test_gen = TestGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(self, source_code: str) -> CodeVerificationReport:
        """Run the full verification pipeline on *source_code*.

        Steps
        -----
        1. Syntax check via :func:`compile`.
        2. Security scan via :class:`SecurityScanner`.
        3. Test generation via :class:`TestGenerator`.
        4. Test execution via :class:`SandboxExecutor` (pytest).
        5. Verdict computation.
        6. Confidence score and report-ID generation.
        """
        timestamp = make_timestamp()
        code_hash = compute_code_hash(source_code)
        code_length_bytes = len(source_code.encode("utf-8"))
        code_line_count = len(source_code.splitlines())

        report = CodeVerificationReport(
            timestamp=timestamp,
            code_hash=code_hash,
            code_length_bytes=code_length_bytes,
            code_line_count=code_line_count,
        )

        failure_reasons: List[str] = []

        # ── Step 1: Syntax check ──────────────────────────────────────
        try:
            compile(source_code, "<input>", "exec")
            report.syntax_valid = True
        except SyntaxError as exc:
            report.syntax_valid = False
            report.syntax_error = str(exc)
            failure_reasons.append(f"Syntax error: {exc}")

            report.verdict = "FAILED"
            report.failure_reasons = failure_reasons
            report.compute_confidence()
            report.compute_report_id()
            return report

        # ── Step 2: Security scan ─────────────────────────────────────
        security_report = self.scanner.scan(source_code)
        report.security_risk_level = security_report.risk_level
        report.security_issue_count = security_report.total_issues
        report.security_issues = [
            _vuln_to_dict(v) for v in security_report.vulnerabilities
        ]

        if security_report.risk_level in ("critical", "high"):
            failure_reasons.append(
                f"Security risk level is {security_report.risk_level} "
                f"({security_report.total_issues} issue(s) found)"
            )

        # ── Step 3: Generate tests ────────────────────────────────────
        try:
            suite = self.test_gen.generate(source_code)
            generated_test_code = suite.test_code
            report.tests_generated = suite.test_count
        except Exception as exc:
            logger.warning("Test generation failed: %s", exc)
            generated_test_code = ""
            report.tests_generated = 0

        # ── Step 4: Execute tests ─────────────────────────────────────
        if report.tests_generated > 0 and generated_test_code:
            try:
                sandbox_result = await self.sandbox.run_pytest(
                    source_code, generated_test_code
                )
                report.test_output = sandbox_result.stdout + sandbox_result.stderr
                report.test_execution_time_ms = sandbox_result.execution_time_ms

                counts = _parse_pytest_counts(report.test_output)
                report.tests_passed = counts["passed"]
                report.tests_failed = counts["failed"]

                total_run = report.tests_passed + report.tests_failed
                if total_run > 0:
                    report.test_pass_rate = report.tests_passed / total_run
                else:
                    report.test_pass_rate = 0.0

            except Exception as exc:
                logger.warning("Test execution failed: %s", exc)
                report.test_output = str(exc)
                report.test_pass_rate = 0.0
        else:
            # No tests to run — treat as neutral (pass rate 0.0).
            report.test_pass_rate = 0.0

        if report.test_pass_rate < 0.8 and report.tests_generated > 0:
            failure_reasons.append(
                f"Test pass rate {report.test_pass_rate:.0%} is below 80% threshold"
            )

        # ── Step 5: Compute verdict ───────────────────────────────────
        try:
            if failure_reasons:
                report.verdict = "FAILED"
            else:
                report.verdict = "PASSED"
        except Exception as exc:
            report.verdict = "ERROR"
            failure_reasons.append(f"Unexpected error: {exc}")

        report.failure_reasons = failure_reasons

        # ── Step 6: Confidence & report ID ────────────────────────────
        report.compute_confidence()
        report.compute_report_id()
        return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _vuln_to_dict(vuln: Any) -> Dict[str, Any]:
    """Convert a :class:`~reasonforge.codeverify.security_scanner.Vulnerability`
    to a plain dictionary with the fields expected by the report schema.
    """
    return {
        "rule_id": vuln.rule_id,
        "severity": vuln.severity,
        "line_number": vuln.line_number,
        "description": vuln.description,
        "category": vuln.category,
    }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


async def verify_code(
    source_code: str,
    timeout: int = 30,
) -> CodeVerificationReport:
    """One-liner convenience: create a pipeline and verify *source_code*.

    Parameters
    ----------
    source_code:
        Python source code to verify.
    timeout:
        Maximum wall-clock seconds for sandbox execution.

    Returns
    -------
    CodeVerificationReport
        The fully populated verification report.
    """
    pipeline = VerificationPipeline(timeout=timeout)
    return await pipeline.verify(source_code)
