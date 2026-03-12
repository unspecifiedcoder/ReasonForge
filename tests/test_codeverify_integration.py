# tests/test_codeverify_integration.py
"""End-to-end integration tests for the ReasonForge code verification pipeline.

Each test exercises the full chain: VerificationPipeline -> CodeVerificationReport
-> CodeVerificationCertifier -> CertificateRegistry, verifying that a real code
sample flows through sandbox execution, security scanning, test generation,
pytest execution, confidence scoring, certificate creation, and registry
look-up without any mocking.
"""

from __future__ import annotations

import pytest

from reasonforge.certificates.registry import CertificateRegistry
from reasonforge.codeverify.certificate import CodeVerificationCertifier
from reasonforge.codeverify.pipeline import VerificationPipeline


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline() -> VerificationPipeline:
    return VerificationPipeline(timeout=30)


@pytest.fixture()
def certifier() -> CodeVerificationCertifier:
    registry = CertificateRegistry()
    return CodeVerificationCertifier(registry=registry)


# ---------------------------------------------------------------------------
# Helper: a non-trivial, None-safe, well-typed sample function that the
# generated tests can exercise without unexpected failures.
# ---------------------------------------------------------------------------

CLEAN_CODE = (
    "def fibonacci(n: int) -> int:\n"
    '    """Return the *n*-th Fibonacci number.\n'
    "\n"
    "    >>> fibonacci(0)\n"
    "    0\n"
    "    >>> fibonacci(1)\n"
    "    1\n"
    "    >>> fibonacci(10)\n"
    "    55\n"
    '    """\n'
    "    if n is None:\n"
    "        return 0\n"
    "    if not isinstance(n, int) or n < 0:\n"
    "        return 0\n"
    "    a, b = 0, 1\n"
    "    for _ in range(n):\n"
    "        a, b = b, a + b\n"
    "    return a\n"
)

UNSAFE_CODE = (
    "import os\n"
    "import subprocess\n"
    "\n"
    "def run_cmd(cmd: str) -> str:\n"
    "    return subprocess.check_output(cmd, shell=True).decode()\n"
)

SYNTAX_ERROR_CODE = "def broken(\n    return 42\n"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestPipelineToCertificate:
    """Full pipeline -> certificate round-trip tests."""

    @pytest.mark.asyncio
    async def test_clean_code_produces_passing_certificate(
        self, pipeline: VerificationPipeline, certifier: CodeVerificationCertifier
    ) -> None:
        report = await pipeline.verify(CLEAN_CODE)

        assert report.syntax_valid is True
        assert report.security_risk_level in ("safe", "low")
        assert report.tests_generated >= 1
        assert report.confidence_score > 0.0
        assert report.report_id != ""
        assert report.code_hash != ""
        assert report.code_length_bytes > 0
        assert report.code_line_count > 0

        cert_id = certifier.certify_and_register(report)

        assert len(cert_id) == 64
        assert certifier.verify_certificate(cert_id) is True

        cert = certifier.get_certificate(cert_id)
        assert cert is not None
        assert cert.task_hash == report.code_hash
        assert cert.domain == "code"
        assert cert.overall_verdict == report.verdict
        assert cert.total_steps == report.tests_generated
        assert cert.verified_steps == report.tests_passed

    @pytest.mark.asyncio
    async def test_unsafe_code_produces_failing_certificate(
        self, pipeline: VerificationPipeline, certifier: CodeVerificationCertifier
    ) -> None:
        report = await pipeline.verify(UNSAFE_CODE)

        assert report.syntax_valid is True
        assert report.security_risk_level in ("high", "critical")
        assert report.verdict == "FAILED"
        assert len(report.failure_reasons) >= 1

        cert_id = certifier.certify_and_register(report)
        cert = certifier.get_certificate(cert_id)
        assert cert is not None
        assert cert.overall_verdict == "FAILED"

    @pytest.mark.asyncio
    async def test_syntax_error_produces_failed_report_no_tests(
        self, pipeline: VerificationPipeline, certifier: CodeVerificationCertifier
    ) -> None:
        report = await pipeline.verify(SYNTAX_ERROR_CODE)

        assert report.syntax_valid is False
        assert report.syntax_error is not None
        assert report.verdict == "FAILED"
        assert report.tests_generated == 0
        assert report.tests_passed == 0

        cert_id = certifier.certify_and_register(report)
        cert = certifier.get_certificate(cert_id)
        assert cert is not None
        assert cert.overall_verdict == "FAILED"
        assert cert.total_steps == 0
        assert cert.verified_steps == 0


class TestReportSerialisation:
    """Verify that the report -> dict -> certificate chain preserves data."""

    @pytest.mark.asyncio
    async def test_report_to_dict_round_trip(
        self, pipeline: VerificationPipeline
    ) -> None:
        report = await pipeline.verify(CLEAN_CODE)
        d = report.to_dict()

        assert d["report_id"] == report.report_id
        assert d["code_hash"] == report.code_hash
        assert d["syntax"]["valid"] is True
        assert d["syntax"]["error"] is None
        assert d["security"]["risk_level"] == report.security_risk_level
        assert d["security"]["issue_count"] == report.security_issue_count
        assert d["tests"]["generated"] == report.tests_generated
        assert d["tests"]["passed"] == report.tests_passed
        assert d["tests"]["failed"] == report.tests_failed
        assert d["tests"]["pass_rate"] == report.test_pass_rate
        assert d["verdict"]["result"] == report.verdict
        assert d["verdict"]["confidence_score"] == report.confidence_score


class TestConvenienceFunction:
    """Test the top-level verify_code() one-liner convenience function."""

    @pytest.mark.asyncio
    async def test_verify_code_shorthand(self) -> None:
        from reasonforge.codeverify.pipeline import verify_code

        report = await verify_code(CLEAN_CODE, timeout=30)
        assert report.syntax_valid is True
        assert report.report_id != ""
        assert report.confidence_score > 0.0


class TestDeterminism:
    """Verify that identical inputs produce identical report and cert IDs."""

    @pytest.mark.asyncio
    async def test_same_code_same_hash(self, pipeline: VerificationPipeline) -> None:
        r1 = await pipeline.verify(CLEAN_CODE)
        r2 = await pipeline.verify(CLEAN_CODE)
        assert r1.code_hash == r2.code_hash

    @pytest.mark.asyncio
    async def test_different_code_different_hash(
        self, pipeline: VerificationPipeline
    ) -> None:
        r1 = await pipeline.verify(CLEAN_CODE)
        r2 = await pipeline.verify("def noop():\n    pass\n")
        assert r1.code_hash != r2.code_hash
