# reasonforge/codeverify/api.py
#
# REST API layer for the ReasonForge code verification service.
#
# Exposes a FastAPI application with endpoints for verifying Python source
# code (syntax, security, and generated-test execution), health checks, and
# service metadata.  The app is created via a ``create_app()`` factory so
# that tests and alternative configurations can instantiate fresh instances.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

VERSION: str = "0.1.0"
_MAX_SOURCE_BYTES: int = 100 * 1024  # 100 KB
_START_TIME: float = time.monotonic()

# ---------------------------------------------------------------------------
# Request / response models (dataclass fallback when pydantic is absent)
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel

    class VerifyRequest(BaseModel):
        """Payload for the ``POST /verify`` endpoint."""

        source_code: str
        timeout: int = 30
        skip_security: bool = False
        skip_tests: bool = False

    class VerifyResponse(BaseModel):
        """Structured result returned by ``POST /verify``."""

        report_id: str
        verdict: str
        confidence_score: float
        syntax_valid: bool
        security_risk_level: str
        security_issue_count: int
        tests_generated: int
        tests_passed: int
        tests_failed: int
        test_pass_rate: float
        failure_reasons: List[str]
        code_hash: str
        timestamp: str

    class HealthResponse(BaseModel):
        """Payload returned by ``GET /health``."""

        status: str
        version: str
        uptime_seconds: float

    class ErrorResponse(BaseModel):
        """Generic error envelope."""

        error: str
        detail: str

except ImportError:

    @dataclass
    class VerifyRequest:  # type: ignore[no-redef]
        """Payload for the ``POST /verify`` endpoint."""

        source_code: str
        timeout: int = 30
        skip_security: bool = False
        skip_tests: bool = False

    @dataclass
    class VerifyResponse:  # type: ignore[no-redef]
        """Structured result returned by ``POST /verify``."""

        report_id: str = ""
        verdict: str = ""
        confidence_score: float = 0.0
        syntax_valid: bool = True
        security_risk_level: str = "safe"
        security_issue_count: int = 0
        tests_generated: int = 0
        tests_passed: int = 0
        tests_failed: int = 0
        test_pass_rate: float = 0.0
        failure_reasons: List[str] = field(default_factory=list)
        code_hash: str = ""
        timestamp: str = ""

    @dataclass
    class HealthResponse:  # type: ignore[no-redef]
        """Payload returned by ``GET /health``."""

        status: str = "healthy"
        version: str = ""
        uptime_seconds: float = 0.0

    @dataclass
    class ErrorResponse:  # type: ignore[no-redef]
        """Generic error envelope."""

        error: str = ""
        detail: str = ""


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from reasonforge.codeverify.pipeline import VerificationPipeline

    def create_app() -> FastAPI:
        """Build and return a configured :class:`FastAPI` application."""

        application = FastAPI(
            title="ReasonForge Code Verification API",
            version=VERSION,
            description=(
                "Verifies Python source code via syntax checking, "
                "AST-based security scanning, and auto-generated test execution."
            ),
        )

        pipeline = VerificationPipeline()

        # -- POST /verify ---------------------------------------------------

        @application.post(
            "/verify",
            response_model=VerifyResponse,
            responses={
                400: {"model": ErrorResponse},
                500: {"model": ErrorResponse},
            },
        )
        async def verify(request: VerifyRequest) -> Any:
            """Verify a snippet of Python source code."""

            # --- input validation ------------------------------------------
            if not request.source_code or not request.source_code.strip():
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "empty_source",
                        "detail": "source_code must not be empty.",
                    },
                )

            if len(request.source_code.encode("utf-8")) > _MAX_SOURCE_BYTES:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "source_too_large",
                        "detail": (
                            f"source_code exceeds the {_MAX_SOURCE_BYTES} byte limit."
                        ),
                    },
                )

            # --- run pipeline ----------------------------------------------
            #
            # The pipeline constructor accepts ``timeout`` while
            # ``skip_security`` / ``skip_tests`` are not part of the
            # pipeline interface.  We construct a per-request pipeline
            # when a non-default timeout is provided and handle the
            # skip flags by zeroing the corresponding report fields
            # after the pipeline completes.
            try:
                request_pipeline: VerificationPipeline
                if request.timeout != 30:
                    request_pipeline = VerificationPipeline(
                        timeout=request.timeout,
                    )
                else:
                    request_pipeline = pipeline

                report = await request_pipeline.verify(request.source_code)

                # Respect skip flags by resetting sections the caller
                # chose to omit.
                if request.skip_security:
                    report.security_risk_level = "skipped"
                    report.security_issue_count = 0
                    report.security_issues = []

                if request.skip_tests:
                    report.tests_generated = 0
                    report.tests_passed = 0
                    report.tests_failed = 0
                    report.test_pass_rate = 0.0

            except Exception:
                logger.exception("Verification pipeline failed")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "verification_failed",
                        "detail": "An internal error occurred during verification.",
                    },
                )

            return VerifyResponse(
                report_id=report.report_id,
                verdict=report.verdict,
                confidence_score=report.confidence_score,
                syntax_valid=report.syntax_valid,
                security_risk_level=report.security_risk_level,
                security_issue_count=report.security_issue_count,
                tests_generated=report.tests_generated,
                tests_passed=report.tests_passed,
                tests_failed=report.tests_failed,
                test_pass_rate=report.test_pass_rate,
                failure_reasons=report.failure_reasons,
                code_hash=report.code_hash,
                timestamp=report.timestamp,
            )

        # -- GET /health ----------------------------------------------------

        @application.get("/health", response_model=HealthResponse)
        async def health() -> HealthResponse:
            """Return service health status and uptime."""
            uptime = time.monotonic() - _START_TIME
            return HealthResponse(
                status="healthy",
                version=VERSION,
                uptime_seconds=round(uptime, 2),
            )

        # -- GET / ----------------------------------------------------------

        @application.get("/")
        async def root() -> Dict[str, str]:
            """Return basic service metadata."""
            return {
                "name": "ReasonForge Code Verification API",
                "version": VERSION,
                "description": (
                    "Verifies Python source code via syntax checking, "
                    "security scanning, and auto-generated test execution."
                ),
            }

        return application

except ImportError:

    def create_app() -> Any:  # type: ignore[misc]
        """Raise when FastAPI is not installed."""
        raise ImportError(
            "FastAPI is required to run the ReasonForge verification API. "
            "Install it with:  pip install fastapi uvicorn"
        )


# ---------------------------------------------------------------------------
# Module-level app instance (for ``uvicorn reasonforge.codeverify.api:app``)
# ---------------------------------------------------------------------------

app: Optional[FastAPI] = None  # type: ignore[assignment]
try:
    app = create_app()
except (ImportError, Exception):
    app = None

# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to run the API server. "
            "Install it with:  pip install uvicorn"
        ) from exc

    if app is None:
        raise RuntimeError(
            "Could not create the FastAPI application. "
            "Ensure fastapi and its dependencies are installed."
        )

    uvicorn.run(app, host="0.0.0.0", port=8000)
