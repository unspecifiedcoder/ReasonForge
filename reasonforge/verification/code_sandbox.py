"""
ReasonForge - Code Sandbox

Run miner code submissions in an isolated Docker container.
Prevents: filesystem access, network access, fork bombs, resource exhaustion.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from typing import Optional

logger = logging.getLogger("reasonforge.verification.sandbox")


class CodeSandbox:
    """Execute code in a sandboxed Docker container."""

    def __init__(
        self,
        image: str = "reasonforge-sandbox:latest",
        timeout: int = 30,
        memory_limit: str = "256m",
        cpu_quota: int = 50000,
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_period = 100000
        self.cpu_quota = cpu_quota
        self._client = None
        self._available: Optional[bool] = None

    def _get_client(self):
        """Lazy-initialize Docker client."""
        if self._client is None:
            try:
                import docker

                self._client = docker.from_env()
            except ImportError:
                raise ImportError(
                    "docker package not installed. Install with: pip install docker>=7.0.0"
                )
            except Exception as e:
                logger.error("Docker not available: %s", e)
                raise
        return self._client

    async def is_available(self) -> bool:
        """Check if Docker is available."""
        if self._available is not None:
            return self._available
        try:
            client = self._get_client()
            client.ping()
            self._available = True
        except Exception:
            self._available = False
        return self._available

    async def run_tests(self, code_b64: str) -> float:
        """
        Execute code in sandbox and run any included test cases.

        Returns:
            Float 0.0-1.0 based on test pass rate.
        """
        try:
            code = base64.b64decode(code_b64).decode("utf-8")
        except Exception:
            return 0.0

        # Validate code size
        if len(code) > 500_000:
            logger.warning("Code artifact too large")
            return 0.0

        # Security check: reject obviously dangerous code
        if self._contains_dangerous_patterns(code):
            logger.warning("Code contains dangerous patterns")
            return 0.0

        try:
            client = self._get_client()

            # Run in isolated container
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                None,
                lambda: client.containers.run(
                    self.image,
                    command=["python3", "-c", code],
                    detach=True,
                    mem_limit=self.memory_limit,
                    cpu_period=self.cpu_period,
                    cpu_quota=self.cpu_quota,
                    network_disabled=True,
                    read_only=True,
                    tmpfs={"/tmp": "size=64m"},
                ),
            )

            try:
                result = await loop.run_in_executor(
                    None, lambda: container.wait(timeout=self.timeout)
                )
                logs = await loop.run_in_executor(
                    None, lambda: container.logs().decode("utf-8", errors="replace")
                )

                if result["StatusCode"] == 0:
                    return self._parse_test_results(logs)
                return 0.0
            finally:
                await loop.run_in_executor(None, lambda: container.remove(force=True))

        except Exception as e:
            logger.error("Sandbox execution failed: %s", e)
            return 0.0

    async def lint(self, code_b64: str) -> float:
        """Run static analysis on code. Returns quality score 0.0-1.0."""
        try:
            code = base64.b64decode(code_b64).decode("utf-8")
        except Exception:
            return 0.0

        # Basic static analysis without Docker
        score = 0.5  # Base score
        lines = code.split("\n")

        # Check for docstrings
        if '"""' in code or "'''" in code:
            score += 0.1

        # Check for type hints
        if ":" in code and "->" in code:
            score += 0.1

        # Check for error handling
        if "try:" in code or "except" in code:
            score += 0.1

        # Penalize very short code
        if len(lines) < 5:
            score -= 0.2

        # Penalize no functions/classes
        if "def " not in code and "class " not in code:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _contains_dangerous_patterns(self, code: str) -> bool:
        """Check for obviously dangerous code patterns."""
        dangerous = [
            r"import\s+subprocess",
            r"import\s+shutil",
            r"os\.system\(",
            r"os\.exec",
            r"os\.popen\(",
            r"__import__\(",
            r"eval\(",
            r"exec\(",
            r"open\(.*/etc/",
            r"fork\(\)",
            r"import\s+socket",
        ]
        for pattern in dangerous:
            if re.search(pattern, code):
                return True
        return False

    def _parse_test_results(self, logs: str) -> float:
        """Parse test output for pass/fail counts."""
        # Try pytest format
        match = re.search(r"(\d+)\s+passed", logs)
        if match:
            passed = int(match.group(1))
            failed_match = re.search(r"(\d+)\s+failed", logs)
            failed = int(failed_match.group(1)) if failed_match else 0
            total = passed + failed
            return passed / total if total > 0 else 0.0

        # Try unittest format
        match = re.search(r"Ran\s+(\d+)\s+test", logs)
        if match:
            total = int(match.group(1))
            if "OK" in logs:
                return 1.0
            fail_match = re.search(r"failures=(\d+)", logs)
            failures = int(fail_match.group(1)) if fail_match else 0
            return max(0.0, (total - failures) / total) if total > 0 else 0.0

        # Fallback: if code ran without error, partial credit
        return 0.7
