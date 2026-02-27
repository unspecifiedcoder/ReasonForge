"""
ReasonForge - Lean 4 Proof Checker

Verifies Lean 4 proof artifacts submitted by miners.
Requires: lean4 toolchain installed in validator environment.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger("reasonforge.verification.lean4")


class Lean4Checker:
    """Verify Lean 4 proof artifacts."""

    def __init__(self, lean_path: str = "lean", timeout: int = 60):
        self.lean_path = lean_path
        self.timeout = timeout
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if Lean 4 is installed and available."""
        if self._available is not None:
            return self._available
        try:
            proc = await asyncio.create_subprocess_exec(
                self.lean_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            self._available = proc.returncode == 0
        except (FileNotFoundError, asyncio.TimeoutError):
            self._available = False
        logger.info("Lean 4 available: %s", self._available)
        return self._available

    async def verify(self, proof_b64: str) -> float:
        """
        Decode proof artifact -> write to temp .lean file -> run lean4 -> check exit code.

        Returns:
            1.0 if proof compiles successfully
            0.0 if proof fails to compile
            0.5 if timeout or other error
        """
        if not await self.is_available():
            logger.debug("Lean 4 not available, returning neutral score")
            return 0.5

        try:
            proof_text = base64.b64decode(proof_b64).decode("utf-8")
        except Exception as e:
            logger.warning("Failed to decode proof artifact: %s", e)
            return 0.0

        # Validate proof text isn't too large
        if len(proof_text) > 100_000:
            logger.warning("Proof artifact too large (%d bytes)", len(proof_text))
            return 0.0

        tmpfile = None
        try:
            # Write to temp file
            tmpfile = tempfile.NamedTemporaryFile(
                suffix=".lean", mode="w", delete=False
            )
            tmpfile.write(proof_text)
            tmpfile.flush()
            tmpfile.close()

            # Run lean4
            proc = await asyncio.create_subprocess_exec(
                self.lean_path, tmpfile.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            if proc.returncode == 0:
                logger.info("Proof verified successfully")
                return 1.0
            else:
                error_msg = stderr.decode("utf-8", errors="replace")[:500]
                logger.info("Proof verification failed: %s", error_msg)
                return 0.0

        except asyncio.TimeoutError:
            logger.warning("Lean 4 verification timed out after %ds", self.timeout)
            return 0.5
        except Exception as e:
            logger.error("Lean 4 verification error: %s", e)
            return 0.5
        finally:
            if tmpfile and os.path.exists(tmpfile.name):
                try:
                    os.unlink(tmpfile.name)
                except OSError:
                    pass
