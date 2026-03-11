"""Lean 4 formal-proof verifier for ReasonForge."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from reasonforge.translation.types import StepTranslation
from .verdict import StepVerdict, VerificationVerdict

logger = logging.getLogger(__name__)


class Lean4Verifier:
    """Verify reasoning chains by compiling Lean 4 proof terms."""

    def __init__(
        self,
        lean_toolchain: str = "leanprover/lean4:v4.8.0",
        timeout: int = 120,
    ) -> None:
        self.lean_toolchain = lean_toolchain
        self.timeout = timeout
        self._available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _check_available(self) -> bool:
        """Return *True* if ``lake`` (Lean build tool) is on ``$PATH``."""
        if self._available is None:
            self._available = shutil.which("lake") is not None
        return self._available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify_chain(
        self,
        task_id: str,
        translations: List[StepTranslation],
        context: str = "",
    ) -> VerificationVerdict:
        """Verify every translated step via the Lean 4 toolchain.

        If Lean is not installed the verdict marks every step as failed
        with an explanatory error message.
        """
        if not self._check_available():
            step_verdicts = [
                StepVerdict(
                    step_id=t.step_id,
                    verified=False,
                    error_message="Lean 4 not installed",
                    formal_representation=t.formal_representation,
                )
                for t in translations
            ]
            return VerificationVerdict(
                task_id=task_id,
                overall="FAILED",
                step_verdicts=step_verdicts,
                total_steps=len(translations),
                verified_steps=0,
                failure_points=list(step_verdicts),
                raw_output="Lean 4 toolchain not found on this machine.",
            )

        # TODO: full Lean workspace build path
        # 1. Create temp workspace
        # 2. Generate lakefile.lean via _generate_lakefile
        # 3. Write per-step .lean files via _generate_step_file
        # 4. Generate chain.lean importing all steps
        # 5. Run ``lake build`` and parse compiler output
        logger.warning(
            "Lean 4 is available but full build pipeline is not yet "
            "implemented. Returning FAILED verdict."
        )
        step_verdicts = [
            StepVerdict(
                step_id=t.step_id,
                verified=False,
                error_message="Lean 4 build pipeline not yet implemented",
                formal_representation=t.formal_representation,
            )
            for t in translations
        ]
        return VerificationVerdict(
            task_id=task_id,
            overall="FAILED",
            step_verdicts=step_verdicts,
            total_steps=len(translations),
            verified_steps=0,
            failure_points=list(step_verdicts),
            raw_output="Lean 4 build pipeline pending implementation.",
        )

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------

    def _generate_lakefile(self, workspace: Path) -> None:
        """Write a minimal ``lakefile.lean`` into *workspace*."""
        content = (
            "import Lake\n"
            "open Lake DSL\n\n"
            "package «reasonforge_proof» where\n"
            "  leanOptions := #[]\n\n"
            "@[default_target]\n"
            "lean_lib «ReasonForgeProof» where\n"
            '  srcDir := "."\n'
        )
        (workspace / "lakefile.lean").write_text(content, encoding="utf-8")

    def _generate_step_file(
        self, workspace: Path, step_id: int, lean_code: str
    ) -> None:
        """Write ``Step{N}.lean`` containing the translated proof term."""
        filename = f"Step{step_id}.lean"
        (workspace / filename).write_text(lean_code, encoding="utf-8")

    def _generate_chain_file(self, workspace: Path, step_ids: List[int]) -> None:
        """Write ``Chain.lean`` that imports every step module."""
        imports = "\n".join(f"import Step{sid}" for sid in step_ids)
        (workspace / "Chain.lean").write_text(imports + "\n", encoding="utf-8")

    async def _run_lake_build(self, workspace: Path) -> str:
        """Execute ``lake build`` inside *workspace* and return stdout+stderr."""
        proc = await asyncio.create_subprocess_exec(
            "lake",
            "build",
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=self.timeout
        )
        return (stdout or b"").decode() + (stderr or b"").decode()
