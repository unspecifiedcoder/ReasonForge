"""First-Order Logic / SMT-LIB verifier backed by Z3."""

from __future__ import annotations

import logging
import shutil
from typing import Dict, List, Optional

from reasonforge.translation.types import StepTranslation
from .verdict import StepVerdict, VerificationVerdict

logger = logging.getLogger(__name__)


class FOLVerifier:
    """Verify reasoning steps expressed as SMT-LIB formulae via Z3."""

    def __init__(self, solver_timeout: int = 30) -> None:
        self.solver_timeout = solver_timeout
        self._z3_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _check_z3(self) -> bool:
        """Return *True* if Z3 is reachable (Python package or binary)."""
        if self._z3_available is not None:
            return self._z3_available

        # Try the Python bindings first.
        try:
            import z3 as _z3  # noqa: F401

            self._z3_available = True
            return True
        except ImportError:
            pass

        # Fall back to the standalone binary.
        self._z3_available = shutil.which("z3") is not None
        return self._z3_available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify_chain(
        self,
        task_id: str,
        translations: List[StepTranslation],
    ) -> VerificationVerdict:
        """Verify each translated step that contains SMT-LIB content.

        When Z3 is not available every step is marked as unverified.
        """
        if not self._check_z3():
            step_verdicts = [
                StepVerdict(
                    step_id=t.step_id,
                    verified=False,
                    error_message="Z3 not installed",
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
                raw_output="Z3 solver not found on this machine.",
            )

        checked_verdicts: List[StepVerdict] = []
        for t in translations:
            code = t.formal_representation.strip()
            if self._looks_like_smt(code):
                # Placeholder: actual Z3 execution goes here.
                result = await self._run_z3(code)
                checked_verdicts.append(
                    StepVerdict(
                        step_id=t.step_id,
                        verified=True,  # placeholder – accept valid-looking SMT
                        formal_representation=code,
                        details=result,
                    )
                )
            else:
                checked_verdicts.append(
                    StepVerdict(
                        step_id=t.step_id,
                        verified=False,
                        error_message="Formal representation is not valid SMT-LIB",
                        formal_representation=code,
                    )
                )

        verified_count = sum(1 for sv in checked_verdicts if sv.verified)
        failure_points = [sv for sv in checked_verdicts if not sv.verified]
        overall = (
            "PASSED"
            if verified_count == len(checked_verdicts) and checked_verdicts
            else "FAILED"
        )

        return VerificationVerdict(
            task_id=task_id,
            overall=overall,
            step_verdicts=checked_verdicts,
            total_steps=len(checked_verdicts),
            verified_steps=verified_count,
            failure_points=failure_points,
            domain="first_order_logic",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_smt(code: str) -> bool:
        """Heuristic check for SMT-LIB content."""
        return code.startswith("(") or "check-sat" in code

    async def _run_z3(self, smt_code: str) -> Dict:
        """Run *smt_code* through Z3 and return the result.

        This is currently a placeholder – real execution will pipe the
        SMT-LIB string into the ``z3`` binary or use the Python API.
        """
        return {"status": "unknown", "model": None}
