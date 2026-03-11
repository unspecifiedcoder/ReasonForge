"""Stub ZK certificate prover for the ReasonForge Proof Layer."""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from reasonforge.verification.verdict import StepVerdict

from .schema import VerificationCertificate

logger = logging.getLogger(__name__)


class CertificateProver:
    """Generates :class:`VerificationCertificate` instances.

    Currently operates in *stub mode* -- the ``zk_proof`` field is
    populated with a placeholder value.  A future release will
    integrate a real ZK proving backend (e.g. Groth16 / PLONK).
    """

    def __init__(self, params_path: Optional[str] = None) -> None:
        self.params_path = params_path
        logger.info(
            "CertificateProver initialised (ZK proving is in stub mode, "
            "params_path=%s)",
            params_path,
        )

    async def generate_certificate(
        self,
        task_id: str,
        task_hash: str,
        domain: str,
        proof_level: str,
        step_verdicts: List[StepVerdict],
        validator_count: int = 1,
    ) -> VerificationCertificate:
        """Create a verification certificate from step-level verdicts.

        Parameters
        ----------
        task_id:
            Unique identifier for the reasoning task.
        task_hash:
            SHA-256 hash of the original task content.
        domain:
            Verification domain (e.g. ``"mathematics"``).
        proof_level:
            One of ``"formal"``, ``"standard"``, ``"quick"``.
        step_verdicts:
            Per-step verification results.
        validator_count:
            Number of validators that participated in consensus.

        Returns
        -------
        VerificationCertificate
            A fully-populated certificate (stub ZK proof).
        """
        total_steps = len(step_verdicts)
        verified_steps = sum(1 for sv in step_verdicts if sv.verified)

        # Determine overall verdict
        if verified_steps == total_steps and total_steps > 0:
            overall_verdict = "VERIFIED"
        elif verified_steps > 0:
            overall_verdict = "PARTIALLY_VERIFIED"
        else:
            overall_verdict = "FAILED"

        cert = VerificationCertificate(
            task_hash=task_hash,
            domain=domain,
            proof_level=proof_level,
            total_steps=total_steps,
            verified_steps=verified_steps,
            overall_verdict=overall_verdict,
            timestamp=int(time.time()),
            validator_count=validator_count,
            zk_proof=b"STUB_ZK_PROOF",
        )
        cert.compute_certificate_id()

        logger.info(
            "Certificate generated: id=%s verdict=%s (%d/%d steps)",
            cert.certificate_id[:12],
            overall_verdict,
            verified_steps,
            total_steps,
        )
        return cert
