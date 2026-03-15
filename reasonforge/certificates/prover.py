"""ZK certificate prover for the ReasonForge Proof Layer.

Generates :class:`VerificationCertificate` instances with real Groth16
ZK proofs when ``snarkjs`` and the circuit build artefacts are
available, falling back to stub mode otherwise.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from reasonforge.verification.verdict import StepVerdict

from .schema import VerificationCertificate

logger = logging.getLogger(__name__)

# Graceful import of the ZK prover -- falls back if snarkjs/circom
# are not installed or build artefacts are missing.
try:
    from reasonforge.certificates.zk_prover import (
        generate_proof,
        is_zk_available,
    )

    ZK_AVAILABLE = is_zk_available()
except Exception:  # noqa: BLE001
    ZK_AVAILABLE = False

if not ZK_AVAILABLE:
    logger.info(
        "ZK proving is unavailable (snarkjs or build artefacts not found). "
        "CertificateProver will operate in stub mode."
    )


class CertificateProver:
    """Generates :class:`VerificationCertificate` instances.

    When the ZK proving pipeline is available (``snarkjs`` installed
    and circuit build artefacts present), certificates carry a real
    Groth16 proof.  Otherwise, the ``zk_proof`` field is populated
    with a stub placeholder.
    """

    def __init__(self, params_path: Optional[str] = None) -> None:
        self.params_path = params_path
        mode = "real ZK" if ZK_AVAILABLE else "stub"
        logger.info(
            "CertificateProver initialised (%s mode, params_path=%s)",
            mode,
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
            A fully-populated certificate with a ZK proof (real or
            stub depending on availability).
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

        timestamp = int(time.time())

        cert = VerificationCertificate(
            task_hash=task_hash,
            domain=domain,
            proof_level=proof_level,
            total_steps=total_steps,
            verified_steps=verified_steps,
            overall_verdict=overall_verdict,
            timestamp=timestamp,
            validator_count=validator_count,
        )

        # Attempt real ZK proof generation
        if ZK_AVAILABLE:
            try:
                binary_verdicts = [1 if sv.verified else 0 for sv in step_verdicts]
                proof_bytes, vk = await generate_proof(
                    task_hash=task_hash,
                    overall_verdict=overall_verdict,
                    total_steps=total_steps,
                    verified_steps=verified_steps,
                    timestamp=timestamp,
                    step_verdicts=binary_verdicts,
                )
                cert.zk_proof = proof_bytes
                cert.verification_key = vk
                logger.info("Real ZK proof generated for certificate")
            except Exception:  # noqa: BLE001
                logger.warning(
                    "ZK proof generation failed, falling back to stub",
                    exc_info=True,
                )
                cert.zk_proof = b"STUB_ZK_PROOF"
        else:
            cert.zk_proof = b"STUB_ZK_PROOF"
            logger.warning("ZK proving unavailable, using stub proof")

        cert.compute_certificate_id()

        logger.info(
            "Certificate generated: id=%s verdict=%s (%d/%d steps)",
            cert.certificate_id[:12],
            overall_verdict,
            verified_steps,
            total_steps,
        )
        return cert
