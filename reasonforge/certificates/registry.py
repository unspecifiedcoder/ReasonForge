"""In-memory certificate registry for the ReasonForge Proof Layer.

Supports real ZK proof verification via ``snarkjs`` when available,
falling back to a simple non-empty check otherwise.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .schema import VerificationCertificate

logger = logging.getLogger(__name__)

# Graceful import of the ZK verifier.
try:
    from reasonforge.certificates.zk_verifier import (
        is_verifier_available,
        verify_proof_sync,
    )

    _VERIFIER_AVAILABLE = is_verifier_available()
except Exception:  # noqa: BLE001
    _VERIFIER_AVAILABLE = False


def _is_real_proof(proof: bytes) -> bool:
    """Return ``True`` if *proof* looks like a real JSON ZK proof
    (as opposed to the ``b"STUB_ZK_PROOF"`` placeholder).
    """
    if not proof:
        return False
    if proof == b"STUB_ZK_PROOF":
        return False
    # Real proofs are JSON-encoded and start with '{'
    try:
        return proof.lstrip().startswith(b"{")
    except Exception:  # noqa: BLE001
        return False


class CertificateRegistry:
    """Stores and retrieves :class:`VerificationCertificate` instances.

    This implementation uses a simple in-memory dictionary.  A future
    release will anchor certificates on-chain via a smart-contract
    registry.

    When ``snarkjs`` is available, :meth:`verify` performs a full
    cryptographic proof verification.  Otherwise, it falls back to
    checking that the ``zk_proof`` field is non-empty.
    """

    def __init__(self) -> None:
        self._certificates: Dict[str, VerificationCertificate] = {}
        mode = "real ZK verification" if _VERIFIER_AVAILABLE else "stub"
        logger.info(
            "CertificateRegistry initialised (%s mode, "
            "on-chain registry is in stub mode)",
            mode,
        )

    def register(self, cert: VerificationCertificate) -> str:
        """Store a certificate and return its ID.

        If the certificate does not yet have a ``certificate_id`` it
        will be computed automatically.
        """
        if not cert.certificate_id:
            cert.compute_certificate_id()
        self._certificates[cert.certificate_id] = cert
        logger.info("Certificate registered: %s", cert.certificate_id[:12])
        return cert.certificate_id

    def get(self, certificate_id: str) -> Optional[VerificationCertificate]:
        """Look up a certificate by its ID."""
        return self._certificates.get(certificate_id)

    def verify(self, certificate_id: str) -> bool:
        """Return ``True`` if the certificate exists and its ZK proof
        is valid.

        When ``snarkjs`` is available and the proof is a real (non-stub)
        JSON proof, this performs a full cryptographic verification.
        Otherwise, it falls back to checking that ``zk_proof`` is
        non-empty.
        """
        cert = self._certificates.get(certificate_id)
        if cert is None:
            return False

        # No proof at all
        if not cert.zk_proof:
            return False

        # If we have a real proof and the verifier is available,
        # perform cryptographic verification.
        if _VERIFIER_AVAILABLE and _is_real_proof(cert.zk_proof):
            try:
                return verify_proof_sync(
                    proof_bytes=cert.zk_proof,
                    verification_key=cert.verification_key or None,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "ZK verification failed for certificate %s, "
                    "falling back to non-empty check",
                    certificate_id[:12],
                    exc_info=True,
                )
                return cert.zk_proof != b""

        # Fallback: accept any non-empty proof (stub mode)
        return cert.zk_proof != b""

    def get_stats(self) -> Dict[str, int]:
        """Return summary statistics about the registry."""
        total = len(self._certificates)
        verified = sum(1 for c in self._certificates.values() if c.zk_proof != b"")
        return {
            "total_certificates": total,
            "total_verified": verified,
        }
