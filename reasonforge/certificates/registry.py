"""In-memory certificate registry for the ReasonForge Proof Layer."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .schema import VerificationCertificate

logger = logging.getLogger(__name__)


class CertificateRegistry:
    """Stores and retrieves :class:`VerificationCertificate` instances.

    This implementation uses a simple in-memory dictionary.  A future
    release will anchor certificates on-chain via a smart-contract
    registry.
    """

    def __init__(self) -> None:
        self._certificates: Dict[str, VerificationCertificate] = {}
        logger.info(
            "CertificateRegistry initialised (on-chain registry is in stub mode)"
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
        """Return ``True`` if the certificate exists and carries a ZK proof."""
        cert = self._certificates.get(certificate_id)
        if cert is None:
            return False
        return cert.zk_proof != b""

    def get_stats(self) -> Dict:
        """Return summary statistics about the registry."""
        total = len(self._certificates)
        verified = sum(1 for c in self._certificates.values() if c.zk_proof != b"")
        return {
            "total_certificates": total,
            "total_verified": verified,
        }
