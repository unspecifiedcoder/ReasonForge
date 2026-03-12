# reasonforge/codeverify/certificate.py
#
# Bridges code verification reports to the existing ZK certificate
# infrastructure.  Converts a CodeVerificationReport into a
# VerificationCertificate and optionally registers it in a
# CertificateRegistry for later look-up and verification.

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from reasonforge.certificates.prover import CertificateProver
from reasonforge.certificates.registry import CertificateRegistry
from reasonforge.certificates.schema import VerificationCertificate
from reasonforge.codeverify.report import CodeVerificationReport

logger = logging.getLogger(__name__)


def _parse_iso_timestamp(iso_str: str) -> int:
    """Parse an ISO 8601 timestamp string and return a Unix epoch integer.

    Handles the common trailing ``Z`` shorthand for UTC by replacing it
    with the full ``+00:00`` offset that :meth:`datetime.fromisoformat`
    expects on Python 3.9.
    """
    normalised = iso_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalised)
    return int(dt.timestamp())


class CodeVerificationCertifier:
    """Convert :class:`CodeVerificationReport` instances into
    :class:`VerificationCertificate` objects and manage their lifecycle
    through the ZK certificate infrastructure.
    """

    def __init__(self, registry: Optional[CertificateRegistry] = None) -> None:
        self._registry = registry if registry is not None else CertificateRegistry()
        self._prover = CertificateProver()
        logger.info("CodeVerificationCertifier initialised")

    # ------------------------------------------------------------------
    # Certificate creation
    # ------------------------------------------------------------------

    def create_certificate(
        self, report: CodeVerificationReport
    ) -> VerificationCertificate:
        """Map a :class:`CodeVerificationReport` onto a
        :class:`VerificationCertificate`.

        The mapping translates code-verification concepts (tests, pass
        rates, code hashes) into the generic certificate schema used by
        the ReasonForge proof layer.

        Parameters
        ----------
        report:
            A fully-populated code verification report.

        Returns
        -------
        VerificationCertificate
            A certificate with a computed ``certificate_id``.
        """
        cert = VerificationCertificate(
            task_hash=report.code_hash,
            domain="code",
            proof_level="standard",
            total_steps=report.tests_generated,
            verified_steps=report.tests_passed,
            overall_verdict=report.verdict,
            timestamp=_parse_iso_timestamp(report.timestamp),
            validator_count=1,
            zk_proof=b"STUB_ZK_PROOF",
        )
        cert.compute_certificate_id()

        logger.info(
            "Certificate created from report: cert_id=%s verdict=%s (%d/%d tests)",
            cert.certificate_id[:12],
            cert.overall_verdict,
            cert.verified_steps,
            cert.total_steps,
        )
        return cert

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def certify_and_register(self, report: CodeVerificationReport) -> str:
        """Create a certificate from *report*, register it, and return
        the ``certificate_id``.

        Parameters
        ----------
        report:
            A fully-populated code verification report.

        Returns
        -------
        str
            The deterministic certificate ID assigned to the new
            certificate.
        """
        cert = self.create_certificate(report)
        certificate_id = self._registry.register(cert)
        logger.info("Certificate registered: %s", certificate_id[:12])
        return certificate_id

    # ------------------------------------------------------------------
    # Look-up / verification
    # ------------------------------------------------------------------

    def verify_certificate(self, certificate_id: str) -> bool:
        """Check whether the certificate identified by *certificate_id*
        is present in the registry and carries a valid ZK proof.

        Returns ``False`` if no registry is available.
        """
        if self._registry is None:
            return False
        return self._registry.verify(certificate_id)

    def get_certificate(self, certificate_id: str) -> Optional[VerificationCertificate]:
        """Retrieve a previously-registered certificate by its ID.

        Returns ``None`` if the registry is unavailable or the
        certificate is not found.
        """
        if self._registry is None:
            return None
        return self._registry.get(certificate_id)
