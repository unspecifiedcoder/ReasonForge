"""Verification certificate schema for the ReasonForge Proof Layer."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class VerificationCertificate:
    """A zero-knowledge verification certificate that attests to the
    correctness of a verified reasoning chain.

    Fields are grouped into:
      - identity / metadata
      - verification summary
      - validator consensus
      - ZK proof artefacts
      - on-chain anchoring
    """

    # ── Identity / metadata ──────────────────────
    certificate_id: str = ""
    version: int = 1
    task_hash: str = ""
    domain: str = "mathematics"
    proof_level: str = "standard"

    # ── Verification summary ─────────────────────
    total_steps: int = 0
    verified_steps: int = 0
    overall_verdict: str = "FAILED"
    timestamp: int = 0

    # ── Validator consensus ──────────────────────
    validator_count: int = 0
    validator_threshold: int = 3
    validator_commitment: str = ""

    # ── ZK proof artefacts ───────────────────────
    zk_proof: bytes = b""
    verification_key: str = ""

    # ── On-chain anchoring ───────────────────────
    chain_id: int = 0
    registry_address: str = ""
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None

    # ── Public verification URL ──────────────────
    verify_url: str = ""

    # ── Methods ──────────────────────────────────

    def compute_certificate_id(self) -> str:
        """Derive a deterministic certificate ID from core fields.

        The hash covers task_hash, overall_verdict, total_steps,
        verified_steps, and timestamp so that any change to those
        fields produces a completely different ID.
        """
        payload = (
            self.task_hash
            + self.overall_verdict
            + str(self.total_steps)
            + str(self.verified_steps)
            + str(self.timestamp)
        )
        self.certificate_id = hashlib.sha256(payload.encode()).hexdigest()
        return self.certificate_id

    def to_dict(self) -> Dict:
        """Serialise the certificate to a plain dictionary."""
        return asdict(self)
