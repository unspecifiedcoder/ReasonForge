"""ReasonForge Proof-Layer protocol synapse definitions.

This module defines the over-the-wire message types exchanged between
validators and miners.  When ``bittensor`` is available the classes
inherit from ``bt.Synapse``; otherwise a lightweight shim is used so
the rest of the codebase can be imported without the bittensor SDK.
"""

from __future__ import annotations

import hashlib
import json
from typing import List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Bittensor Synapse base -- with graceful fallback
# ---------------------------------------------------------------------------

try:
    import bittensor as bt

    SynapseBase = bt.Synapse
except ImportError:

    class _SynapseShim(BaseModel):  # type: ignore[no-redef]
        """Minimal stand-in for ``bt.Synapse`` when bittensor is not installed."""

        class Config:
            arbitrary_types_allowed = True

        dendrite: Optional[object] = None
        axon: Optional[object] = None

        def deserialize(self) -> dict:
            return self.model_dump()

        def to_headers(self) -> dict:
            return {}

        def body_hash(self) -> str:
            raw = json.dumps(self.model_dump(), sort_keys=True, default=str)
            return hashlib.sha256(raw.encode()).hexdigest()

    SynapseBase = _SynapseShim  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# TranslationTask
# ---------------------------------------------------------------------------


class TranslationTask(SynapseBase):  # type: ignore[valid-type, misc]
    """A reasoning-chain translation task sent from a validator to a miner.

    **Immutable fields** are set by the validator and must not be
    modified by the miner.  **Mutable fields** are populated by the
    miner before returning the synapse.
    """

    # ── Immutable (set by validator) ─────────────
    task_id: str = ""
    original_query: str = ""
    reasoning_chain: Optional[List[dict]] = None
    domain: str = "mathematics"
    difficulty: int = Field(default=5, ge=1, le=10)
    proof_level: str = "standard"
    timeout_seconds: int = 300

    # ── Mutable (set by miner) ───────────────────
    translations: Optional[List[dict]] = None
    compilation_status: Optional[str] = None
    full_proof_artifact: Optional[str] = None
    translation_time_ms: Optional[int] = None
    submission_hash: Optional[str] = None

    required_hash_fields: List[str] = [
        "task_id",
        "original_query",
        "domain",
        "difficulty",
        "proof_level",
    ]

    def deserialize(self) -> dict:
        """Return the miner-produced payload."""
        return {
            "task_id": self.task_id,
            "translations": self.translations,
            "compilation_status": self.compilation_status,
            "full_proof_artifact": self.full_proof_artifact,
            "translation_time_ms": self.translation_time_ms,
            "submission_hash": self.submission_hash,
        }


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


class VerificationResult(SynapseBase):  # type: ignore[valid-type, misc]
    """Epoch-level verification results sent back to miners."""

    epoch_id: int = 0
    miner_uid: int = 0
    tasks_translated: int = 0
    steps_compiled: int = 0
    steps_total: int = 0
    compilation_rate: float = 0.0
    epoch_score: float = 0.0
    rank: int = 0
    tao_earned: float = 0.0

    required_hash_fields: List[str] = [
        "epoch_id",
        "miner_uid",
    ]

    def deserialize(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "miner_uid": self.miner_uid,
            "tasks_translated": self.tasks_translated,
            "steps_compiled": self.steps_compiled,
            "steps_total": self.steps_total,
            "compilation_rate": self.compilation_rate,
            "epoch_score": self.epoch_score,
            "rank": self.rank,
            "tao_earned": self.tao_earned,
        }


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------


class HealthCheck(SynapseBase):  # type: ignore[valid-type, misc]
    """Simple health-check synapse used for liveness probes."""

    status: Optional[str] = None
    supported_domains: Optional[List[str]] = None
    model_info: Optional[str] = None
    version: Optional[str] = None

    required_hash_fields: List[str] = []

    def deserialize(self) -> dict:
        return {
            "status": self.status,
            "supported_domains": self.supported_domains,
            "model_info": self.model_info,
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def create_translation_task(
    task_id: str,
    original_query: str,
    reasoning_chain: List[dict],
    domain: str = "mathematics",
    difficulty: int = 5,
    proof_level: str = "standard",
    timeout_seconds: int = 300,
) -> TranslationTask:
    """Convenience factory for building a :class:`TranslationTask`."""
    return TranslationTask(
        task_id=task_id,
        original_query=original_query,
        reasoning_chain=reasoning_chain,
        domain=domain,
        difficulty=difficulty,
        proof_level=proof_level,
        timeout_seconds=timeout_seconds,
    )


def verify_submission_hash(synapse: TranslationTask) -> bool:
    """Check whether the miner's ``submission_hash`` matches the
    translations payload.

    Returns ``False`` if either field is missing.
    """
    if synapse.translations is None or synapse.submission_hash is None:
        return False
    payload = json.dumps(synapse.translations, sort_keys=True, default=str)
    expected = hashlib.sha256(payload.encode()).hexdigest()
    return expected == synapse.submission_hash
