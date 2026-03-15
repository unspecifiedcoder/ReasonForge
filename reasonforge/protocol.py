"""
ReasonForge - Bittensor Wire Protocol

Defines all Synapse subclasses for validator<->miner communication.
Gracefully degrades when bittensor is not installed (for testing).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Synapse base: use real bt.Synapse if available, else a Pydantic shim
# ──────────────────────────────────────────────


class _DendriteMetadata(BaseModel):
    hotkey: str = ""
    ip: str = ""
    port: int = 0


class _AxonMetadata(BaseModel):
    hotkey: str = ""
    ip: str = ""
    port: int = 0


class _SynapseShim(BaseModel):
    """Minimal Synapse shim for environments without bittensor."""

    class Config:
        arbitrary_types_allowed = True

    dendrite: Optional[_DendriteMetadata] = None
    axon: Optional[_AxonMetadata] = None

    def deserialize(self) -> dict:
        return self.model_dump()

    def to_headers(self) -> dict:
        return {}

    def body_hash(self) -> str:
        data = json.dumps(
            {
                k: v
                for k, v in self.model_dump().items()
                if k in getattr(self, "required_hash_fields", [])
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()


try:
    import bittensor as bt

    SynapseBase = bt.Synapse
except ImportError:
    SynapseBase = _SynapseShim  # type: ignore[misc, assignment]


# ──────────────────────────────────────────────
# Synapse Definitions
# ──────────────────────────────────────────────


class ReasoningTask(SynapseBase):  # type: ignore[valid-type, misc]
    """Validator -> Miner: Here is a reasoning task to solve."""

    # -- Immutable fields (set by validator, read by miner) --
    task_id: str = ""
    problem: str = ""
    domain: str = ""
    difficulty: int = Field(default=5, ge=1, le=10)
    timeout_seconds: int = 300
    context: Optional[str] = None
    constraints: Optional[str] = None

    # -- Mutable fields (filled by miner, read back by validator) --
    reasoning_steps: Optional[List[dict]] = None
    final_answer: Optional[str] = None
    proof_status: Optional[str] = None
    proof_artifact: Optional[str] = None
    code_artifact: Optional[str] = None
    time_taken_ms: Optional[int] = None
    submission_hash: Optional[str] = None

    required_hash_fields: List[str] = ["task_id", "problem", "domain", "difficulty"]

    def deserialize(self) -> dict:
        return {
            "task_id": self.task_id,
            "steps": self.reasoning_steps or [],
            "final_answer": self.final_answer,
            "proof_status": self.proof_status,
            "proof_artifact": self.proof_artifact,
            "code_artifact": self.code_artifact,
            "time_taken_ms": self.time_taken_ms,
            "submission_hash": self.submission_hash,
        }

    def compute_submission_hash(self) -> str:
        """Compute SHA-256 hash of steps + final_answer for integrity check."""
        steps_json = json.dumps(self.reasoning_steps or [], sort_keys=True)
        payload = f"{self.task_id}:{steps_json}:{self.final_answer or ''}"
        return hashlib.sha256(payload.encode()).hexdigest()


class HealthCheck(SynapseBase):  # type: ignore[valid-type, misc]
    """Validator -> Miner: Are you alive and what are your capabilities?"""

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


class TaskResult(SynapseBase):  # type: ignore[valid-type, misc]
    """Validator -> Miner: Here are your scores for a batch of tasks (informational)."""

    epoch_id: int = 0
    miner_uid: int = 0
    scores: Optional[List[dict]] = None
    s_epoch: Optional[float] = None
    rank: Optional[int] = None
    total_tao: Optional[float] = None

    required_hash_fields: List[str] = ["epoch_id", "miner_uid"]

    def deserialize(self) -> dict:
        return {
            "epoch_id": self.epoch_id,
            "scores": self.scores,
            "s_epoch": self.s_epoch,
            "rank": self.rank,
        }


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────


def verify_submission_hash(synapse: ReasoningTask) -> bool:
    """Verify the integrity of a miner's submission."""
    if not synapse.submission_hash:
        return False
    expected = synapse.compute_submission_hash()
    return synapse.submission_hash == expected


def create_reasoning_task(
    task_id: Optional[str] = None,
    problem: str = "",
    domain: str = "mathematics",
    difficulty: int = 5,
    timeout_seconds: int = 300,
    context: Optional[str] = None,
    constraints: Optional[str] = None,
) -> ReasoningTask:
    """Factory for creating a ReasoningTask synapse."""
    return ReasoningTask(
        task_id=task_id or str(uuid.uuid4()),
        problem=problem,
        domain=domain,
        difficulty=difficulty,
        timeout_seconds=timeout_seconds,
        context=context,
        constraints=constraints,
    )
