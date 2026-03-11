"""Verdict data-classes for the ReasonForge verification pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StepVerdict:
    """Result of verifying a single reasoning step."""

    step_id: int
    verified: bool
    error_message: Optional[str] = None
    formal_representation: str = ""
    details: Dict = field(default_factory=dict)


@dataclass
class FailureReport:
    """Detailed report about a verification failure."""

    failed_step_id: int
    original_reasoning: str = ""
    formal_translation: str = ""
    verification_error: str = ""
    suggested_fix: Optional[str] = None
    cascade_impact: List[int] = field(default_factory=list)
    last_valid_step: int = 0
    partial_correctness: float = 0.0


@dataclass
class VerificationVerdict:
    """Aggregated verification result for an entire reasoning chain."""

    task_id: str
    overall: str = "FAILED"
    step_verdicts: List[StepVerdict] = field(default_factory=list)
    total_steps: int = 0
    verified_steps: int = 0
    failure_points: List[StepVerdict] = field(default_factory=list)
    failure_report: Optional[FailureReport] = None
    domain: str = "mathematics"
    proof_level: str = "standard"
    verification_time_ms: int = 0
    raw_output: Optional[str] = None
    translator_uids: List[int] = field(default_factory=list)
    verdict_hash: str = ""

    def compute_verdict_hash(self) -> str:
        """Compute a deterministic SHA-256 hash over the core verdict fields."""
        payload = {
            "task_id": self.task_id,
            "overall": self.overall,
            "steps": [
                {"id": sv.step_id, "verified": sv.verified} for sv in self.step_verdicts
            ],
        }
        serialised = json.dumps(payload, sort_keys=True)
        self.verdict_hash = hashlib.sha256(serialised.encode()).hexdigest()
        return self.verdict_hash
