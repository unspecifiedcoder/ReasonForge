"""
ReasonForge - Data Types & Protocol Constants

All data structures and protocol constants for the ReasonForge subnet.
Constants match the whitepaper specification exactly.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

# ──────────────────────────────────────────────
# Protocol Constants
# ──────────────────────────────────────────────

# CMS Weights (Eq. 2)
W_QUALITY = 0.40
W_ACCURACY = 0.30
W_NOVELTY = 0.15
W_EFFICIENCY = 0.15

# Emission Split (Eq. 1)
EMISSION_MINER_SHARE = 0.90
EMISSION_VALIDATOR_SHARE = 0.10

# PEB Parameters (Eq. 4)
PEB_ALPHA = 0.20
PEB_K = 10  # Top-K miners eligible
PEB_STREAK_CAP = 10  # Max streak value

# Breakthrough (Eq. 6)
BREAKTHROUGH_MULTIPLIER = 2.0
BREAKTHROUGH_THRESHOLD = 0.8  # Min CMS to qualify

# Trap Problems (Eq. 9)
TRAP_RATE = 0.15  # 15% of tasks are traps
TRAP_THRESHOLD = 0.30  # theta_trap

# Similarity Detection
SIMILARITY_THRESHOLD = 0.95
SIMILARITY_PENALTY = 0.50

# Validator Slashing (Eq. 10)
VAS_SLASH_THRESHOLD = 0.60  # theta_slash
VAS_SLASH_GAMMA = 0.05  # gamma

# Validator Reputation
VAS_REP_THRESHOLD = 0.80
VAS_REP_MAX_MULTIPLIER = 1.50

# Operational
TASKS_PER_EPOCH = 12
VALIDATORS_PER_TASK = 3
MICRO_ROUND_SECONDS = 300  # 5 minutes
EPOCH_HOURS = 24

# Difficulty Multiplier Map (difficulty 1-10 -> D(t))
DIFFICULTY_MULTIPLIER: Dict[int, float] = {
    1: 1.0,
    2: 1.0,
    3: 1.25,
    4: 1.25,
    5: 1.5,
    6: 1.5,
    7: 1.75,
    8: 1.75,
    9: 2.0,
    10: 2.0,
}

# Objective/Consensus Split (Eq. 13)
OBJECTIVE_WEIGHT = 0.60
CONSENSUS_WEIGHT = 0.40
CONSENSUS_TRIM_DELTA = 0.10  # Trim top/bottom 10%


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────


class Domain(str, Enum):
    MATHEMATICS = "mathematics"
    CODE = "code"
    SCIENTIFIC = "scientific"
    STRATEGIC = "strategic"
    CAUSAL = "causal"
    ETHICAL = "ethical"


class TaskSource(str, Enum):
    USER_API = "user_api"
    BENCHMARK = "benchmark"
    SYNTHETIC = "synthetic"
    TRAP = "trap"


# Domain-Specific Check Weights (Eq. 11)
DOMAIN_CHECK_WEIGHTS: Dict[Domain, Dict[str, float]] = {
    Domain.MATHEMATICS: {"proof": 0.60, "steps": 0.25, "numerical": 0.15},
    Domain.CODE: {"tests": 0.50, "static_analysis": 0.20, "formal": 0.30},
    Domain.SCIENTIFIC: {"simulation": 0.40, "statistics": 0.35, "citations": 0.25},
    Domain.STRATEGIC: {"solver": 0.50, "constraints": 0.30, "equilibrium": 0.20},
    Domain.CAUSAL: {"docalculus": 0.40, "bootstrap": 0.35, "dag": 0.25},
    Domain.ETHICAL: {"coverage": 0.30, "logic": 0.70},
}


# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────


@dataclass
class Task:
    """A reasoning task assigned to miners."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem: str = ""
    domain: Domain = Domain.MATHEMATICS
    difficulty: int = 5
    timeout_seconds: int = MICRO_ROUND_SECONDS
    source: TaskSource = TaskSource.SYNTHETIC
    is_trap: bool = False
    ground_truth_score: Optional[float] = None
    previously_unsolved: bool = False

    @property
    def difficulty_multiplier(self) -> float:
        return DIFFICULTY_MULTIPLIER.get(self.difficulty, 1.0)


@dataclass
class ReasoningStep:
    """A single step in a miner's reasoning chain."""

    step_id: int = 0
    reasoning: str = ""
    evidence: str = ""
    confidence: float = 0.0
    formal_proof_fragment: Optional[str] = None


@dataclass
class MinerSubmission:
    """A miner's submission for a task."""

    task_id: str = ""
    miner_id: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    proof_status: Optional[str] = None  # "VERIFIED" / "FAILED" / None
    time_ms: int = 0
    submission_hash: str = ""
    submitted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of submission content."""
        content = f"{self.task_id}:{self.miner_id}:{self.final_answer}"
        for step in self.steps:
            content += f":{step.reasoning}"
        self.submission_hash = hashlib.sha256(content.encode()).hexdigest()
        return self.submission_hash


@dataclass
class DimensionScores:
    """Scores across the four quality dimensions."""

    quality: float = 0.0
    accuracy: float = 0.0
    novelty: float = 0.0
    efficiency: float = 0.0

    @property
    def cms(self) -> float:
        """Compute Composite Miner Score (Eq. 2)."""
        return (
            W_QUALITY * self.quality
            + W_ACCURACY * self.accuracy
            + W_NOVELTY * self.novelty
            + W_EFFICIENCY * self.efficiency
        )


@dataclass
class ValidatorScore:
    """A validator's score for a specific miner's submission."""

    validator_id: str = ""
    objective_score: float = 0.0
    consensus_score: float = 0.0
    final_score: float = 0.0
    dimension_scores: Optional[DimensionScores] = None


@dataclass
class MinerState:
    """Persistent miner state across epochs."""

    miner_id: str = ""
    name: str = ""
    epoch_scores: List[float] = field(default_factory=list)
    epoch_tasks: List[str] = field(default_factory=list)
    trap_scores: List[float] = field(default_factory=list)
    s_epoch: float = 0.0
    peb: float = 0.0
    rank: int = 0
    streak: int = 0
    total_tao_earned: float = 0.0
    epoch_tao: float = 0.0
    task_count: int = 0
    breakthroughs: int = 0

    @property
    def trap_score_avg(self) -> float:
        if not self.trap_scores:
            return 1.0  # No traps encountered = perfect
        return sum(self.trap_scores) / len(self.trap_scores)

    @property
    def trap_penalty(self) -> float:
        """Compute trap penalty (Eq. 9)."""
        if not self.trap_scores:
            return 1.0
        avg = self.trap_score_avg
        if avg >= TRAP_THRESHOLD:
            return 1.0
        return max(0.0, avg / TRAP_THRESHOLD)


@dataclass
class ValidatorState:
    """Persistent validator state across epochs."""

    validator_id: str = ""
    name: str = ""
    stake: float = 0.0
    vas_history: List[float] = field(default_factory=list)
    current_vas: float = 1.0
    reputation_multiplier: float = 1.0
    total_tao_earned: float = 0.0
    epoch_tao: float = 0.0
    slashed_amount: float = 0.0
    evaluations_count: int = 0

    @property
    def vas_7d_avg(self) -> float:
        if not self.vas_history:
            return 1.0
        recent = self.vas_history[-7:]
        return sum(recent) / len(recent)

    @property
    def vas_30d_avg(self) -> float:
        if not self.vas_history:
            return 1.0
        recent = self.vas_history[-30:]
        return sum(recent) / len(recent)

    def compute_reputation_multiplier(self) -> float:
        """Compute reputation multiplier from 30-day VAS average."""
        avg = self.vas_30d_avg
        if avg <= VAS_REP_THRESHOLD:
            self.reputation_multiplier = 1.0
        else:
            bonus = 0.5 * (avg - VAS_REP_THRESHOLD) / 0.20
            self.reputation_multiplier = min(VAS_REP_MAX_MULTIPLIER, 1.0 + bonus)
        return self.reputation_multiplier

    def compute_slash(self) -> float:
        """Compute slashing amount (Eq. 10)."""
        avg = self.vas_7d_avg
        if avg >= VAS_SLASH_THRESHOLD:
            self.slashed_amount = 0.0
        else:
            self.slashed_amount = VAS_SLASH_GAMMA * self.stake * (VAS_SLASH_THRESHOLD - avg) ** 2
        return self.slashed_amount


@dataclass
class EpochResult:
    """Results from a single epoch simulation."""

    epoch_id: int = 0
    total_emission: float = 0.0
    miner_pool: float = 0.0
    validator_pool: float = 0.0
    miner_results: List[dict] = field(default_factory=list)
    validator_results: List[dict] = field(default_factory=list)
    tasks_processed: int = 0
    traps_injected: int = 0
    breakthroughs: int = 0
    avg_cms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
