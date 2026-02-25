"""
ReasonForge - The Decentralized Marketplace for Verifiable Intelligence

A Bittensor subnet proposal implementing verifiable multi-step reasoning
with 13 whitepaper formulas as working code.
"""

__version__ = "0.1.0"

from .types import (
    Domain,
    TaskSource,
    Task,
    ReasoningStep,
    MinerSubmission,
    DimensionScores,
    ValidatorScore,
    MinerState,
    ValidatorState,
    EpochResult,
)

from .engine import ScoringEngine
from .simulator import (
    MinerProfile,
    ValidatorProfile,
    EpochSimulator,
    create_default_miners,
    create_default_validators,
)
from .task_generator import TaskGenerator
from .plagiarism import PlagiarismDetector

__all__ = [
    "Domain",
    "TaskSource",
    "Task",
    "ReasoningStep",
    "MinerSubmission",
    "DimensionScores",
    "ValidatorScore",
    "MinerState",
    "ValidatorState",
    "EpochResult",
    "ScoringEngine",
    "MinerProfile",
    "ValidatorProfile",
    "EpochSimulator",
    "TaskGenerator",
    "PlagiarismDetector",
    "create_default_miners",
    "create_default_validators",
]
