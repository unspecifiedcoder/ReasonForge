"""
ReasonForge - The Decentralized Marketplace for Verifiable Intelligence

A Bittensor subnet proposal implementing verifiable multi-step reasoning
with 13 whitepaper formulas as working code.
"""

__version__ = "0.1.0"

from .engine import ScoringEngine
from .plagiarism import PlagiarismDetector
from .simulator import (
    EpochSimulator,
    MinerProfile,
    ValidatorProfile,
    create_default_miners,
    create_default_validators,
)
from .task_generator import TaskGenerator
from .types import (
    DimensionScores,
    Domain,
    EpochResult,
    MinerState,
    MinerSubmission,
    ReasoningStep,
    Task,
    TaskSource,
    ValidatorScore,
    ValidatorState,
)

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
