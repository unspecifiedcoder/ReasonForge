"""
ReasonForge - Validator Module

Provides the validator-side scoring pipeline, consensus, weight setting,
task management, and trap management.
"""

from .consensus import compute_consensus_score
from .scoring import ValidatorScorer
from .task_manager import TaskManager
from .trap_manager import TrapManager
from .weight_setter import WeightSetter

__all__ = [
    "ValidatorScorer",
    "compute_consensus_score",
    "WeightSetter",
    "TaskManager",
    "TrapManager",
]
