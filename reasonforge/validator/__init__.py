"""
ReasonForge - Validator Module

Provides the validator-side scoring pipeline, consensus, weight setting,
task management, and trap management.
"""

from .scoring import ValidatorScorer
from .consensus import compute_consensus_score
from .weight_setter import WeightSetter
from .task_manager import TaskManager
from .trap_manager import TrapManager

__all__ = [
    "ValidatorScorer",
    "compute_consensus_score",
    "WeightSetter",
    "TaskManager",
    "TrapManager",
]
