"""
ReasonForge - Consensus Module

Stake-weighted trimmed median for multi-validator consensus scoring.
Wraps the MVP ScoringEngine's consensus computation for production use.
"""

from __future__ import annotations

from typing import List, Tuple

from ..engine import ScoringEngine
from ..types import CONSENSUS_TRIM_DELTA


def compute_consensus_score(
    validator_scores: List[Tuple[float, float]],
    trim_delta: float = CONSENSUS_TRIM_DELTA,
) -> float:
    """
    Stake-weighted trimmed median (Eq. 12).
    Reuses ScoringEngine.compute_consensus_score from MVP.

    Args:
        validator_scores: List of (score, stake) tuples.
        trim_delta: Fraction to trim from top and bottom.

    Returns:
        Consensus score as a float.
    """
    return ScoringEngine.compute_consensus_score(validator_scores, trim_delta)
