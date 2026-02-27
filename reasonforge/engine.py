"""
ReasonForge - Scoring Engine

All 13 whitepaper formulas implemented as pure static methods.
Stateless design: no hidden state, easy to test.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .types import (
    BREAKTHROUGH_MULTIPLIER,
    BREAKTHROUGH_THRESHOLD,
    CONSENSUS_TRIM_DELTA,
    CONSENSUS_WEIGHT,
    OBJECTIVE_WEIGHT,
    PEB_ALPHA,
    PEB_K,
    PEB_STREAK_CAP,
    TRAP_THRESHOLD,
    VAS_SLASH_GAMMA,
    VAS_SLASH_THRESHOLD,
    W_ACCURACY,
    W_EFFICIENCY,
    W_NOVELTY,
    W_QUALITY,
    DimensionScores,
    MinerState,
    ValidatorState,
)


class ScoringEngine:
    """
    Stateless scoring engine implementing all 13 whitepaper formulas.
    All methods are @staticmethod — pure functions with no side effects.
    """

    @staticmethod
    def compute_cms(scores: DimensionScores) -> float:
        """
        Eq. 2 — Composite Miner Score (CMS)
        CMS(m,t) = 0.40*Q + 0.30*A + 0.15*N + 0.15*Eff
        """
        return (
            W_QUALITY * scores.quality
            + W_ACCURACY * scores.accuracy
            + W_NOVELTY * scores.novelty
            + W_EFFICIENCY * scores.efficiency
        )

    @staticmethod
    def compute_s_epoch(
        cms_list: List[float],
        diff_multipliers: List[float],
        trap_penalty: float,
    ) -> float:
        """
        Eq. 3 — Epoch Score
        S_epoch(m) = (1/|T_m|) * sum(CMS(m,t) * D(t)) * trap_penalty(m)
        """
        if not cms_list:
            return 0.0
        n = len(cms_list)
        weighted_sum = sum(c * d for c, d in zip(cms_list, diff_multipliers))
        return (weighted_sum / n) * trap_penalty

    @staticmethod
    def compute_peb(rank: int, streak: int) -> float:
        """
        Eq. 4 — Persistent Excellence Bonus
        PEB(m) = alpha * (1/rank) * sqrt(min(streak, cap))
        Only for miners with rank <= K.
        """
        if rank < 1 or rank > PEB_K:
            return 0.0
        capped_streak = min(streak, PEB_STREAK_CAP)
        if capped_streak <= 0:
            return 0.0
        return PEB_ALPHA * (1.0 / rank) * math.sqrt(capped_streak)

    @staticmethod
    def distribute_miner_emissions(miners: List[MinerState], pool: float) -> List[float]:
        """
        Eq. 5 — Final Miner Reward
        R(m) = E_miner * [S_epoch(m) * (1 + PEB(m))] / sum_j[S_epoch(j) * (1 + PEB(j))]
        Conserves total emission: sum(rewards) == pool.
        """
        weighted = [m.s_epoch * (1.0 + m.peb) for m in miners]
        total_weight = sum(weighted)
        if total_weight <= 0:
            # Distribute equally if all scores are zero
            n = len(miners)
            return [pool / n for _ in miners] if n > 0 else []
        return [(w / total_weight) * pool for w in weighted]

    @staticmethod
    def apply_breakthrough(cms: float, is_breakthrough: bool) -> float:
        """
        Eq. 6 — Breakthrough Multiplier
        CMS_breakthrough = CMS * 2.0  (if previously_unsolved AND CMS > 0.8)
        """
        if is_breakthrough and cms > BREAKTHROUGH_THRESHOLD:
            return cms * BREAKTHROUGH_MULTIPLIER
        return cms

    @staticmethod
    def compute_vas(v_scores: List[float], consensus_scores: List[float]) -> float:
        """
        Eq. 7 — Validator Accuracy Score (VAS)
        VAS(v) = 1 - (1/|T_v|) * sum|score_v(m,t) - score_consensus(m,t)|
        """
        if not v_scores:
            return 1.0
        n = len(v_scores)
        total_dev = sum(abs(v - c) for v, c in zip(v_scores, consensus_scores))
        return max(0.0, 1.0 - total_dev / n)

    @staticmethod
    def distribute_validator_emissions(
        validators: List[ValidatorState], pool: float
    ) -> List[float]:
        """
        Eq. 8 — Validator Reward
        R_v(v) = E_validator * [VAS(v) * stake(v) * rep_mult(v)] /
                 sum_k[VAS(k) * stake(k) * rep_mult(k)]
        """
        weighted = [v.current_vas * v.stake * v.reputation_multiplier for v in validators]
        total_weight = sum(weighted)
        if total_weight <= 0:
            n = len(validators)
            return [pool / n for _ in validators] if n > 0 else []
        return [(w / total_weight) * pool for w in weighted]

    @staticmethod
    def compute_trap_penalty(trap_scores: List[float]) -> float:
        """
        Eq. 9 — Trap Penalty
        penalty = max(0, avg/theta)  if avg < theta
                = 1.0                if avg >= theta
        """
        if not trap_scores:
            return 1.0
        avg = sum(trap_scores) / len(trap_scores)
        if avg >= TRAP_THRESHOLD:
            return 1.0
        return max(0.0, avg / TRAP_THRESHOLD)

    @staticmethod
    def compute_slash(stake: float, vas_7d_avg: float) -> float:
        """
        Eq. 10 — Validator Slashing
        slash(v) = gamma * stake * max(0, theta_slash - VAS_7d_avg)^2
        """
        if vas_7d_avg >= VAS_SLASH_THRESHOLD:
            return 0.0
        return VAS_SLASH_GAMMA * stake * (VAS_SLASH_THRESHOLD - vas_7d_avg) ** 2

    @staticmethod
    def compute_objective_score(checks: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Eq. 11 — Objective Score
        O_score(m,t) = sum_k(omega_k * check_k)
        """
        total = 0.0
        for key, weight in weights.items():
            total += weight * checks.get(key, 0.0)
        return total

    @staticmethod
    def compute_consensus_score(
        val_scores_stakes: List[Tuple[float, float]],
        trim_delta: float = CONSENSUS_TRIM_DELTA,
    ) -> float:
        """
        Eq. 12 — Consensus Score (Stake-weighted trimmed median)
        C_score = TrimmedMedian_delta({score_v * stake_v / sum_stake : v in V})
        Trim delta from top and bottom when |V| >= 5.
        """
        if not val_scores_stakes:
            return 0.0

        total_stake = sum(s for _, s in val_scores_stakes)
        if total_stake <= 0:
            return 0.0

        # Compute stake-weighted scores
        weighted = [(score, stake / total_stake) for score, stake in val_scores_stakes]
        # Sort by score for trimming
        weighted.sort(key=lambda x: x[0])

        n = len(weighted)

        # Apply trimming only when we have enough validators
        if n >= 5:
            trim_count = max(1, int(n * trim_delta))
            weighted = weighted[trim_count : n - trim_count]

        if not weighted:
            return 0.0

        # Stake-weighted average of remaining scores
        total_w = sum(w for _, w in weighted)
        if total_w <= 0:
            return sum(s for s, _ in weighted) / len(weighted)
        return sum(s * w for s, w in weighted) / total_w

    @staticmethod
    def compute_final_score(o_score: float, c_score: float) -> float:
        """
        Eq. 13 — Final Score
        FinalScore(m,t) = 0.60 * O_score + 0.40 * C_score
        """
        return OBJECTIVE_WEIGHT * o_score + CONSENSUS_WEIGHT * c_score
