"""
ReasonForge - Type and Constant Tests

Validates data types, protocol constants, and computed properties.
"""

import pytest
from reasonforge.types import (
    DIFFICULTY_MULTIPLIER,
    TRAP_THRESHOLD,
    VAS_REP_THRESHOLD,
    VAS_REP_MAX_MULTIPLIER,
    VAS_SLASH_THRESHOLD,
    VAS_SLASH_GAMMA,
    DimensionScores,
    MinerState,
    ValidatorState,
    Task,
    Domain,
    W_QUALITY,
    W_ACCURACY,
    W_NOVELTY,
    W_EFFICIENCY,
)


class TestDifficultyMultiplier:
    def test_difficulty_multiplier_map(self):
        """All 10 difficulty levels mapped correctly."""
        expected = {
            1: 1.0, 2: 1.0, 3: 1.25, 4: 1.25,
            5: 1.5, 6: 1.5, 7: 1.75, 8: 1.75,
            9: 2.0, 10: 2.0,
        }
        for level, mult in expected.items():
            assert DIFFICULTY_MULTIPLIER[level] == mult
            # Also test Task property
            task = Task(difficulty=level)
            assert task.difficulty_multiplier == mult

    def test_difficulty_multiplier_complete(self):
        """All 10 levels are present."""
        assert len(DIFFICULTY_MULTIPLIER) == 10
        for i in range(1, 11):
            assert i in DIFFICULTY_MULTIPLIER


class TestDimensionScores:
    def test_dimension_scores_cms(self):
        """CMS property computes Eq. 2 correctly."""
        scores = DimensionScores(quality=0.9, accuracy=0.8, novelty=0.7, efficiency=0.6)
        expected = W_QUALITY * 0.9 + W_ACCURACY * 0.8 + W_NOVELTY * 0.7 + W_EFFICIENCY * 0.6
        assert abs(scores.cms - expected) < 1e-10

    def test_dimension_scores_zeros(self):
        """All zero scores -> CMS = 0."""
        scores = DimensionScores()
        assert scores.cms == 0.0

    def test_dimension_scores_ones(self):
        """All 1.0 scores -> CMS = 1.0 (weights sum to 1)."""
        scores = DimensionScores(quality=1.0, accuracy=1.0, novelty=1.0, efficiency=1.0)
        weights_sum = W_QUALITY + W_ACCURACY + W_NOVELTY + W_EFFICIENCY
        assert abs(weights_sum - 1.0) < 1e-10
        assert abs(scores.cms - 1.0) < 1e-10


class TestMinerState:
    def test_miner_state_trap_penalty_no_traps(self):
        """No trap scores -> penalty = 1.0."""
        ms = MinerState(miner_id="m1")
        assert ms.trap_penalty == 1.0
        assert ms.trap_score_avg == 1.0

    def test_miner_state_trap_penalty_above_threshold(self):
        """Trap scores above threshold -> penalty = 1.0."""
        ms = MinerState(miner_id="m1", trap_scores=[0.5, 0.4])
        assert ms.trap_penalty == 1.0

    def test_miner_state_trap_penalty_below_threshold(self):
        """Trap scores below threshold -> penalty < 1.0."""
        ms = MinerState(miner_id="m1", trap_scores=[0.1, 0.2])
        avg = 0.15
        expected = avg / TRAP_THRESHOLD  # 0.15 / 0.30 = 0.5
        assert abs(ms.trap_penalty - expected) < 1e-10

    def test_miner_state_trap_penalty_zero(self):
        """All zero trap scores -> penalty = 0.0."""
        ms = MinerState(miner_id="m1", trap_scores=[0.0, 0.0])
        assert ms.trap_penalty == 0.0


class TestValidatorState:
    def test_validator_state_vas_avg_empty(self):
        """No VAS history -> avg = 1.0."""
        vs = ValidatorState(validator_id="v1")
        assert vs.vas_7d_avg == 1.0
        assert vs.vas_30d_avg == 1.0

    def test_validator_state_vas_7d_avg(self):
        """7-day rolling average works correctly."""
        vs = ValidatorState(
            validator_id="v1",
            vas_history=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        )
        # Last 7: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] ... wait, last 7 elements:
        # [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] -> no
        # vas_history[-7:] = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        # Wait: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        # [-7:] = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        last7 = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        expected = sum(last7) / 7
        assert abs(vs.vas_7d_avg - expected) < 1e-10

    def test_validator_reputation_multiplier_below_threshold(self):
        """VAS avg below threshold -> rep = 1.0."""
        vs = ValidatorState(validator_id="v1", vas_history=[0.5, 0.6, 0.7])
        result = vs.compute_reputation_multiplier()
        assert result == 1.0

    def test_validator_reputation_multiplier_above_threshold(self):
        """VAS avg above threshold -> rep > 1.0, capped at 1.5."""
        vs = ValidatorState(validator_id="v1", vas_history=[0.95] * 30)
        result = vs.compute_reputation_multiplier()
        # avg = 0.95, bonus = 0.5 * (0.95 - 0.80) / 0.20 = 0.5 * 0.75 = 0.375
        expected = min(1.5, 1.0 + 0.5 * (0.95 - 0.80) / 0.20)
        assert abs(result - expected) < 1e-10

    def test_validator_reputation_max_cap(self):
        """Perfect VAS -> rep capped at 1.5."""
        vs = ValidatorState(validator_id="v1", vas_history=[1.0] * 30)
        result = vs.compute_reputation_multiplier()
        assert result == VAS_REP_MAX_MULTIPLIER

    def test_validator_slash_computation(self):
        """Slashing computed correctly for low VAS."""
        vs = ValidatorState(validator_id="v1", stake=5000, vas_history=[0.40] * 7)
        result = vs.compute_slash()
        expected = VAS_SLASH_GAMMA * 5000 * (VAS_SLASH_THRESHOLD - 0.40) ** 2
        assert abs(result - expected) < 1e-10


class TestTaskDefaults:
    def test_task_defaults(self):
        """Task has sensible defaults."""
        task = Task()
        assert task.domain == Domain.MATHEMATICS
        assert task.difficulty == 5
        assert task.is_trap is False
        assert task.previously_unsolved is False
        assert task.ground_truth_score is None
        assert len(task.task_id) > 0

    def test_task_difficulty_multiplier_default(self):
        """Default difficulty 5 -> multiplier 1.5."""
        task = Task()
        assert task.difficulty_multiplier == 1.5
