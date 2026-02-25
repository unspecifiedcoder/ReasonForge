"""
ReasonForge - Engine Formula Tests

Tests every formula method in ScoringEngine against hand-calculated values.
Each test states which equation is being tested.
"""

import math
import pytest

from reasonforge.types import DimensionScores, MinerState, ValidatorState
from reasonforge.engine import ScoringEngine


# ──────────────────────────────────────────────
# Eq. 2 — CMS
# ──────────────────────────────────────────────

class TestCMS:
    def test_cms_computation(self):
        """Eq.2: CMS = 0.40*Q + 0.30*A + 0.15*N + 0.15*Eff"""
        scores = DimensionScores(quality=0.8, accuracy=0.7, novelty=0.6, efficiency=0.5)
        expected = 0.40 * 0.8 + 0.30 * 0.7 + 0.15 * 0.6 + 0.15 * 0.5
        # = 0.32 + 0.21 + 0.09 + 0.075 = 0.695
        assert abs(ScoringEngine.compute_cms(scores) - expected) < 1e-10
        assert abs(ScoringEngine.compute_cms(scores) - 0.695) < 1e-10

    def test_cms_boundary_zeros(self):
        """Eq.2: All zeros -> CMS = 0.0"""
        scores = DimensionScores(quality=0.0, accuracy=0.0, novelty=0.0, efficiency=0.0)
        assert ScoringEngine.compute_cms(scores) == 0.0

    def test_cms_boundary_ones(self):
        """Eq.2: All ones -> CMS = 1.0"""
        scores = DimensionScores(quality=1.0, accuracy=1.0, novelty=1.0, efficiency=1.0)
        expected = 0.40 + 0.30 + 0.15 + 0.15
        assert abs(ScoringEngine.compute_cms(scores) - expected) < 1e-10
        assert abs(ScoringEngine.compute_cms(scores) - 1.0) < 1e-10

    def test_cms_matches_property(self):
        """Eq.2: ScoringEngine.compute_cms matches DimensionScores.cms property"""
        scores = DimensionScores(quality=0.9, accuracy=0.85, novelty=0.7, efficiency=0.6)
        assert abs(ScoringEngine.compute_cms(scores) - scores.cms) < 1e-10


# ──────────────────────────────────────────────
# Eq. 3 — Epoch Score
# ──────────────────────────────────────────────

class TestSEpoch:
    def test_s_epoch_basic(self):
        """Eq.3: S_epoch with known values"""
        cms_list = [0.6, 0.8, 0.7]
        diff_mults = [1.0, 1.5, 1.25]
        # weighted_sum = 0.6*1.0 + 0.8*1.5 + 0.7*1.25 = 0.6 + 1.2 + 0.875 = 2.675
        # avg = 2.675 / 3 = 0.89166...
        # penalty = 1.0
        expected = 2.675 / 3
        assert abs(ScoringEngine.compute_s_epoch(cms_list, diff_mults, 1.0) - expected) < 1e-10

    def test_s_epoch_with_penalty(self):
        """Eq.3: S_epoch with trap penalty applied"""
        cms_list = [0.8]
        diff_mults = [1.5]
        penalty = 0.5
        expected = (0.8 * 1.5 / 1) * 0.5  # = 0.6
        assert abs(ScoringEngine.compute_s_epoch(cms_list, diff_mults, penalty) - expected) < 1e-10

    def test_s_epoch_empty(self):
        """Eq.3: Empty CMS list -> 0.0"""
        assert ScoringEngine.compute_s_epoch([], [], 1.0) == 0.0


# ──────────────────────────────────────────────
# Eq. 4 — PEB
# ──────────────────────────────────────────────

class TestPEB:
    def test_peb_rank1_streak4(self):
        """Eq.4: PEB = 0.20 * (1/1) * sqrt(4) = 0.40"""
        result = ScoringEngine.compute_peb(rank=1, streak=4)
        expected = 0.20 * 1.0 * math.sqrt(4)  # = 0.40
        assert abs(result - expected) < 1e-10
        assert abs(result - 0.40) < 1e-10

    def test_peb_outside_topk(self):
        """Eq.4: rank=11 (outside top-K=10) -> 0.0"""
        assert ScoringEngine.compute_peb(rank=11, streak=5) == 0.0

    def test_peb_streak_cap(self):
        """Eq.4: streak=100 should cap at 10"""
        result = ScoringEngine.compute_peb(rank=1, streak=100)
        expected = 0.20 * 1.0 * math.sqrt(10)
        assert abs(result - expected) < 1e-10

    def test_peb_rank5_streak9(self):
        """Eq.4: PEB = 0.20 * (1/5) * sqrt(9) = 0.12"""
        result = ScoringEngine.compute_peb(rank=5, streak=9)
        expected = 0.20 * (1.0 / 5) * math.sqrt(9)  # = 0.12
        assert abs(result - expected) < 1e-10

    def test_peb_zero_streak(self):
        """Eq.4: streak=0 -> 0.0"""
        assert ScoringEngine.compute_peb(rank=1, streak=0) == 0.0

    def test_peb_rank0_invalid(self):
        """Eq.4: rank=0 (invalid) -> 0.0"""
        assert ScoringEngine.compute_peb(rank=0, streak=5) == 0.0


# ──────────────────────────────────────────────
# Eq. 5 — Miner Emission Distribution
# ──────────────────────────────────────────────

class TestMinerEmissions:
    def test_emission_conservation(self):
        """Eq.5: sum(rewards) == pool"""
        miners = [
            MinerState(miner_id="m1", s_epoch=0.9, peb=0.1),
            MinerState(miner_id="m2", s_epoch=0.6, peb=0.0),
            MinerState(miner_id="m3", s_epoch=0.3, peb=0.05),
        ]
        pool = 90.0
        rewards = ScoringEngine.distribute_miner_emissions(miners, pool)
        assert abs(sum(rewards) - pool) < 1e-10

    def test_emission_monotonic(self):
        """Eq.5: higher S_epoch * (1+PEB) -> higher reward"""
        miners = [
            MinerState(miner_id="m1", s_epoch=0.9, peb=0.1),
            MinerState(miner_id="m2", s_epoch=0.6, peb=0.0),
            MinerState(miner_id="m3", s_epoch=0.3, peb=0.0),
        ]
        rewards = ScoringEngine.distribute_miner_emissions(miners, 90.0)
        assert rewards[0] > rewards[1] > rewards[2]

    def test_emission_all_zero(self):
        """Eq.5: all scores zero -> equal distribution"""
        miners = [
            MinerState(miner_id="m1", s_epoch=0.0, peb=0.0),
            MinerState(miner_id="m2", s_epoch=0.0, peb=0.0),
        ]
        rewards = ScoringEngine.distribute_miner_emissions(miners, 90.0)
        assert abs(rewards[0] - 45.0) < 1e-10
        assert abs(rewards[1] - 45.0) < 1e-10


# ──────────────────────────────────────────────
# Eq. 6 — Breakthrough
# ──────────────────────────────────────────────

class TestBreakthrough:
    def test_breakthrough_applied(self):
        """Eq.6: CMS=0.9 on unsolved task -> 0.9 * 2.0 = 1.8"""
        result = ScoringEngine.apply_breakthrough(0.9, is_breakthrough=True)
        assert abs(result - 1.8) < 1e-10

    def test_breakthrough_not_applied_low_cms(self):
        """Eq.6: CMS=0.7 (below threshold) -> unchanged"""
        result = ScoringEngine.apply_breakthrough(0.7, is_breakthrough=True)
        assert abs(result - 0.7) < 1e-10

    def test_breakthrough_not_applied_not_unsolved(self):
        """Eq.6: Not previously unsolved -> unchanged"""
        result = ScoringEngine.apply_breakthrough(0.9, is_breakthrough=False)
        assert abs(result - 0.9) < 1e-10

    def test_breakthrough_at_threshold(self):
        """Eq.6: CMS=0.8 exactly (not > 0.8) -> unchanged"""
        result = ScoringEngine.apply_breakthrough(0.8, is_breakthrough=True)
        assert abs(result - 0.8) < 1e-10


# ──────────────────────────────────────────────
# Eq. 7 — VAS
# ──────────────────────────────────────────────

class TestVAS:
    def test_vas_perfect(self):
        """Eq.7: all scores match consensus -> VAS = 1.0"""
        v_scores = [0.8, 0.7, 0.6]
        consensus = [0.8, 0.7, 0.6]
        assert abs(ScoringEngine.compute_vas(v_scores, consensus) - 1.0) < 1e-10

    def test_vas_deviation(self):
        """Eq.7: known deviations"""
        v_scores = [0.9, 0.5, 0.7]
        consensus = [0.8, 0.7, 0.6]
        # deviations: |0.1| + |0.2| + |0.1| = 0.4
        # VAS = 1 - 0.4/3 = 1 - 0.1333... = 0.8666...
        expected = 1.0 - 0.4 / 3
        assert abs(ScoringEngine.compute_vas(v_scores, consensus) - expected) < 1e-10

    def test_vas_empty(self):
        """Eq.7: empty scores -> 1.0"""
        assert ScoringEngine.compute_vas([], []) == 1.0


# ──────────────────────────────────────────────
# Eq. 8 — Validator Emission Distribution
# ──────────────────────────────────────────────

class TestValidatorEmissions:
    def test_validator_emission_conservation(self):
        """Eq.8: sum(validator rewards) == pool"""
        validators = [
            ValidatorState(validator_id="v1", stake=5000, current_vas=0.95, reputation_multiplier=1.4),
            ValidatorState(validator_id="v2", stake=3000, current_vas=0.90, reputation_multiplier=1.2),
            ValidatorState(validator_id="v3", stake=1000, current_vas=0.70, reputation_multiplier=1.0),
        ]
        pool = 10.0
        rewards = ScoringEngine.distribute_validator_emissions(validators, pool)
        assert abs(sum(rewards) - pool) < 1e-10

    def test_validator_higher_stake_more_reward(self):
        """Eq.8: higher stake*VAS*rep -> more reward"""
        validators = [
            ValidatorState(validator_id="v1", stake=5000, current_vas=0.95, reputation_multiplier=1.0),
            ValidatorState(validator_id="v2", stake=1000, current_vas=0.95, reputation_multiplier=1.0),
        ]
        rewards = ScoringEngine.distribute_validator_emissions(validators, 10.0)
        assert rewards[0] > rewards[1]


# ──────────────────────────────────────────────
# Eq. 9 — Trap Penalty
# ──────────────────────────────────────────────

class TestTrapPenalty:
    def test_trap_above_threshold(self):
        """Eq.9: avg >= theta_trap -> penalty = 1.0"""
        assert ScoringEngine.compute_trap_penalty([0.5, 0.4, 0.3]) == 1.0

    def test_trap_below_threshold(self):
        """Eq.9: avg < theta_trap -> penalty = avg/theta"""
        scores = [0.1, 0.2]  # avg = 0.15
        expected = 0.15 / 0.30  # = 0.5
        assert abs(ScoringEngine.compute_trap_penalty(scores) - expected) < 1e-10

    def test_trap_zero(self):
        """Eq.9: all zeros -> penalty = 0.0"""
        assert ScoringEngine.compute_trap_penalty([0.0, 0.0]) == 0.0

    def test_trap_empty(self):
        """Eq.9: no trap scores -> penalty = 1.0 (no traps encountered)"""
        assert ScoringEngine.compute_trap_penalty([]) == 1.0


# ──────────────────────────────────────────────
# Eq. 10 — Slash
# ──────────────────────────────────────────────

class TestSlash:
    def test_slash_below_threshold(self):
        """Eq.10: VAS_avg=0.40 -> slash = gamma * stake * (0.60-0.40)^2"""
        stake = 5000.0
        vas_avg = 0.40
        expected = 0.05 * 5000 * (0.60 - 0.40) ** 2  # = 0.05 * 5000 * 0.04 = 10.0
        assert abs(ScoringEngine.compute_slash(stake, vas_avg) - expected) < 1e-10
        assert abs(ScoringEngine.compute_slash(stake, vas_avg) - 10.0) < 1e-10

    def test_slash_above_threshold(self):
        """Eq.10: VAS_avg=0.80 (above threshold) -> 0.0"""
        assert ScoringEngine.compute_slash(5000.0, 0.80) == 0.0

    def test_slash_at_threshold(self):
        """Eq.10: VAS_avg=0.60 exactly -> 0.0"""
        assert ScoringEngine.compute_slash(5000.0, 0.60) == 0.0


# ──────────────────────────────────────────────
# Eq. 11 — Objective Score
# ──────────────────────────────────────────────

class TestObjectiveScore:
    def test_objective_score(self):
        """Eq.11: weighted sum of checks"""
        checks = {"proof": 0.9, "steps": 0.8, "numerical": 0.7}
        weights = {"proof": 0.60, "steps": 0.25, "numerical": 0.15}
        expected = 0.60 * 0.9 + 0.25 * 0.8 + 0.15 * 0.7
        # = 0.54 + 0.20 + 0.105 = 0.845
        assert abs(ScoringEngine.compute_objective_score(checks, weights) - expected) < 1e-10

    def test_objective_missing_check(self):
        """Eq.11: missing check defaults to 0"""
        checks = {"proof": 0.9}
        weights = {"proof": 0.60, "steps": 0.25, "numerical": 0.15}
        expected = 0.60 * 0.9 + 0.25 * 0.0 + 0.15 * 0.0
        assert abs(ScoringEngine.compute_objective_score(checks, weights) - expected) < 1e-10


# ──────────────────────────────────────────────
# Eq. 12 — Consensus Score
# ──────────────────────────────────────────────

class TestConsensusScore:
    def test_consensus_trimmed_median(self):
        """Eq.12: verify trimming works with 5+ validators"""
        # 5 validators -> trim 1 from each end
        val_scores_stakes = [
            (0.5, 1000),  # trimmed (lowest)
            (0.7, 2000),
            (0.8, 3000),
            (0.85, 2000),
            (0.95, 1000),  # trimmed (highest)
        ]
        result = ScoringEngine.compute_consensus_score(val_scores_stakes)
        # After sorting by score: [(0.5,1000),(0.7,2000),(0.8,3000),(0.85,2000),(0.95,1000)]
        # Trim 1 from each end: [(0.7,2000),(0.8,3000),(0.85,2000)]
        # Total stake = 1000+2000+3000+2000+1000 = 9000
        # weights: 2000/9000, 3000/9000, 2000/9000
        # total_w = 7000/9000
        # result = (0.7*2000/9000 + 0.8*3000/9000 + 0.85*2000/9000) / (7000/9000)
        # = (1400 + 2400 + 1700) / 7000 = 5500/7000 = 0.78571...
        total_stake = 9000
        numerator = 0.7 * (2000 / total_stake) + 0.8 * (3000 / total_stake) + 0.85 * (2000 / total_stake)
        denom = (2000 + 3000 + 2000) / total_stake
        expected = numerator / denom
        assert abs(result - expected) < 1e-6

    def test_consensus_stake_weighted(self):
        """Eq.12: higher stake influences median more"""
        # 3 validators (no trimming since < 5)
        val1 = [(0.6, 100), (0.8, 9000), (0.9, 100)]
        val2 = [(0.6, 9000), (0.8, 100), (0.9, 100)]
        r1 = ScoringEngine.compute_consensus_score(val1)
        r2 = ScoringEngine.compute_consensus_score(val2)
        # With high stake on 0.8, r1 should be closer to 0.8
        # With high stake on 0.6, r2 should be closer to 0.6
        assert r1 > r2

    def test_consensus_empty(self):
        """Eq.12: empty -> 0.0"""
        assert ScoringEngine.compute_consensus_score([]) == 0.0


# ──────────────────────────────────────────────
# Eq. 13 — Final Score
# ──────────────────────────────────────────────

class TestFinalScore:
    def test_final_score(self):
        """Eq.13: FinalScore = 0.60*O + 0.40*C"""
        result = ScoringEngine.compute_final_score(0.85, 0.75)
        expected = 0.60 * 0.85 + 0.40 * 0.75  # = 0.51 + 0.30 = 0.81
        assert abs(result - expected) < 1e-10
        assert abs(result - 0.81) < 1e-10

    def test_final_score_zeros(self):
        """Eq.13: all zeros -> 0.0"""
        assert ScoringEngine.compute_final_score(0.0, 0.0) == 0.0

    def test_final_score_ones(self):
        """Eq.13: all ones -> 1.0"""
        result = ScoringEngine.compute_final_score(1.0, 1.0)
        assert abs(result - 1.0) < 1e-10
