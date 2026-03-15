"""
ReasonForge - Scoring Pipeline Tests

Tests for the validator scoring pipeline, dimension scoring,
and integration with the MVP engine.
"""

import pytest

from reasonforge.types import DimensionScores, Domain, Task
from reasonforge.validator.consensus import compute_consensus_score
from reasonforge.validator.scoring import ValidatorScorer
from reasonforge.validator.trap_manager import TrapManager
from reasonforge.validator.weight_setter import WeightSetter


class TestValidatorScorer:
    """Test the ValidatorScorer scoring pipeline."""

    @pytest.fixture
    def scorer(self):
        return ValidatorScorer(lean4_enabled=False, sandbox_enabled=False)

    @pytest.fixture
    def math_task(self):
        return Task(
            task_id="test-001",
            problem="Prove sqrt(2) is irrational",
            domain=Domain.MATHEMATICS,
            difficulty=5,
        )

    @pytest.fixture
    def good_response(self):
        return {
            "steps": [
                {
                    "step_id": 0,
                    "reasoning": "Assume sqrt(2) = p/q where gcd(p,q)=1. " * 5,
                    "evidence": "Proof by contradiction",
                    "confidence": 0.9,
                },
                {
                    "step_id": 1,
                    "reasoning": "Then p^2 = 2q^2 implies p is even. " * 5,
                    "evidence": "Even/odd argument",
                    "confidence": 0.85,
                },
                {
                    "step_id": 2,
                    "reasoning": "Let p=2k, then q is also even. Contradiction. " * 5,
                    "evidence": "Both even contradicts gcd(p,q)=1",
                    "confidence": 0.95,
                },
            ],
            "final_answer": "sqrt(2) is irrational by proof by contradiction.",
            "time_taken_ms": 5000,
        }

    @pytest.fixture
    def empty_response(self):
        return {
            "steps": [],
            "final_answer": "",
            "time_taken_ms": 100,
        }

    def test_quality_good_response(self, scorer, math_task, good_response):
        quality = scorer._score_quality(math_task, good_response)
        assert 0.5 < quality <= 1.0

    def test_quality_empty_response(self, scorer, math_task, empty_response):
        quality = scorer._score_quality(math_task, empty_response)
        assert quality == 0.0

    def test_novelty_good_response(self, scorer, math_task, good_response):
        novelty = scorer._score_novelty(math_task, good_response)
        assert 0.0 <= novelty <= 1.0
        assert novelty > 0.0

    def test_novelty_empty_response(self, scorer, math_task, empty_response):
        novelty = scorer._score_novelty(math_task, empty_response)
        assert novelty == 0.0

    def test_efficiency_normal_time(self, scorer, math_task, good_response):
        efficiency = scorer._score_efficiency(math_task, good_response)
        assert 0.5 < efficiency <= 1.0

    def test_efficiency_timeout(self, scorer, math_task):
        response = {"time_taken_ms": 400_000}  # > 300s timeout
        efficiency = scorer._score_efficiency(math_task, response)
        assert efficiency == 0.0

    def test_efficiency_suspiciously_fast(self, scorer, math_task):
        response = {"time_taken_ms": 10}  # < 1% of timeout
        efficiency = scorer._score_efficiency(math_task, response)
        assert efficiency == 0.2

    @pytest.mark.asyncio
    async def test_compute_dimensions(self, scorer, math_task, good_response):
        dims = await scorer.compute_dimensions(math_task, good_response)
        assert isinstance(dims, DimensionScores)
        assert 0.0 <= dims.quality <= 1.0
        assert 0.0 <= dims.accuracy <= 1.0
        assert 0.0 <= dims.novelty <= 1.0
        assert 0.0 <= dims.efficiency <= 1.0


class TestTrapManager:
    """Test trap problem management."""

    def test_evaluate_math_trap_correct(self, sample_trap_task):
        tm = TrapManager()
        score = tm.evaluate_trap_response(sample_trap_task, "The answer is 4")
        assert score == 1.0

    def test_evaluate_math_trap_wrong(self, sample_trap_task):
        tm = TrapManager()
        score = tm.evaluate_trap_response(sample_trap_task, "The answer is 5")
        assert score == 0.0

    def test_evaluate_empty_answer(self, sample_trap_task):
        tm = TrapManager()
        score = tm.evaluate_trap_response(sample_trap_task, "")
        assert score == 0.0

    def test_record_and_retrieve(self):
        tm = TrapManager()
        tm.record_trap_score(1, 0.8)
        tm.record_trap_score(1, 0.9)
        scores = tm.get_trap_scores(1)
        assert len(scores) == 2
        assert scores[0] == 0.8

    def test_trap_penalty(self):
        tm = TrapManager()
        tm.record_trap_score(1, 0.9)
        tm.record_trap_score(1, 0.8)
        penalty = tm.get_trap_penalty(1)
        assert penalty == 1.0  # Above threshold

    def test_trap_penalty_low_scores(self):
        tm = TrapManager()
        tm.record_trap_score(1, 0.1)
        tm.record_trap_score(1, 0.2)
        penalty = tm.get_trap_penalty(1)
        assert penalty < 1.0  # Below threshold


class TestWeightSetter:
    """Test weight computation."""

    def test_compute_weights_basic(self):
        ws = WeightSetter()
        states = {
            0: {"s_epoch": 0.8, "peb": 0.1},
            1: {"s_epoch": 0.6, "peb": 0.0},
            2: {"s_epoch": 0.4, "peb": 0.05},
        }
        uids, weights = ws.compute_weights(states, n=10)
        assert len(uids) == 3
        assert len(weights) == 3
        # Weights should be normalized
        total = sum(weights) if not hasattr(weights, "sum") else float(weights.sum())
        assert abs(total - 1.0) < 1e-6

    def test_compute_weights_empty(self):
        ws = WeightSetter()
        uids, weights = ws.compute_weights({}, n=10)
        assert len(uids) == 0

    def test_compute_weights_all_zero(self):
        ws = WeightSetter()
        states = {
            0: {"s_epoch": 0.0, "peb": 0.0},
            1: {"s_epoch": 0.0, "peb": 0.0},
        }
        uids, weights = ws.compute_weights(states, n=10)
        assert len(uids) == 0

    def test_higher_score_higher_weight(self):
        ws = WeightSetter()
        states = {
            0: {"s_epoch": 0.9, "peb": 0.2},
            1: {"s_epoch": 0.3, "peb": 0.0},
        }
        uids, weights = ws.compute_weights(states, n=10)
        weights_list = list(weights) if not hasattr(weights, "tolist") else weights.tolist()
        assert weights_list[0] > weights_list[1]


class TestConsensus:
    """Test consensus scoring."""

    def test_basic_consensus(self):
        scores = [(0.8, 100.0), (0.7, 100.0), (0.9, 100.0)]
        result = compute_consensus_score(scores)
        assert 0.0 <= result <= 1.0

    def test_empty_consensus(self):
        assert compute_consensus_score([]) == 0.0

    def test_consensus_with_different_stakes(self):
        scores = [(0.9, 1000.0), (0.5, 10.0)]
        result = compute_consensus_score(scores)
        # High-stake validator should dominate
        assert result > 0.6
