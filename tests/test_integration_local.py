"""
ReasonForge - Local Integration Tests

End-to-end tests: validator sends task -> miner processes -> validator scores.
Runs without a real blockchain - tests the full pipeline locally.
"""

import pytest
from reasonforge.types import Task, Domain, DimensionScores, MinerState
from reasonforge.engine import ScoringEngine
from reasonforge.protocol import ReasoningTask, create_reasoning_task, verify_submission_hash
from reasonforge.validator.scoring import ValidatorScorer
from reasonforge.validator.trap_manager import TrapManager
from reasonforge.validator.weight_setter import WeightSetter
from reasonforge.validator.task_manager import TaskManager
from reasonforge.security.sanitizer import InputSanitizer


class TestFullEpochCycle:
    """Test a complete epoch cycle without blockchain."""

    @pytest.fixture
    def task_manager(self):
        return TaskManager(benchmark_dir="benchmarks", seed=42)

    @pytest.fixture
    def scorer(self):
        return ValidatorScorer(lean4_enabled=False, sandbox_enabled=False)

    @pytest.fixture
    def weight_setter(self):
        return WeightSetter()

    def test_task_generation(self, task_manager):
        """Tasks can be generated."""
        tasks = task_manager.generate_epoch_tasks(count=12, trap_rate=0.15)
        assert len(tasks) == 12
        # At least 1 trap
        traps = [t for t in tasks if t.is_trap]
        assert len(traps) >= 1

    def test_synapse_roundtrip(self):
        """Synapse can be created, filled, and deserialized."""
        synapse = create_reasoning_task(
            problem="What is 2+2?",
            domain="mathematics",
            difficulty=2,
        )
        assert synapse.task_id

        # Simulate miner filling in response
        synapse.reasoning_steps = [
            {"step_id": 0, "reasoning": "2+2=4", "confidence": 0.99}
        ]
        synapse.final_answer = "4"
        synapse.time_taken_ms = 100
        synapse.submission_hash = synapse.compute_submission_hash()

        # Verify hash
        assert verify_submission_hash(synapse) is True

        # Deserialize
        data = synapse.deserialize()
        assert data["final_answer"] == "4"
        assert len(data["steps"]) == 1

    @pytest.mark.asyncio
    async def test_scoring_pipeline(self, scorer):
        """Scorer produces valid dimension scores."""
        task = Task(
            task_id="int-001",
            problem="Prove sqrt(2) is irrational",
            domain=Domain.MATHEMATICS,
            difficulty=5,
        )
        response = {
            "steps": [
                {
                    "step_id": 0,
                    "reasoning": "Assume for contradiction that sqrt(2) = p/q in lowest terms.",
                    "evidence": "Proof by contradiction",
                    "confidence": 0.9,
                },
                {
                    "step_id": 1,
                    "reasoning": "Then p^2 = 2q^2, so p is even. Let p=2k, then q^2=2k^2, so q is even too.",
                    "evidence": "Algebraic manipulation",
                    "confidence": 0.95,
                },
            ],
            "final_answer": "sqrt(2) is irrational",
            "time_taken_ms": 5000,
        }

        dims = await scorer.compute_dimensions(task, response)
        assert isinstance(dims, DimensionScores)
        assert dims.quality > 0
        assert dims.efficiency > 0

        # CMS should be valid
        cms = ScoringEngine.compute_cms(dims)
        assert 0.0 <= cms <= 1.0

    def test_weight_computation(self, weight_setter):
        """Weights are computed and normalized correctly."""
        miner_states = {
            0: {"s_epoch": 0.9, "peb": 0.15},
            1: {"s_epoch": 0.7, "peb": 0.0},
            2: {"s_epoch": 0.5, "peb": 0.05},
            3: {"s_epoch": 0.3, "peb": 0.0},
        }
        uids, weights = weight_setter.compute_weights(miner_states, n=10)

        assert len(uids) == 4
        total = sum(weights)
        assert abs(total - 1.0) < 1e-6

        # Best miner should get highest weight
        weights_list = list(weights)
        assert weights_list[0] > weights_list[1] > weights_list[2] > weights_list[3]

    def test_trap_detection(self):
        """Trap problems correctly identify good and bad answers."""
        tm = TrapManager()
        trap = Task(
            task_id="trap-001",
            problem="What is 2+2?",
            domain=Domain.MATHEMATICS,
            difficulty=2,
            is_trap=True,
            ground_truth_score=1.0,
        )

        # Good answer
        good_score = tm.evaluate_trap_response(trap, "The answer is 4")
        assert good_score == 1.0

        # Bad answer
        bad_score = tm.evaluate_trap_response(trap, "The answer is 5")
        assert bad_score == 0.0

        # Record and check penalty
        tm.record_trap_score(1, good_score)
        tm.record_trap_score(2, bad_score)

        assert tm.get_trap_penalty(1) == 1.0
        assert tm.get_trap_penalty(2) < 1.0

    def test_input_sanitization(self):
        """Oversized inputs are sanitized."""
        synapse = ReasoningTask(
            task_id="test",
            problem="test",
            domain="mathematics",
            difficulty=5,
            reasoning_steps=[{"reasoning": "x" * 20_000, "confidence": 5.0}] * 60,
            final_answer="x" * 100_000,
        )

        InputSanitizer.sanitize_submission(synapse)

        assert len(synapse.reasoning_steps) <= 50
        assert len(synapse.reasoning_steps[0]["reasoning"]) <= 10_000
        assert synapse.reasoning_steps[0]["confidence"] == 1.0
        assert len(synapse.final_answer) <= 50_000

    def test_emission_conservation(self):
        """Emission conservation: all emitted TAO is distributed."""
        total_emission = 100.0
        miners = [
            MinerState(miner_id="m1", s_epoch=0.9, peb=0.15, rank=1, streak=5),
            MinerState(miner_id="m2", s_epoch=0.7, peb=0.0, rank=2, streak=0),
            MinerState(miner_id="m3", s_epoch=0.5, peb=0.05, rank=3, streak=2),
        ]

        miner_pool = total_emission * 0.9
        rewards = ScoringEngine.distribute_miner_emissions(miners, miner_pool)

        assert abs(sum(rewards) - miner_pool) < 1e-6

    def test_multi_domain_task_generation(self, task_manager):
        """Tasks span multiple domains."""
        tasks = task_manager.generate_epoch_tasks(count=24)
        domains = set(t.domain for t in tasks)
        # Should have tasks from multiple domains
        assert len(domains) >= 2


class TestCrashRecovery:
    """Test state persistence and recovery."""

    def test_save_and_restore_state(self, temp_db_path):
        from reasonforge.state.database import StateDatabase

        db = StateDatabase(temp_db_path)

        # Save state
        state = {
            "epoch_id": 10,
            "miner_states": {
                "0": {"s_epoch": 0.9, "peb": 0.1},
                "1": {"s_epoch": 0.7, "peb": 0.0},
            },
        }
        db.save_checkpoint(state, epoch_id=10)

        # Restore
        loaded = db.load_latest_checkpoint()
        assert loaded is not None
        assert loaded["epoch_id"] == 10
        assert loaded["miner_states"]["0"]["s_epoch"] == 0.9

        db.close()
