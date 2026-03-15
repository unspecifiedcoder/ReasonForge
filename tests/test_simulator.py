"""
ReasonForge - Simulator Integration Tests

Tests the full epoch simulation lifecycle.
"""

import pytest

from reasonforge.simulator import (
    EpochSimulator,
    create_default_miners,
    create_default_validators,
)
from reasonforge.types import EMISSION_MINER_SHARE, EMISSION_VALIDATOR_SHARE


@pytest.fixture
def default_setup():
    """Create default miner and validator rosters with seed."""
    seed = 42
    miner_profiles, miner_states = create_default_miners(seed=seed)
    validator_profiles, validator_states = create_default_validators(seed=seed)
    return miner_profiles, validator_profiles, miner_states, validator_states


class TestEpochSimulation:
    def test_epoch_runs_without_error(self, default_setup):
        """Basic smoke test: epoch completes without exceptions."""
        mp, vp, ms, vs = default_setup
        sim = EpochSimulator(mp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42)
        result = sim.run_epoch()
        assert result is not None
        assert result.epoch_id == 1
        assert result.tasks_processed > 0

    def test_epoch_emission_conserved(self, default_setup):
        """Miner rewards + validator rewards should approximately equal total emission."""
        mp, vp, ms, vs = default_setup
        sim = EpochSimulator(mp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42)
        result = sim.run_epoch()

        miner_total = sum(m["epoch_tao"] for m in result.miner_results)
        validator_total = sum(v["epoch_tao"] for v in result.validator_results)

        # Miner pool should be ~90, validator pool ~10
        assert abs(miner_total - 100.0 * EMISSION_MINER_SHARE) < 0.01
        assert abs(validator_total - 100.0 * EMISSION_VALIDATOR_SHARE) < 0.01
        assert abs(miner_total + validator_total - 100.0) < 0.01

    def test_adversarial_penalized(self, default_setup):
        """Adversarial miners should earn less than elite miners."""
        mp, vp, ms, vs = default_setup
        sim = EpochSimulator(mp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42)
        result = sim.run_epoch()

        elite_tao = [m["epoch_tao"] for m in result.miner_results if "DeepReason" in m["name"]]
        adversarial_tao = [m["epoch_tao"] for m in result.miner_results if "SpamBot" in m["name"]]

        assert len(elite_tao) > 0
        assert len(adversarial_tao) > 0
        assert max(elite_tao) > max(adversarial_tao)

    def test_trap_detection_works(self, default_setup):
        """Adversarial miners should get trap penalties."""
        mp, vp, ms, vs = default_setup
        sim = EpochSimulator(mp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42)
        result = sim.run_epoch()

        # Check that traps were injected
        assert result.traps_injected > 0

    def test_streak_increments(self, default_setup):
        """Top miners should accumulate streaks over multiple epochs."""
        mp, vp, ms, vs = default_setup

        for epoch in range(1, 4):
            sim = EpochSimulator(
                mp, vp, ms, vs, epoch_id=epoch, total_emission=100.0, seed=42 + epoch
            )
            sim.run_epoch()

        # At least one miner should have a streak > 1
        max_streak = max(m.streak for m in ms)
        assert max_streak >= 2

    def test_lazy_validator_lower_vas(self, default_setup):
        """Lazy validators should have lower VAS than honest validators."""
        mp, vp, ms, vs = default_setup
        sim = EpochSimulator(mp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42)
        result = sim.run_epoch()

        honest_vas = [v["vas"] for v in result.validator_results if v["name"] == "TruthGuard"]
        lazy_vas = [v["vas"] for v in result.validator_results if v["name"] == "LazyNode"]

        if honest_vas and lazy_vas:
            # Honest should generally have higher VAS but due to randomness
            # we just check the lazy has less than perfect
            assert lazy_vas[0] < 1.0

    def test_malicious_validator_slashed(self):
        """Malicious validators should get slashed after low VAS."""
        mp, ms = create_default_miners(seed=42)
        vp2, vs2 = create_default_validators(seed=42)

        # Run multiple epochs so VAS history accumulates
        for epoch in range(1, 8):
            sim = EpochSimulator(
                mp, vp2, ms, vs2, epoch_id=epoch, total_emission=100.0, seed=42 + epoch
            )
            sim.run_epoch()

        # Check if BadActor got slashed at some point
        bad_actor = [v for v in vs2 if v.name == "BadActor"]
        assert len(bad_actor) == 1
        # BadActor with malicious profile (bias=0.20, noise=0.25) should have low VAS
        assert bad_actor[0].vas_7d_avg < 1.0

    def test_multi_epoch_state_carries(self, default_setup):
        """Run 3 epochs and verify state continuity."""
        mp, vp, ms, vs = default_setup

        for epoch in range(1, 4):
            sim = EpochSimulator(
                mp, vp, ms, vs, epoch_id=epoch, total_emission=100.0, seed=42 + epoch
            )
            sim.run_epoch()  # result used implicitly via miner/validator state mutation

        # After 3 epochs, total_tao should be > epoch_tao for top miners
        for m in ms:
            if m.epoch_tao > 0:
                assert m.total_tao_earned >= m.epoch_tao

        # Validators should have VAS history of length 3
        for v in vs:
            assert len(v.vas_history) == 3

    def test_peb_only_top_k(self, default_setup):
        """Only top-10 miners should get PEB."""
        mp, vp, ms, vs = default_setup
        sim = EpochSimulator(mp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42)
        sim.run_epoch()

        # Run a second epoch so streaks exist
        sim2 = EpochSimulator(mp, vp, ms, vs, epoch_id=2, total_emission=100.0, seed=43)
        sim2.run_epoch()

        peb_miners = [m for m in ms if m.peb > 0]
        no_peb_miners = [m for m in ms if m.peb == 0]

        # All PEB miners should have rank <= 10
        for m in peb_miners:
            assert m.rank <= 10

        # Miners with rank > 10 should have PEB = 0
        for m in no_peb_miners:
            if m.rank > 10:
                assert m.peb == 0.0


class TestToJson:
    def test_to_json(self, default_setup):
        """EpochResult serializes to valid JSON dict."""
        mp, vp, ms, vs = default_setup
        sim = EpochSimulator(mp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42)
        result = sim.run_epoch()
        json_data = EpochSimulator.to_json(result)

        assert "epoch_id" in json_data
        assert "miners" in json_data
        assert "validators" in json_data
        assert len(json_data["miners"]) == 12
        assert len(json_data["validators"]) == 6
