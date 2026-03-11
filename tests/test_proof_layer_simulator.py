"""Tests for the Proof Layer simulator: translator profiles and CTS-based epochs.

Mirrors the structure of test_simulator.py but exercises
TranslatorProfile, ProofLayerSimulator, and create_default_translators.
"""

from __future__ import annotations

import pytest

from reasonforge.types import (
    EMISSION_MINER_SHARE,
    EMISSION_VALIDATOR_SHARE,
    Domain,
    Task,
    TranslationScores,
)
from reasonforge.simulator import (
    TRANSLATOR_TIERS,
    ProofLayerSimulator,
    TranslatorProfile,
    create_default_translators,
    create_default_validators,
)


# ──────────────────────────────────────────────
# TranslatorProfile
# ──────────────────────────────────────────────


class TestTranslatorTiers:
    def test_all_tiers_exist(self) -> None:
        """All five expected tiers are defined."""
        expected = {"elite", "strong", "mid", "weak", "adversarial"}
        assert set(TRANSLATOR_TIERS.keys()) == expected

    def test_tier_keys(self) -> None:
        """Each tier has the required dimension keys."""
        for tier_name, tier in TRANSLATOR_TIERS.items():
            assert "comp" in tier, f"{tier_name} missing 'comp'"
            assert "corr" in tier, f"{tier_name} missing 'corr'"
            assert "compl" in tier, f"{tier_name} missing 'compl'"
            assert "eff" in tier, f"{tier_name} missing 'eff'"
            assert "var" in tier, f"{tier_name} missing 'var'"

    def test_elite_better_than_adversarial(self) -> None:
        """Elite base scores are strictly higher than adversarial."""
        elite = TRANSLATOR_TIERS["elite"]
        adv = TRANSLATOR_TIERS["adversarial"]
        for key in ("comp", "corr", "compl", "eff"):
            assert elite[key] > adv[key]


class TestTranslatorProfile:
    def test_instantiation(self) -> None:
        tp = TranslatorProfile("t-1", "TestTranslator", "elite", seed=42)
        assert tp.miner_id == "t-1"
        assert tp.name == "TestTranslator"
        assert tp.tier == "elite"

    def test_domain_bonuses_populated(self) -> None:
        tp = TranslatorProfile("t-1", "Test", "mid", seed=42)
        for domain in Domain:
            assert domain in tp.domain_bonuses
            assert -0.05 <= tp.domain_bonuses[domain] <= 0.10

    def test_translate_task_returns_scores_and_submission(self) -> None:
        tp = TranslatorProfile("t-1", "Test", "strong", seed=42)
        task = Task(domain=Domain.MATHEMATICS, difficulty=5)
        scores, submission = tp.translate_task(task)
        assert isinstance(scores, TranslationScores)
        assert 0.0 <= scores.compilation <= 1.0
        assert 0.0 <= scores.correctness <= 1.0
        assert 0.0 <= scores.completeness <= 1.0
        assert 0.0 <= scores.efficiency <= 1.0
        assert submission.miner_id == "t-1"
        assert submission.task_id == task.task_id
        assert len(submission.steps) >= 2
        assert submission.submission_hash != ""

    def test_translate_task_deterministic_with_seed(self) -> None:
        """Same seed produces identical results."""
        task = Task(domain=Domain.CODE, difficulty=7)
        tp1 = TranslatorProfile("t-1", "A", "elite", seed=99)
        tp2 = TranslatorProfile("t-1", "A", "elite", seed=99)
        s1, sub1 = tp1.translate_task(task)
        s2, sub2 = tp2.translate_task(task)
        assert s1.compilation == s2.compilation
        assert s1.correctness == s2.correctness
        assert sub1.submission_hash == sub2.submission_hash

    def test_translate_task_proof_status(self) -> None:
        """Proof status is VERIFIED when compilation > 0.7."""
        # Elite with seed that gives high compilation
        tp = TranslatorProfile("t-1", "Elite", "elite", seed=42)
        task = Task(domain=Domain.MATHEMATICS, difficulty=3)
        scores, submission = tp.translate_task(task)
        # Elite base compilation is 0.95, should almost always exceed 0.7
        if scores.compilation > 0.7:
            assert submission.proof_status == "VERIFIED"
        else:
            assert submission.proof_status == "FAILED"

    def test_adversarial_lower_cts(self) -> None:
        """Adversarial translators produce lower CTS than elite on average."""
        task = Task(domain=Domain.LOGIC, difficulty=5)
        elite_cts = []
        adv_cts = []
        for i in range(20):
            e = TranslatorProfile("e", "E", "elite", seed=i)
            a = TranslatorProfile("a", "A", "adversarial", seed=i + 1000)
            es, _ = e.translate_task(task)
            as_, _ = a.translate_task(task)
            elite_cts.append(es.cts)
            adv_cts.append(as_.cts)
        assert sum(elite_cts) / len(elite_cts) > sum(adv_cts) / len(adv_cts)

    def test_formal_proof_fragment_populated(self) -> None:
        """Steps should have formal_proof_fragment set."""
        tp = TranslatorProfile("t-1", "Test", "mid", seed=42)
        task = Task(domain=Domain.CODE, difficulty=5)
        _, submission = tp.translate_task(task)
        for step in submission.steps:
            assert step.formal_proof_fragment is not None
            assert "stub" in step.formal_proof_fragment.lower()


# ──────────────────────────────────────────────
# create_default_translators
# ──────────────────────────────────────────────


class TestCreateDefaultTranslators:
    def test_returns_correct_counts(self) -> None:
        profiles, states = create_default_translators(seed=42)
        assert len(profiles) == 12
        assert len(states) == 12

    def test_profiles_have_unique_ids(self) -> None:
        profiles, _ = create_default_translators(seed=42)
        ids = [p.miner_id for p in profiles]
        assert len(set(ids)) == len(ids)

    def test_states_match_profiles(self) -> None:
        profiles, states = create_default_translators(seed=42)
        for p, s in zip(profiles, states):
            assert p.miner_id == s.miner_id
            assert p.name == s.name

    def test_deterministic_with_seed(self) -> None:
        p1, s1 = create_default_translators(seed=42)
        p2, s2 = create_default_translators(seed=42)
        for a, b in zip(p1, p2):
            assert a.miner_id == b.miner_id
            assert a.name == b.name
            assert a.tier == b.tier


# ──────────────────────────────────────────────
# ProofLayerSimulator
# ──────────────────────────────────────────────


@pytest.fixture
def proof_layer_setup():
    """Create default translator and validator rosters."""
    seed = 42
    translator_profiles, miner_states = create_default_translators(seed=seed)
    validator_profiles, validator_states = create_default_validators(seed=seed)
    return translator_profiles, validator_profiles, miner_states, validator_states


class TestProofLayerSimulation:
    def test_epoch_runs_without_error(self, proof_layer_setup) -> None:
        """Smoke test: proof-layer epoch completes."""
        tp, vp, ms, vs = proof_layer_setup
        sim = ProofLayerSimulator(
            tp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42
        )
        result = sim.run_epoch()
        assert result is not None
        assert result.epoch_id == 1
        assert result.tasks_processed > 0

    def test_emission_conserved(self, proof_layer_setup) -> None:
        """Translator rewards + validator rewards = total emission."""
        tp, vp, ms, vs = proof_layer_setup
        sim = ProofLayerSimulator(
            tp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42
        )
        result = sim.run_epoch()

        miner_total = sum(m["epoch_tao"] for m in result.miner_results)
        validator_total = sum(v["epoch_tao"] for v in result.validator_results)

        assert abs(miner_total - 100.0 * EMISSION_MINER_SHARE) < 0.01
        assert abs(validator_total - 100.0 * EMISSION_VALIDATOR_SHARE) < 0.01
        assert abs(miner_total + validator_total - 100.0) < 0.01

    def test_adversarial_translators_penalized(self, proof_layer_setup) -> None:
        """Adversarial translators earn less than elite translators."""
        tp, vp, ms, vs = proof_layer_setup
        sim = ProofLayerSimulator(
            tp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42
        )
        result = sim.run_epoch()

        elite_tao = [
            m["epoch_tao"] for m in result.miner_results if "ProofSmith" in m["name"]
        ]
        adv_tao = [
            m["epoch_tao"] for m in result.miner_results if "GarbageForge" in m["name"]
        ]

        assert len(elite_tao) > 0
        assert len(adv_tao) > 0
        assert max(elite_tao) > max(adv_tao)

    def test_trap_detection(self, proof_layer_setup) -> None:
        """Traps are injected in proof-layer epochs."""
        tp, vp, ms, vs = proof_layer_setup
        sim = ProofLayerSimulator(
            tp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42
        )
        result = sim.run_epoch()
        assert result.traps_injected > 0

    def test_multi_epoch_state_carries(self, proof_layer_setup) -> None:
        """State accumulates across multiple proof-layer epochs."""
        tp, vp, ms, vs = proof_layer_setup

        for epoch in range(1, 4):
            sim = ProofLayerSimulator(
                tp, vp, ms, vs, epoch_id=epoch, total_emission=100.0, seed=42 + epoch
            )
            sim.run_epoch()

        # After 3 epochs, total_tao should be >= epoch_tao
        for m in ms:
            if m.epoch_tao > 0:
                assert m.total_tao_earned >= m.epoch_tao

        # Validators should have 3 VAS entries
        for v in vs:
            assert len(v.vas_history) == 3

    def test_streak_increments(self, proof_layer_setup) -> None:
        """Top translators accumulate streaks."""
        tp, vp, ms, vs = proof_layer_setup

        for epoch in range(1, 4):
            sim = ProofLayerSimulator(
                tp, vp, ms, vs, epoch_id=epoch, total_emission=100.0, seed=42 + epoch
            )
            sim.run_epoch()

        max_streak = max(m.streak for m in ms)
        assert max_streak >= 2

    def test_avg_cts_populated(self, proof_layer_setup) -> None:
        """avg_cms field (reused for avg CTS) is non-zero."""
        tp, vp, ms, vs = proof_layer_setup
        sim = ProofLayerSimulator(
            tp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42
        )
        result = sim.run_epoch()
        assert result.avg_cms > 0.0

    def test_result_has_12_translators(self, proof_layer_setup) -> None:
        """Result contains all 12 translator entries."""
        tp, vp, ms, vs = proof_layer_setup
        sim = ProofLayerSimulator(
            tp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42
        )
        result = sim.run_epoch()
        assert len(result.miner_results) == 12

    def test_result_has_6_validators(self, proof_layer_setup) -> None:
        """Result contains all 6 validator entries."""
        tp, vp, ms, vs = proof_layer_setup
        sim = ProofLayerSimulator(
            tp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42
        )
        result = sim.run_epoch()
        assert len(result.validator_results) == 6

    def test_peb_only_top_k(self, proof_layer_setup) -> None:
        """Only top-10 translators get PEB after multiple epochs."""
        tp, vp, ms, vs = proof_layer_setup
        sim = ProofLayerSimulator(
            tp, vp, ms, vs, epoch_id=1, total_emission=100.0, seed=42
        )
        sim.run_epoch()
        sim2 = ProofLayerSimulator(
            tp, vp, ms, vs, epoch_id=2, total_emission=100.0, seed=43
        )
        sim2.run_epoch()

        peb_translators = [m for m in ms if m.peb > 0]
        for m in peb_translators:
            assert m.rank <= 10
