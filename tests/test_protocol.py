"""
ReasonForge - Protocol & CTS Scoring Tests

Tests for TranslationTask, VerificationResult, HealthCheck synapse classes,
helper functions (create_translation_task, verify_submission_hash), and
Composite Translation Score (CTS) computation.
"""

from __future__ import annotations

import hashlib
import json

from reasonforge.engine import ScoringEngine
from reasonforge.protocol import (
    HealthCheck,
    TranslationTask,
    VerificationResult,
    create_translation_task,
    verify_submission_hash,
)
from reasonforge.types import (
    W_COMPILATION,
    W_COMPLETENESS,
    W_CORRECTNESS,
    W_CTS_EFFICIENCY,
    TranslationScores,
)


# ──────────────────────────────────────────────
# TranslationTask
# ──────────────────────────────────────────────


class TestTranslationTask:
    def test_default_values(self) -> None:
        """TranslationTask has sensible defaults."""
        task = TranslationTask()
        assert task.task_id == ""
        assert task.original_query == ""
        assert task.reasoning_chain is None
        assert task.domain == "mathematics"
        assert task.difficulty == 5
        assert task.proof_level == "standard"
        assert task.timeout_seconds == 300
        assert task.translations is None
        assert task.compilation_status is None
        assert task.full_proof_artifact is None
        assert task.translation_time_ms is None
        assert task.submission_hash is None

    def test_custom_values(self) -> None:
        """Fields can be set at construction time."""
        chain = [{"step": 1, "text": "let x = 1"}]
        task = TranslationTask(
            task_id="t-100",
            original_query="Prove 1+1=2",
            reasoning_chain=chain,
            domain="logic",
            difficulty=8,
            proof_level="formal",
            timeout_seconds=600,
        )
        assert task.task_id == "t-100"
        assert task.original_query == "Prove 1+1=2"
        assert task.reasoning_chain == chain
        assert task.domain == "logic"
        assert task.difficulty == 8
        assert task.proof_level == "formal"
        assert task.timeout_seconds == 600

    def test_deserialize(self) -> None:
        """deserialize() returns the miner-produced payload dict."""
        task = TranslationTask(
            task_id="t-200",
            translations=[{"step": 1, "lean": "sorry"}],
            compilation_status="success",
            full_proof_artifact="artifact-data",
            translation_time_ms=1234,
            submission_hash="abc",
        )
        d = task.deserialize()
        assert d["task_id"] == "t-200"
        assert d["translations"] == [{"step": 1, "lean": "sorry"}]
        assert d["compilation_status"] == "success"
        assert d["full_proof_artifact"] == "artifact-data"
        assert d["translation_time_ms"] == 1234
        assert d["submission_hash"] == "abc"

    def test_deserialize_defaults(self) -> None:
        """deserialize() with default (empty) task has None mutable fields."""
        task = TranslationTask()
        d = task.deserialize()
        assert d["task_id"] == ""
        assert d["translations"] is None
        assert d["compilation_status"] is None
        assert d["submission_hash"] is None


# ──────────────────────────────────────────────
# VerificationResult
# ──────────────────────────────────────────────


class TestVerificationResult:
    def test_default_values(self) -> None:
        """VerificationResult has sensible defaults."""
        vr = VerificationResult()
        assert vr.epoch_id == 0
        assert vr.miner_uid == 0
        assert vr.tasks_translated == 0
        assert vr.compilation_rate == 0.0
        assert vr.epoch_score == 0.0
        assert vr.rank == 0
        assert vr.tao_earned == 0.0

    def test_custom_values(self) -> None:
        """VerificationResult fields can be overridden."""
        vr = VerificationResult(
            epoch_id=42,
            miner_uid=7,
            tasks_translated=10,
            steps_compiled=45,
            steps_total=50,
            compilation_rate=0.90,
            epoch_score=0.85,
            rank=3,
            tao_earned=1.5,
        )
        assert vr.epoch_id == 42
        assert vr.miner_uid == 7
        assert vr.steps_compiled == 45
        assert vr.steps_total == 50

    def test_deserialize(self) -> None:
        """deserialize() returns the correct dict."""
        vr = VerificationResult(epoch_id=1, miner_uid=2, epoch_score=0.75)
        d = vr.deserialize()
        assert d["epoch_id"] == 1
        assert d["miner_uid"] == 2
        assert d["epoch_score"] == 0.75
        assert d["tasks_translated"] == 0
        assert d["tao_earned"] == 0.0


# ──────────────────────────────────────────────
# HealthCheck
# ──────────────────────────────────────────────


class TestHealthCheck:
    def test_default_values(self) -> None:
        """HealthCheck defaults are all None."""
        hc = HealthCheck()
        assert hc.status is None
        assert hc.supported_domains is None
        assert hc.model_info is None
        assert hc.version is None

    def test_custom_values(self) -> None:
        """HealthCheck fields can be set."""
        hc = HealthCheck(
            status="ok",
            supported_domains=["mathematics", "code"],
            model_info="lean4-v1",
            version="0.1.0",
        )
        assert hc.status == "ok"
        assert hc.supported_domains == ["mathematics", "code"]

    def test_deserialize(self) -> None:
        """deserialize() returns the correct dict."""
        hc = HealthCheck(status="ok", version="0.2.0")
        d = hc.deserialize()
        assert d["status"] == "ok"
        assert d["version"] == "0.2.0"
        assert d["supported_domains"] is None
        assert d["model_info"] is None


# ──────────────────────────────────────────────
# create_translation_task helper
# ──────────────────────────────────────────────


class TestCreateTranslationTask:
    def test_factory_basic(self) -> None:
        """create_translation_task() returns a TranslationTask with given values."""
        chain = [{"step": 0, "text": "assume P"}]
        task = create_translation_task(
            task_id="f-001",
            original_query="Prove P -> P",
            reasoning_chain=chain,
        )
        assert isinstance(task, TranslationTask)
        assert task.task_id == "f-001"
        assert task.original_query == "Prove P -> P"
        assert task.reasoning_chain == chain
        assert task.domain == "mathematics"
        assert task.difficulty == 5
        assert task.proof_level == "standard"
        assert task.timeout_seconds == 300

    def test_factory_custom_params(self) -> None:
        """create_translation_task() forwards optional parameters."""
        task = create_translation_task(
            task_id="f-002",
            original_query="Sort a list",
            reasoning_chain=[],
            domain="code",
            difficulty=9,
            proof_level="formal",
            timeout_seconds=120,
        )
        assert task.domain == "code"
        assert task.difficulty == 9
        assert task.proof_level == "formal"
        assert task.timeout_seconds == 120


# ──────────────────────────────────────────────
# verify_submission_hash helper
# ──────────────────────────────────────────────


class TestVerifySubmissionHash:
    def _compute_hash(self, translations: list) -> str:
        """Compute the expected SHA-256 hash for a translations payload."""
        payload = json.dumps(translations, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def test_valid_hash(self) -> None:
        """verify_submission_hash() returns True when hashes match."""
        translations = [{"step": 1, "lean": "sorry"}]
        task = TranslationTask(
            translations=translations,
            submission_hash=self._compute_hash(translations),
        )
        assert verify_submission_hash(task) is True

    def test_invalid_hash(self) -> None:
        """verify_submission_hash() returns False when hashes differ."""
        translations = [{"step": 1, "lean": "sorry"}]
        task = TranslationTask(
            translations=translations,
            submission_hash="0000000000000000000000000000000000000000000000000000000000000000",
        )
        assert verify_submission_hash(task) is False

    def test_missing_translations(self) -> None:
        """verify_submission_hash() returns False when translations is None."""
        task = TranslationTask(translations=None, submission_hash="abc")
        assert verify_submission_hash(task) is False

    def test_missing_hash(self) -> None:
        """verify_submission_hash() returns False when submission_hash is None."""
        task = TranslationTask(
            translations=[{"step": 1}],
            submission_hash=None,
        )
        assert verify_submission_hash(task) is False

    def test_both_missing(self) -> None:
        """verify_submission_hash() returns False when both are None."""
        task = TranslationTask()
        assert verify_submission_hash(task) is False


# ──────────────────────────────────────────────
# CTS — Composite Translation Score
# ──────────────────────────────────────────────


class TestCTS:
    def test_cts_computation(self) -> None:
        """CTS = 0.45*C_comp + 0.30*C_corr + 0.15*C_compl + 0.10*Eff."""
        scores = TranslationScores(
            compilation=0.9,
            correctness=0.8,
            completeness=0.7,
            efficiency=0.6,
        )
        expected = 0.45 * 0.9 + 0.30 * 0.8 + 0.15 * 0.7 + 0.10 * 0.6
        # = 0.405 + 0.24 + 0.105 + 0.06 = 0.81
        result = ScoringEngine.compute_cts(scores)
        assert abs(result - expected) < 1e-10
        assert abs(result - 0.81) < 1e-10

    def test_cts_all_zeros(self) -> None:
        """CTS: All zeros -> 0.0."""
        scores = TranslationScores(
            compilation=0.0,
            correctness=0.0,
            completeness=0.0,
            efficiency=0.0,
        )
        assert ScoringEngine.compute_cts(scores) == 0.0

    def test_cts_all_ones(self) -> None:
        """CTS: All ones -> 1.0 (weights sum to 1.0)."""
        scores = TranslationScores(
            compilation=1.0,
            correctness=1.0,
            completeness=1.0,
            efficiency=1.0,
        )
        expected = 0.45 + 0.30 + 0.15 + 0.10
        result = ScoringEngine.compute_cts(scores)
        assert abs(result - expected) < 1e-10
        assert abs(result - 1.0) < 1e-10

    def test_cts_matches_property(self) -> None:
        """ScoringEngine.compute_cts matches TranslationScores.cts property."""
        scores = TranslationScores(
            compilation=0.85,
            correctness=0.75,
            completeness=0.65,
            efficiency=0.55,
        )
        assert abs(ScoringEngine.compute_cts(scores) - scores.cts) < 1e-10

    def test_cts_weights_sum_to_one(self) -> None:
        """CTS weights must sum to exactly 1.0."""
        total = W_COMPILATION + W_CORRECTNESS + W_COMPLETENESS + W_CTS_EFFICIENCY
        assert abs(total - 1.0) < 1e-10

    def test_cts_compilation_dominates(self) -> None:
        """Compilation has the highest weight (0.45)."""
        assert W_COMPILATION > W_CORRECTNESS
        assert W_COMPILATION > W_COMPLETENESS
        assert W_COMPILATION > W_CTS_EFFICIENCY

    def test_cts_only_compilation(self) -> None:
        """Only compilation=1.0, rest zero -> CTS = 0.45."""
        scores = TranslationScores(compilation=1.0)
        result = ScoringEngine.compute_cts(scores)
        assert abs(result - W_COMPILATION) < 1e-10

    def test_cts_only_correctness(self) -> None:
        """Only correctness=1.0, rest zero -> CTS = 0.30."""
        scores = TranslationScores(correctness=1.0)
        result = ScoringEngine.compute_cts(scores)
        assert abs(result - W_CORRECTNESS) < 1e-10

    def test_cts_only_completeness(self) -> None:
        """Only completeness=1.0, rest zero -> CTS = 0.15."""
        scores = TranslationScores(completeness=1.0)
        result = ScoringEngine.compute_cts(scores)
        assert abs(result - W_COMPLETENESS) < 1e-10

    def test_cts_only_efficiency(self) -> None:
        """Only efficiency=1.0, rest zero -> CTS = 0.10."""
        scores = TranslationScores(efficiency=1.0)
        result = ScoringEngine.compute_cts(scores)
        assert abs(result - W_CTS_EFFICIENCY) < 1e-10

    def test_cts_half_values(self) -> None:
        """CTS with all dimensions at 0.5 -> 0.5."""
        scores = TranslationScores(
            compilation=0.5,
            correctness=0.5,
            completeness=0.5,
            efficiency=0.5,
        )
        expected = 0.5 * (
            W_COMPILATION + W_CORRECTNESS + W_COMPLETENESS + W_CTS_EFFICIENCY
        )
        result = ScoringEngine.compute_cts(scores)
        assert abs(result - expected) < 1e-10
        assert abs(result - 0.5) < 1e-10
