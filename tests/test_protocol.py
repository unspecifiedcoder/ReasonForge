"""
ReasonForge - Protocol Tests

Tests for Synapse serialization, deserialization, and hash verification.
"""

import json
import pytest

from reasonforge.protocol import (
    ReasoningTask,
    HealthCheck,
    TaskResult,
    verify_submission_hash,
    create_reasoning_task,
)


class TestReasoningTaskSynapse:
    """Test ReasoningTask Synapse."""

    def test_create_empty(self):
        task = ReasoningTask()
        assert task.task_id == ""
        assert task.reasoning_steps is None
        assert task.final_answer is None

    def test_create_with_fields(self):
        task = ReasoningTask(
            task_id="t001",
            problem="Solve x+1=2",
            domain="mathematics",
            difficulty=3,
            timeout_seconds=120,
        )
        assert task.task_id == "t001"
        assert task.difficulty == 3

    def test_deserialize(self):
        task = ReasoningTask(
            task_id="t001",
            problem="Test",
            domain="code",
            difficulty=5,
            final_answer="42",
            reasoning_steps=[{"step_id": 0, "reasoning": "think"}],
        )
        data = task.deserialize()
        assert data["task_id"] == "t001"
        assert data["final_answer"] == "42"
        assert len(data["steps"]) == 1

    def test_submission_hash(self):
        task = ReasoningTask(
            task_id="t001",
            reasoning_steps=[{"step_id": 0, "reasoning": "step1"}],
            final_answer="answer",
        )
        h = task.compute_submission_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_submission_hash_consistency(self):
        task1 = ReasoningTask(
            task_id="t001",
            reasoning_steps=[{"step_id": 0, "reasoning": "step1"}],
            final_answer="answer",
        )
        task2 = ReasoningTask(
            task_id="t001",
            reasoning_steps=[{"step_id": 0, "reasoning": "step1"}],
            final_answer="answer",
        )
        assert task1.compute_submission_hash() == task2.compute_submission_hash()

    def test_submission_hash_changes_with_content(self):
        task1 = ReasoningTask(
            task_id="t001", final_answer="answer1",
        )
        task2 = ReasoningTask(
            task_id="t001", final_answer="answer2",
        )
        assert task1.compute_submission_hash() != task2.compute_submission_hash()

    def test_difficulty_validation(self):
        # Valid range
        task = ReasoningTask(difficulty=1)
        assert task.difficulty == 1
        task = ReasoningTask(difficulty=10)
        assert task.difficulty == 10


class TestHealthCheckSynapse:
    def test_create_empty(self):
        hc = HealthCheck()
        assert hc.status is None

    def test_deserialize(self):
        hc = HealthCheck(
            status="ready",
            supported_domains=["mathematics", "code"],
            version="0.1.0",
        )
        data = hc.deserialize()
        assert data["status"] == "ready"
        assert len(data["supported_domains"]) == 2


class TestTaskResultSynapse:
    def test_create(self):
        tr = TaskResult(epoch_id=5, miner_uid=42, s_epoch=0.85)
        assert tr.epoch_id == 5
        assert tr.miner_uid == 42

    def test_deserialize(self):
        tr = TaskResult(
            epoch_id=1,
            miner_uid=3,
            scores=[{"task_id": "t1", "cms": 0.8}],
            s_epoch=0.75,
            rank=2,
        )
        data = tr.deserialize()
        assert data["epoch_id"] == 1
        assert data["rank"] == 2


class TestVerifySubmissionHash:
    def test_valid_hash(self):
        task = ReasoningTask(
            task_id="t001",
            reasoning_steps=[{"step_id": 0, "reasoning": "test"}],
            final_answer="42",
        )
        task.submission_hash = task.compute_submission_hash()
        assert verify_submission_hash(task) is True

    def test_invalid_hash(self):
        task = ReasoningTask(
            task_id="t001",
            final_answer="42",
            submission_hash="invalid_hash",
        )
        assert verify_submission_hash(task) is False

    def test_no_hash(self):
        task = ReasoningTask(task_id="t001")
        assert verify_submission_hash(task) is False


class TestCreateReasoningTask:
    def test_factory(self):
        task = create_reasoning_task(
            problem="Test problem",
            domain="code",
            difficulty=7,
        )
        assert task.problem == "Test problem"
        assert task.domain == "code"
        assert task.difficulty == 7
        assert len(task.task_id) > 0

    def test_factory_default_id(self):
        t1 = create_reasoning_task()
        t2 = create_reasoning_task()
        assert t1.task_id != t2.task_id
