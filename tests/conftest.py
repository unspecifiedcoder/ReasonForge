"""
ReasonForge - Shared Test Fixtures

Common fixtures for all test modules.
"""

import os
import sys
import tempfile

import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    from reasonforge.types import Task, Domain, TaskSource
    return Task(
        task_id="test-task-001",
        problem="Prove that sqrt(2) is irrational.",
        domain=Domain.MATHEMATICS,
        difficulty=5,
        timeout_seconds=300,
        source=TaskSource.SYNTHETIC,
    )


@pytest.fixture
def sample_trap_task():
    """Create a sample trap task for testing."""
    from reasonforge.types import Task, Domain, TaskSource
    return Task(
        task_id="test-trap-001",
        problem="What is 2+2?",
        domain=Domain.MATHEMATICS,
        difficulty=2,
        source=TaskSource.TRAP,
        is_trap=True,
        ground_truth_score=1.0,
    )


@pytest.fixture
def sample_reasoning_response():
    """Create a sample ReasoningTask response."""
    from reasonforge.protocol import ReasoningTask
    return ReasoningTask(
        task_id="test-task-001",
        problem="Prove that sqrt(2) is irrational.",
        domain="mathematics",
        difficulty=5,
        reasoning_steps=[
            {
                "step_id": 0,
                "reasoning": "Assume sqrt(2) is rational, so sqrt(2) = p/q where p,q are integers with no common factors.",
                "evidence": "Proof by contradiction setup",
                "confidence": 0.9,
                "formal_proof_fragment": None,
            },
            {
                "step_id": 1,
                "reasoning": "Then 2 = p^2/q^2, so p^2 = 2q^2. This means p^2 is even, so p is even.",
                "evidence": "If p^2 is even then p is even (contrapositive: odd^2 is odd)",
                "confidence": 0.95,
                "formal_proof_fragment": None,
            },
            {
                "step_id": 2,
                "reasoning": "Let p = 2k. Then 4k^2 = 2q^2, so q^2 = 2k^2. Thus q is also even. Contradiction.",
                "evidence": "Both p and q are even contradicts our assumption of no common factors.",
                "confidence": 0.95,
                "formal_proof_fragment": None,
            },
        ],
        final_answer="sqrt(2) is irrational by proof by contradiction.",
        time_taken_ms=5000,
    )
