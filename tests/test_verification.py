"""Tests for the ReasonForge verification pipeline.

Covers verdict data-classes, Lean4/Code/FOL verifiers, the step
dependency graph, and the cross-validator consensus logic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from reasonforge.translation.types import StepTranslation
from reasonforge.verification.code_verifier import CodeVerifier
from reasonforge.verification.cross_validator import CrossValidator
from reasonforge.verification.fol_verifier import FOLVerifier
from reasonforge.verification.lean4_verifier import Lean4Verifier
from reasonforge.verification.process_supervisor import StepDependencyGraph
from reasonforge.verification.verdict import (
    FailureReport,
    StepVerdict,
    VerificationVerdict,
)


# ======================================================================
# Helpers
# ======================================================================


def _make_translations(codes: List[str]) -> List[StepTranslation]:
    """Build a list of ``StepTranslation`` objects from code strings."""
    return [
        StepTranslation(
            step_id=i,
            original_content=f"step {i}",
            formal_representation=code,
        )
        for i, code in enumerate(codes, start=1)
    ]


# ======================================================================
# 1 – StepVerdict
# ======================================================================


class TestStepVerdict:
    def test_creation_defaults(self) -> None:
        sv = StepVerdict(step_id=1, verified=True)
        assert sv.step_id == 1
        assert sv.verified is True
        assert sv.error_message is None
        assert sv.formal_representation == ""
        assert sv.details == {}

    def test_creation_with_error(self) -> None:
        sv = StepVerdict(step_id=2, verified=False, error_message="bad proof")
        assert sv.verified is False
        assert sv.error_message == "bad proof"

    def test_creation_with_details(self) -> None:
        sv = StepVerdict(
            step_id=3,
            verified=True,
            formal_representation="x + 1 = 2",
            details={"solver": "z3"},
        )
        assert sv.formal_representation == "x + 1 = 2"
        assert sv.details["solver"] == "z3"


# ======================================================================
# 2 – FailureReport
# ======================================================================


class TestFailureReport:
    def test_creation_defaults(self) -> None:
        fr = FailureReport(failed_step_id=5)
        assert fr.failed_step_id == 5
        assert fr.original_reasoning == ""
        assert fr.formal_translation == ""
        assert fr.verification_error == ""
        assert fr.suggested_fix is None
        assert fr.cascade_impact == []
        assert fr.last_valid_step == 0
        assert fr.partial_correctness == 0.0

    def test_creation_full(self) -> None:
        fr = FailureReport(
            failed_step_id=3,
            original_reasoning="By induction",
            formal_translation="theorem foo : ...",
            verification_error="type mismatch",
            suggested_fix="Check base case",
            cascade_impact=[4, 5, 6],
            last_valid_step=2,
            partial_correctness=0.4,
        )
        assert fr.cascade_impact == [4, 5, 6]
        assert fr.partial_correctness == 0.4


# ======================================================================
# 3 – VerificationVerdict
# ======================================================================


class TestVerificationVerdict:
    def test_creation_defaults(self) -> None:
        vv = VerificationVerdict(task_id="t-1")
        assert vv.task_id == "t-1"
        assert vv.overall == "FAILED"
        assert vv.step_verdicts == []
        assert vv.total_steps == 0
        assert vv.verified_steps == 0
        assert vv.failure_points == []
        assert vv.failure_report is None
        assert vv.domain == "mathematics"
        assert vv.verdict_hash == ""

    def test_compute_verdict_hash_deterministic(self) -> None:
        sv1 = StepVerdict(step_id=1, verified=True)
        sv2 = StepVerdict(step_id=2, verified=False)
        vv = VerificationVerdict(
            task_id="t-2",
            overall="FAILED",
            step_verdicts=[sv1, sv2],
        )
        h1 = vv.compute_verdict_hash()
        h2 = vv.compute_verdict_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_verdict_hash_changes_with_data(self) -> None:
        vv_a = VerificationVerdict(task_id="a", overall="PASSED")
        vv_b = VerificationVerdict(task_id="b", overall="PASSED")
        assert vv_a.compute_verdict_hash() != vv_b.compute_verdict_hash()


# ======================================================================
# 4–5 – Lean4Verifier
# ======================================================================


class TestLean4Verifier:
    def test_instantiation(self) -> None:
        lv = Lean4Verifier(lean_toolchain="test/lean:v1", timeout=60)
        assert lv.lean_toolchain == "test/lean:v1"
        assert lv.timeout == 60
        assert lv._available is None

    def test_check_available_caches(self) -> None:
        lv = Lean4Verifier()
        # Force-set the cache so we don't depend on the host environment.
        lv._available = False
        assert lv._check_available() is False
        lv._available = True
        assert lv._check_available() is True

    @pytest.mark.asyncio
    async def test_verify_chain_lean_not_available(self) -> None:
        lv = Lean4Verifier()
        lv._available = False
        translations = _make_translations(["theorem t1 : True := trivial"])
        result = await lv.verify_chain("task-lean", translations)

        assert result.task_id == "task-lean"
        assert result.overall == "FAILED"
        assert result.total_steps == 1
        assert result.verified_steps == 0
        assert len(result.step_verdicts) == 1
        assert result.step_verdicts[0].error_message == "Lean 4 not installed"
        assert result.raw_output is not None
        assert "not found" in result.raw_output

    @pytest.mark.asyncio
    async def test_verify_chain_lean_not_available_multiple_steps(self) -> None:
        lv = Lean4Verifier()
        lv._available = False
        translations = _make_translations(["step a", "step b", "step c"])
        result = await lv.verify_chain("task-lean-multi", translations)

        assert result.total_steps == 3
        assert len(result.failure_points) == 3
        for sv in result.step_verdicts:
            assert sv.verified is False


# ======================================================================
# 6–9 – CodeVerifier
# ======================================================================


class TestCodeVerifier:
    def test_instantiation(self) -> None:
        cv = CodeVerifier(timeout=30)
        assert cv.timeout == 30

    @pytest.mark.asyncio
    async def test_verify_chain_valid_python(self) -> None:
        cv = CodeVerifier()
        translations = _make_translations(["x = 1 + 2\ny = x * 3"])
        result = await cv.verify_chain("task-code-ok", translations)

        assert result.overall == "PASSED"
        assert result.verified_steps == 1
        assert result.total_steps == 1
        assert result.domain == "code"
        assert result.step_verdicts[0].verified is True

    @pytest.mark.asyncio
    async def test_verify_chain_syntax_error(self) -> None:
        cv = CodeVerifier()
        translations = _make_translations(["def foo(:\n  pass"])
        result = await cv.verify_chain("task-code-syn", translations)

        assert result.overall == "FAILED"
        assert result.verified_steps == 0
        assert result.step_verdicts[0].verified is False
        assert "SyntaxError" in (result.step_verdicts[0].error_message or "")

    @pytest.mark.asyncio
    async def test_verify_chain_dangerous_import_os(self) -> None:
        cv = CodeVerifier()
        translations = _make_translations(["import os\nos.system('ls')"])
        result = await cv.verify_chain("task-code-danger", translations)

        assert result.overall == "FAILED"
        assert result.step_verdicts[0].verified is False
        assert "Dangerous pattern" in (result.step_verdicts[0].error_message or "")

    @pytest.mark.asyncio
    async def test_verify_chain_dangerous_eval(self) -> None:
        cv = CodeVerifier()
        translations = _make_translations(["result = eval('1+1')"])
        result = await cv.verify_chain("task-code-eval", translations)

        assert result.overall == "FAILED"
        assert "Dangerous pattern" in (result.step_verdicts[0].error_message or "")

    @pytest.mark.asyncio
    async def test_verify_chain_empty_representation(self) -> None:
        cv = CodeVerifier()
        translations = _make_translations(["   "])
        result = await cv.verify_chain("task-code-empty", translations)

        assert result.overall == "FAILED"
        assert "Empty formal representation" in (
            result.step_verdicts[0].error_message or ""
        )

    @pytest.mark.asyncio
    async def test_verify_chain_mixed_steps(self) -> None:
        cv = CodeVerifier()
        translations = _make_translations(
            [
                "x = 42",  # valid
                "import subprocess",  # dangerous
                "y = x + 1",  # valid
            ]
        )
        result = await cv.verify_chain("task-code-mixed", translations)

        assert result.overall == "FAILED"
        assert result.total_steps == 3
        assert result.verified_steps == 2
        assert len(result.failure_points) == 1
        assert result.failure_points[0].step_id == 2


# ======================================================================
# 10–11 – FOLVerifier
# ======================================================================


class TestFOLVerifier:
    def test_instantiation(self) -> None:
        fv = FOLVerifier(solver_timeout=10)
        assert fv.solver_timeout == 10
        assert fv._z3_available is None

    def test_check_z3_caches(self) -> None:
        fv = FOLVerifier()
        fv._z3_available = False
        assert fv._check_z3() is False
        fv._z3_available = True
        assert fv._check_z3() is True

    @pytest.mark.asyncio
    async def test_verify_chain_z3_not_available(self) -> None:
        fv = FOLVerifier()
        fv._z3_available = False
        translations = _make_translations(["(assert (= x 1))"])
        result = await fv.verify_chain("task-fol", translations)

        assert result.task_id == "task-fol"
        assert result.overall == "FAILED"
        assert result.total_steps == 1
        assert result.verified_steps == 0
        assert result.step_verdicts[0].error_message == "Z3 not installed"
        assert result.raw_output is not None
        assert "not found" in result.raw_output

    @pytest.mark.asyncio
    async def test_verify_chain_z3_not_available_multiple(self) -> None:
        fv = FOLVerifier()
        fv._z3_available = False
        translations = _make_translations(["(check-sat)", "(assert true)"])
        result = await fv.verify_chain("task-fol-multi", translations)

        assert result.total_steps == 2
        assert len(result.failure_points) == 2
        for sv in result.step_verdicts:
            assert sv.verified is False


# ======================================================================
# 12–14 – StepDependencyGraph
# ======================================================================


class TestStepDependencyGraph:
    # -- construction & DAG validation --

    def test_empty_graph_is_valid(self) -> None:
        g = StepDependencyGraph()
        assert g.validate_dag() is True

    def test_single_node_no_edges(self) -> None:
        g = StepDependencyGraph(steps={1: "a"}, edges={})
        assert g.validate_dag() is True

    def test_linear_chain_is_valid(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a", 2: "b", 3: "c"},
            edges={2: [1], 3: [2]},
        )
        assert g.validate_dag() is True

    def test_cycle_detected(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a", 2: "b"},
            edges={1: [2], 2: [1]},
        )
        assert g.validate_dag() is False

    def test_self_loop_detected(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a"},
            edges={1: [1]},
        )
        assert g.validate_dag() is False

    def test_invalid_dependency_reference(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a"},
            edges={1: [99]},
        )
        assert g.validate_dag() is False

    # -- topological sort --

    def test_topological_sort_linear(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a", 2: "b", 3: "c"},
            edges={2: [1], 3: [2]},
        )
        order = g.get_verification_order()
        assert order == [1, 2, 3]

    def test_topological_sort_diamond(self) -> None:
        # 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
        g = StepDependencyGraph(
            steps={1: "a", 2: "b", 3: "c", 4: "d"},
            edges={2: [1], 3: [1], 4: [2, 3]},
        )
        order = g.get_verification_order()
        assert order[0] == 1
        assert order[-1] == 4
        assert set(order) == {1, 2, 3, 4}

    def test_topological_sort_raises_on_cycle(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a", 2: "b"},
            edges={1: [2], 2: [1]},
        )
        with pytest.raises(ValueError, match="cycle"):
            g.get_verification_order()

    # -- invalidation cascade --

    def test_invalidation_cascade_linear(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a", 2: "b", 3: "c"},
            edges={2: [1], 3: [2]},
        )
        cascade = g.invalidation_cascade(1)
        assert cascade == {2, 3}

    def test_invalidation_cascade_leaf_has_no_impact(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a", 2: "b", 3: "c"},
            edges={2: [1], 3: [2]},
        )
        cascade = g.invalidation_cascade(3)
        assert cascade == set()

    def test_invalidation_cascade_diamond(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a", 2: "b", 3: "c", 4: "d"},
            edges={2: [1], 3: [1], 4: [2, 3]},
        )
        cascade = g.invalidation_cascade(1)
        assert cascade == {2, 3, 4}

    def test_invalidation_cascade_partial(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a", 2: "b", 3: "c", 4: "d"},
            edges={2: [1], 3: [1], 4: [2, 3]},
        )
        cascade = g.invalidation_cascade(2)
        assert cascade == {4}

    def test_invalidation_cascade_excludes_failed_step(self) -> None:
        g = StepDependencyGraph(
            steps={1: "a", 2: "b"},
            edges={2: [1]},
        )
        cascade = g.invalidation_cascade(1)
        assert 1 not in cascade


# ======================================================================
# 15–17 – CrossValidator
# ======================================================================


class TestCrossValidator:
    @staticmethod
    def _make_verdict(
        task_id: str,
        step_results: List[bool],
    ) -> VerificationVerdict:
        svs = [
            StepVerdict(
                step_id=i,
                verified=v,
                formal_representation=f"step-{i}",
            )
            for i, v in enumerate(step_results, start=1)
        ]
        verified_count = sum(1 for s in svs if s.verified)
        overall = "PASSED" if verified_count == len(svs) and svs else "FAILED"
        return VerificationVerdict(
            task_id=task_id,
            overall=overall,
            step_verdicts=svs,
            total_steps=len(svs),
            verified_steps=verified_count,
        )

    def test_unanimous_pass(self) -> None:
        cv = CrossValidator()
        verdicts: Dict[int, Optional[VerificationVerdict]] = {
            1: self._make_verdict("t", [True, True]),
            2: self._make_verdict("t", [True, True]),
            3: self._make_verdict("t", [True, True]),
        }
        result = cv.cross_validate(verdicts)
        assert result.overall == "PASSED"
        assert result.verified_steps == 2
        assert result.translator_uids == [1, 2, 3]
        for sv in result.step_verdicts:
            assert sv.verified is True
            assert sv.details["yes_votes"] == 3

    def test_majority_pass(self) -> None:
        cv = CrossValidator()
        # 2 out of 3 say step 1 passes, only 1 says step 2 passes
        verdicts: Dict[int, Optional[VerificationVerdict]] = {
            1: self._make_verdict("t", [True, True]),
            2: self._make_verdict("t", [True, False]),
            3: self._make_verdict("t", [False, False]),
        }
        result = cv.cross_validate(verdicts)
        # step 1: 2 yes out of 3 -> verified (2 > 1.5)
        assert result.step_verdicts[0].verified is True
        # step 2: 1 yes out of 3 -> not verified (1 <= 1.5)
        assert result.step_verdicts[1].verified is False
        assert result.overall == "FAILED"

    def test_unanimous_fail(self) -> None:
        cv = CrossValidator()
        verdicts: Dict[int, Optional[VerificationVerdict]] = {
            1: self._make_verdict("t", [False]),
            2: self._make_verdict("t", [False]),
        }
        result = cv.cross_validate(verdicts)
        assert result.overall == "FAILED"
        assert result.verified_steps == 0
        assert result.step_verdicts[0].details["yes_votes"] == 0

    def test_no_valid_verdicts(self) -> None:
        cv = CrossValidator()
        verdicts: Dict[int, Optional[VerificationVerdict]] = {1: None, 2: None, 3: None}
        result = cv.cross_validate(verdicts)
        assert result.overall == "FAILED"
        assert result.task_id == "consensus"
        assert result.raw_output is not None
        assert "No valid" in result.raw_output

    def test_some_none_verdicts_ignored(self) -> None:
        cv = CrossValidator()
        verdicts: Dict[int, Optional[VerificationVerdict]] = {
            1: self._make_verdict("t", [True]),
            2: None,
            3: self._make_verdict("t", [True]),
        }
        result = cv.cross_validate(verdicts)
        # 2 valid miners, both say step 1 is True -> verified
        assert result.overall == "PASSED"
        assert result.verified_steps == 1
        assert result.translator_uids == [1, 3]

    def test_exactly_half_does_not_pass(self) -> None:
        """Threshold is strict majority (> N/2), not >=."""
        cv = CrossValidator()
        verdicts: Dict[int, Optional[VerificationVerdict]] = {
            1: self._make_verdict("t", [True]),
            2: self._make_verdict("t", [False]),
        }
        result = cv.cross_validate(verdicts)
        # 1 yes out of 2 valid miners: 1 > 1.0 is False
        assert result.step_verdicts[0].verified is False
        assert result.overall == "FAILED"
