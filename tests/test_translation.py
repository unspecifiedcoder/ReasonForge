"""
ReasonForge - Translation Module Tests

Covers types, prompts, engine, and parsers for the NL-to-Formal pipeline.
"""

from __future__ import annotations

import pytest

from reasonforge.translation.engine import TranslationEngine
from reasonforge.translation.parsers import (
    parse_code_output,
    parse_lean4_output,
    parse_smt_output,
)
from reasonforge.translation.prompts import (
    CODE_SYSTEM_PROMPT,
    LOGIC_SYSTEM_PROMPT,
    MATH_SYSTEM_PROMPT,
)
from reasonforge.translation.types import (
    StepTranslation,
    TranslationRequest,
    TranslationResult,
)


# ──────────────────────────────────────────────
# StepTranslation
# ──────────────────────────────────────────────


class TestStepTranslation:
    def test_creation_with_required_fields(self) -> None:
        """Minimal construction with only required arguments."""
        st = StepTranslation(
            step_id=1,
            original_content="Let x = 1",
            formal_representation="theorem step_1 : True := by trivial",
        )
        assert st.step_id == 1
        assert st.original_content == "Let x = 1"
        assert st.formal_representation == "theorem step_1 : True := by trivial"

    def test_default_values(self) -> None:
        """Optional fields receive sensible defaults."""
        st = StepTranslation(
            step_id=0,
            original_content="",
            formal_representation="",
        )
        assert st.dependencies == []
        assert st.translation_confidence == 0.5
        assert st.compilation_check is False
        assert st.notes is None

    def test_custom_optional_fields(self) -> None:
        """All optional fields can be overridden."""
        st = StepTranslation(
            step_id=3,
            original_content="x > 0",
            formal_representation="lemma pos : 0 < x := by omega",
            dependencies=[1, 2],
            translation_confidence=0.95,
            compilation_check=True,
            notes="verified by Lean",
        )
        assert st.dependencies == [1, 2]
        assert st.translation_confidence == 0.95
        assert st.compilation_check is True
        assert st.notes == "verified by Lean"


# ──────────────────────────────────────────────
# TranslationRequest
# ──────────────────────────────────────────────


class TestTranslationRequest:
    def test_creation_with_defaults(self) -> None:
        """Minimal construction; verify every default."""
        req = TranslationRequest(
            task_id="task-001",
            original_query="Prove that 1 + 1 = 2",
        )
        assert req.task_id == "task-001"
        assert req.original_query == "Prove that 1 + 1 = 2"
        assert req.reasoning_chain == []
        assert req.domain == "mathematics"
        assert req.difficulty == 5
        assert req.proof_level == "standard"

    def test_creation_with_overrides(self) -> None:
        """All fields can be explicitly set."""
        chain = [{"step_id": 1, "content": "step one"}]
        req = TranslationRequest(
            task_id="task-002",
            original_query="Is the code correct?",
            reasoning_chain=chain,
            domain="code",
            difficulty=8,
            proof_level="detailed",
        )
        assert req.reasoning_chain == chain
        assert req.domain == "code"
        assert req.difficulty == 8
        assert req.proof_level == "detailed"


# ──────────────────────────────────────────────
# TranslationResult
# ──────────────────────────────────────────────


class TestTranslationResult:
    def test_creation_with_defaults(self) -> None:
        """Default compilation_status is FAILED, no translations."""
        result = TranslationResult(task_id="task-001")
        assert result.task_id == "task-001"
        assert result.translations == []
        assert result.compilation_status == "FAILED"
        assert result.full_proof is None

    def test_creation_with_translations(self) -> None:
        """Result populated with step translations."""
        steps = [
            StepTranslation(
                step_id=1,
                original_content="a",
                formal_representation="-- a",
            ),
            StepTranslation(
                step_id=2,
                original_content="b",
                formal_representation="-- b",
                compilation_check=True,
            ),
        ]
        result = TranslationResult(
            task_id="task-003",
            translations=steps,
            compilation_status="PASSED",
            full_proof="-- a\n\n-- b",
        )
        assert len(result.translations) == 2
        assert result.compilation_status == "PASSED"
        assert result.full_proof == "-- a\n\n-- b"


# ──────────────────────────────────────────────
# Prompt constants
# ──────────────────────────────────────────────


class TestPrompts:
    def test_math_prompt_exists_and_nonempty(self) -> None:
        assert isinstance(MATH_SYSTEM_PROMPT, str)
        assert len(MATH_SYSTEM_PROMPT) > 0

    def test_code_prompt_exists_and_nonempty(self) -> None:
        assert isinstance(CODE_SYSTEM_PROMPT, str)
        assert len(CODE_SYSTEM_PROMPT) > 0

    def test_logic_prompt_exists_and_nonempty(self) -> None:
        assert isinstance(LOGIC_SYSTEM_PROMPT, str)
        assert len(LOGIC_SYSTEM_PROMPT) > 0

    def test_prompts_are_distinct(self) -> None:
        """Each domain should have a unique prompt."""
        prompts = {MATH_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT, LOGIC_SYSTEM_PROMPT}
        assert len(prompts) == 3


# ──────────────────────────────────────────────
# TranslationEngine — instantiation
# ──────────────────────────────────────────────


class TestTranslationEngineInit:
    def test_default_init(self) -> None:
        engine = TranslationEngine()
        assert engine.backend_name == "openai"
        assert engine.model == "gpt-4o"
        assert engine.domains == ["mathematics", "code", "logic"]

    def test_custom_init(self) -> None:
        engine = TranslationEngine(
            backend="anthropic",
            model="claude-3",
            domains=["logic"],
        )
        assert engine.backend_name == "anthropic"
        assert engine.model == "claude-3"
        assert engine.domains == ["logic"]


# ──────────────────────────────────────────────
# TranslationEngine — translate_step (async)
# ──────────────────────────────────────────────


class TestTranslateStep:
    @pytest.mark.asyncio
    async def test_placeholder_translation(self) -> None:
        """Without a real backend the engine returns a placeholder."""
        engine = TranslationEngine()
        step = {"step_id": 1, "content": "Assume n is even"}
        result = await engine.translate_step(
            step=step,
            domain="mathematics",
            proof_level="standard",
            previous_steps=[],
            original_query="Prove n^2 is even",
        )
        assert isinstance(result, StepTranslation)
        assert result.step_id == 1
        assert result.original_content == "Assume n is even"
        assert result.translation_confidence == 0.0
        assert result.compilation_check is False
        assert "placeholder" in result.formal_representation.lower()

    @pytest.mark.asyncio
    async def test_dependencies_populated(self) -> None:
        """Previous steps are recorded as dependencies."""
        engine = TranslationEngine()
        prev = StepTranslation(
            step_id=1,
            original_content="prior",
            formal_representation="-- prior",
        )
        result = await engine.translate_step(
            step={"step_id": 2, "content": "next"},
            domain="code",
            proof_level="detailed",
            previous_steps=[prev],
            original_query="query",
        )
        assert result.dependencies == [1]

    @pytest.mark.asyncio
    async def test_unknown_domain_falls_back(self) -> None:
        """An unrecognised domain should not raise; it falls back to math."""
        engine = TranslationEngine()
        result = await engine.translate_step(
            step={"step_id": 0, "content": "hello"},
            domain="unknown_domain",
            proof_level="standard",
            previous_steps=[],
            original_query="q",
        )
        assert isinstance(result, StepTranslation)


# ──────────────────────────────────────────────
# TranslationEngine — translate_chain (async)
# ──────────────────────────────────────────────


class TestTranslateChain:
    @pytest.mark.asyncio
    async def test_empty_chain(self) -> None:
        """An empty reasoning chain yields EMPTY status and no proof."""
        engine = TranslationEngine()
        req = TranslationRequest(
            task_id="t-empty",
            original_query="nothing",
            reasoning_chain=[],
        )
        result = await engine.translate_chain(req)
        assert result.task_id == "t-empty"
        assert result.compilation_status == "EMPTY"
        assert result.translations == []
        assert result.full_proof is None

    @pytest.mark.asyncio
    async def test_multi_step_chain(self) -> None:
        """Multiple steps are translated and joined into full_proof."""
        engine = TranslationEngine()
        req = TranslationRequest(
            task_id="t-multi",
            original_query="Prove 2+2=4",
            reasoning_chain=[
                {"step_id": 1, "content": "step one"},
                {"step_id": 2, "content": "step two"},
            ],
        )
        result = await engine.translate_chain(req)
        assert len(result.translations) == 2
        assert result.translations[0].step_id == 1
        assert result.translations[1].step_id == 2
        # Placeholder backend never passes compilation
        assert result.compilation_status == "FAILED"
        assert result.full_proof is not None
        # full_proof joins representations with double newlines
        assert "\n\n" in result.full_proof

    @pytest.mark.asyncio
    async def test_chain_dependencies_accumulate(self) -> None:
        """Each step should list all prior step ids as dependencies."""
        engine = TranslationEngine()
        req = TranslationRequest(
            task_id="t-deps",
            original_query="chain deps",
            reasoning_chain=[
                {"step_id": 1, "content": "a"},
                {"step_id": 2, "content": "b"},
                {"step_id": 3, "content": "c"},
            ],
        )
        result = await engine.translate_chain(req)
        assert result.translations[0].dependencies == []
        assert result.translations[1].dependencies == [1]
        assert result.translations[2].dependencies == [1, 2]


# ──────────────────────────────────────────────
# Parsers — parse_lean4_output
# ──────────────────────────────────────────────


class TestParseLean4Output:
    def test_extracts_lean4_block(self) -> None:
        raw = "Here is the proof:\n```lean4\ntheorem foo := by trivial\n```\nDone."
        assert parse_lean4_output(raw) == "theorem foo := by trivial"

    def test_extracts_lean_block(self) -> None:
        raw = "```lean\nlemma bar := by ring\n```"
        assert parse_lean4_output(raw) == "lemma bar := by ring"

    def test_fallback_to_stripped_text(self) -> None:
        raw = "  no fences here  "
        assert parse_lean4_output(raw) == "no fences here"

    def test_empty_input(self) -> None:
        assert parse_lean4_output("") == ""


# ──────────────────────────────────────────────
# Parsers — parse_code_output
# ──────────────────────────────────────────────


class TestParseCodeOutput:
    def test_extracts_python_block(self) -> None:
        raw = "```python\ndef f(): pass\n```"
        assert parse_code_output(raw) == "def f(): pass"

    def test_extracts_py_block(self) -> None:
        raw = "```py\nx = 1\n```"
        assert parse_code_output(raw) == "x = 1"

    def test_fallback_to_stripped_text(self) -> None:
        raw = "  just plain code  "
        assert parse_code_output(raw) == "just plain code"

    def test_empty_input(self) -> None:
        assert parse_code_output("") == ""


# ──────────────────────────────────────────────
# Parsers — parse_smt_output
# ──────────────────────────────────────────────


class TestParseSMTOutput:
    def test_extracts_smt2_block(self) -> None:
        raw = "```smt2\n(check-sat)\n```"
        assert parse_smt_output(raw) == "(check-sat)"

    def test_extracts_smtlib_block(self) -> None:
        raw = "```smtlib\n(set-logic QF_LIA)\n```"
        assert parse_smt_output(raw) == "(set-logic QF_LIA)"

    def test_extracts_smt_lib_block(self) -> None:
        raw = "```smt-lib\n(declare-const x Int)\n```"
        assert parse_smt_output(raw) == "(declare-const x Int)"

    def test_fallback_to_stripped_text(self) -> None:
        raw = "  (assert true)  "
        assert parse_smt_output(raw) == "(assert true)"

    def test_empty_input(self) -> None:
        assert parse_smt_output("") == ""
