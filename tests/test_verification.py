"""
ReasonForge - Verification Tests

Tests for Lean4 checker, code sandbox, math checker, and fact checker.
"""

import pytest

from reasonforge.verification.code_sandbox import CodeSandbox
from reasonforge.verification.fact_checker import FactChecker
from reasonforge.verification.lean4_checker import Lean4Checker
from reasonforge.verification.math_checker import MathChecker


class TestMathChecker:
    """Test mathematical verification."""

    @pytest.fixture
    def checker(self):
        return MathChecker()

    def test_verify_simple_addition(self, checker):
        score = checker.verify("What is 2+2?", "4")
        assert score == 1.0

    def test_verify_wrong_answer(self, checker):
        score = checker.verify("What is 2+2?", "5")
        assert score == 0.0

    def test_verify_derivative(self, checker):
        score = checker.verify("Find the derivative of x^3", "3x^2")
        assert score == 1.0

    def test_verify_unverifiable(self, checker):
        score = checker.verify(
            "Prove the Riemann hypothesis",
            "This is a complex proof involving..."
        )
        assert score == 0.5  # Can't verify

    def test_verify_empty_answer(self, checker):
        score = checker.verify("What is 1+1?", "")
        assert score == 0.0


class TestFactChecker:
    """Test factual claim verification."""

    @pytest.fixture
    def checker(self):
        return FactChecker()

    def test_verify_known_fact(self, checker):
        text = "The speed of light is 299792458 m/s in a vacuum."
        score = checker.verify_claims(text)
        assert score > 0.5

    def test_verify_empty(self, checker):
        assert checker.verify_claims("") == 0.5

    def test_check_citations_apa(self, checker):
        text = "Studies show (Smith, 2023) that this is true. Also (Jones et al., 2022)."
        score = checker.check_citations(text)
        assert score > 0.0

    def test_check_citations_none(self, checker):
        text = "This is just a plain statement."
        score = checker.check_citations(text)
        assert score == 0.0

    def test_scientific_claims(self, checker):
        text = ("The hypothesis was tested using a control group. "
                "Data analysis showed significant results with p < 0.05. "
                "The experiment measured 3.14 kg of material.")
        score = checker.verify_scientific_claims("scientific", text)
        assert score > 0.5


class TestLean4Checker:
    """Test Lean 4 proof checker."""

    @pytest.fixture
    def checker(self):
        return Lean4Checker()

    @pytest.mark.asyncio
    async def test_not_available(self, checker):
        """If Lean 4 is not installed, should return 0.5."""
        checker._available = False
        import base64
        proof = base64.b64encode(b"theorem test : True := trivial").decode()
        score = await checker.verify(proof)
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_invalid_base64(self, checker):
        checker._available = True
        score = await checker.verify("not_valid_base64!!!")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_oversized_proof(self, checker):
        checker._available = True
        import base64
        big_proof = base64.b64encode(b"x" * 200_000).decode()
        score = await checker.verify(big_proof)
        assert score == 0.0


class TestCodeSandbox:
    """Test code sandbox."""

    @pytest.fixture
    def sandbox(self):
        return CodeSandbox()

    def test_dangerous_pattern_detection(self, sandbox):
        assert sandbox._contains_dangerous_patterns("import subprocess") is True
        assert sandbox._contains_dangerous_patterns("os.system('rm -rf /')") is True
        assert sandbox._contains_dangerous_patterns("eval(user_input)") is True
        assert sandbox._contains_dangerous_patterns("print('hello')") is False

    def test_parse_pytest_output(self, sandbox):
        logs = "5 passed, 2 failed in 1.23s"
        score = sandbox._parse_test_results(logs)
        assert abs(score - 5 / 7) < 0.01

    def test_parse_unittest_output(self, sandbox):
        logs = "Ran 10 tests\n\nOK"
        score = sandbox._parse_test_results(logs)
        assert score == 1.0

    def test_parse_unittest_failure(self, sandbox):
        logs = "Ran 10 tests\nFAILED (failures=3)"
        score = sandbox._parse_test_results(logs)
        assert abs(score - 0.7) < 0.01

    @pytest.mark.asyncio
    async def test_lint_basic(self, sandbox):
        import base64
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    try:
        return f"Hello, {name}!"
    except Exception as e:
        return str(e)
'''
        code_b64 = base64.b64encode(code.encode()).decode()
        score = await sandbox.lint(code_b64)
        assert score > 0.5

    @pytest.mark.asyncio
    async def test_lint_minimal(self, sandbox):
        import base64
        code = "x=1"
        code_b64 = base64.b64encode(code.encode()).decode()
        score = await sandbox.lint(code_b64)
        assert score < 0.5
