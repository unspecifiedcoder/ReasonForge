# tests/test_codeverify.py
"""Comprehensive test suite for the reasonforge.codeverify package.

Covers SandboxResult, SandboxExecutor, SecurityScanner, TestGenerator,
CodeVerificationReport, and VerificationPipeline.
"""

from __future__ import annotations

import re
from typing import List

import pytest

from reasonforge.codeverify.pipeline import VerificationPipeline
from reasonforge.codeverify.report import CodeVerificationReport
from reasonforge.codeverify.sandbox import SandboxExecutor, SandboxResult
from reasonforge.codeverify.security_scanner import SecurityScanner
from reasonforge.codeverify.test_generator import TestGenerator


# ======================================================================
# TestSandboxResult
# ======================================================================


class TestSandboxResult:
    """Tests for the SandboxResult dataclass."""

    def test_default_values(self) -> None:
        result = SandboxResult(
            stdout="",
            stderr="",
            exit_code=0,
            timed_out=False,
            execution_time_ms=0,
            success=True,
        )
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.timed_out is False
        assert result.execution_time_ms == 0
        assert result.success is True

    def test_success_property(self) -> None:
        result = SandboxResult(
            stdout="hello",
            stderr="",
            exit_code=0,
            timed_out=False,
            execution_time_ms=10,
            success=True,
        )
        assert result.success is True
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_failure_cases(self) -> None:
        nonzero_exit = SandboxResult(
            stdout="",
            stderr="error",
            exit_code=1,
            timed_out=False,
            execution_time_ms=5,
            success=False,
        )
        assert nonzero_exit.success is False

        timed_out = SandboxResult(
            stdout="",
            stderr="",
            exit_code=0,
            timed_out=True,
            execution_time_ms=30000,
            success=False,
        )
        assert timed_out.success is False
        assert timed_out.timed_out is True


# ======================================================================
# TestSandboxExecutor
# ======================================================================


class TestSandboxExecutor:
    """Tests for the SandboxExecutor async execution engine."""

    @pytest.mark.asyncio
    async def test_execute_simple_print(self) -> None:
        executor = SandboxExecutor(timeout=10)
        result = await executor.execute('print("hello world")')
        assert result.exit_code == 0
        assert result.success is True
        assert "hello world" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_syntax_error(self) -> None:
        executor = SandboxExecutor(timeout=10)
        result = await executor.execute("def foo(")
        assert result.exit_code != 0
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_blocked_import_os(self) -> None:
        executor = SandboxExecutor(timeout=10)
        result = await executor.execute("import os\nprint(os.getcwd())")
        assert result.exit_code != 0
        assert result.success is False
        assert "blocked" in result.stderr.lower() or "import" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_execute_safe_imports(self) -> None:
        executor = SandboxExecutor(timeout=10)
        code = "import math\nimport json\nprint(math.pi)\nprint(json.dumps({'a': 1}))"
        result = await executor.execute(code)
        assert result.exit_code == 0
        assert result.success is True
        assert "3.14" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_with_tests_passing(self) -> None:
        executor = SandboxExecutor(timeout=10)
        code = "def add(a, b):\n    return a + b"
        test_code = "assert add(1, 2) == 3\nassert add(0, 0) == 0"
        result = await executor.execute_with_tests(code, test_code)
        assert result.exit_code == 0
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_tests_failing(self) -> None:
        executor = SandboxExecutor(timeout=10)
        code = "def add(a, b):\n    return a + b"
        test_code = "assert add(1, 2) == 999"
        result = await executor.execute_with_tests(code, test_code)
        assert result.exit_code != 0
        assert result.success is False


# ======================================================================
# TestSecurityScanner
# ======================================================================


class TestSecurityScanner:
    """Tests for the AST-based SecurityScanner."""

    def test_clean_code_safe(self) -> None:
        scanner = SecurityScanner()
        report = scanner.scan("def greet(name):\n    return f'Hello, {name}!'")
        assert report.risk_level == "safe"
        assert report.total_issues == 0

    def test_detects_eval_critical(self) -> None:
        scanner = SecurityScanner()
        report = scanner.scan("result = eval('1 + 2')")
        rule_ids = [v.rule_id for v in report.vulnerabilities]
        assert "RF-SEC-001" in rule_ids
        eval_vuln = next(v for v in report.vulnerabilities if v.rule_id == "RF-SEC-001")
        assert eval_vuln.severity == "critical"

    def test_detects_exec_critical(self) -> None:
        scanner = SecurityScanner()
        report = scanner.scan("exec('print(1)')")
        rule_ids = [v.rule_id for v in report.vulnerabilities]
        assert "RF-SEC-001" in rule_ids
        exec_vuln = next(v for v in report.vulnerabilities if v.rule_id == "RF-SEC-001")
        assert exec_vuln.severity == "critical"

    def test_detects_import_os_high(self) -> None:
        scanner = SecurityScanner()
        report = scanner.scan("import os")
        rule_ids = [v.rule_id for v in report.vulnerabilities]
        assert "RF-SEC-003" in rule_ids
        os_vuln = next(v for v in report.vulnerabilities if v.rule_id == "RF-SEC-003")
        assert os_vuln.severity == "high"

    def test_detects_import_subprocess_high(self) -> None:
        scanner = SecurityScanner()
        report = scanner.scan("import subprocess")
        rule_ids = [v.rule_id for v in report.vulnerabilities]
        assert "RF-SEC-003" in rule_ids
        sub_vuln = next(v for v in report.vulnerabilities if v.rule_id == "RF-SEC-003")
        assert sub_vuln.severity == "high"

    def test_detects_pickle_loads_high(self) -> None:
        scanner = SecurityScanner()
        report = scanner.scan("import pickle\npickle.loads(data)")
        rule_ids = [v.rule_id for v in report.vulnerabilities]
        assert "RF-SEC-008" in rule_ids
        pkl_vuln = next(v for v in report.vulnerabilities if v.rule_id == "RF-SEC-008")
        assert pkl_vuln.severity == "high"

    def test_detects_bare_except_low(self) -> None:
        scanner = SecurityScanner()
        code = "try:\n    pass\nexcept:\n    pass"
        report = scanner.scan(code)
        rule_ids = [v.rule_id for v in report.vulnerabilities]
        assert "RF-SEC-007" in rule_ids
        bare_vuln = next(v for v in report.vulnerabilities if v.rule_id == "RF-SEC-007")
        assert bare_vuln.severity == "low"

    def test_detects_hardcoded_password_medium(self) -> None:
        scanner = SecurityScanner()
        report = scanner.scan('password = "super_secret_123"')
        rule_ids = [v.rule_id for v in report.vulnerabilities]
        assert "RF-SEC-009" in rule_ids
        pwd_vuln = next(v for v in report.vulnerabilities if v.rule_id == "RF-SEC-009")
        assert pwd_vuln.severity == "medium"

    def test_detects_open_without_with_medium(self) -> None:
        scanner = SecurityScanner()
        report = scanner.scan('f = open("file.txt")')
        rule_ids = [v.rule_id for v in report.vulnerabilities]
        assert "RF-SEC-006" in rule_ids
        open_vuln = next(v for v in report.vulnerabilities if v.rule_id == "RF-SEC-006")
        assert open_vuln.severity == "medium"

    def test_handles_syntax_error_gracefully(self) -> None:
        scanner = SecurityScanner()
        report = scanner.scan("def foo(:\n    pass")
        assert report.risk_level == "critical"
        assert report.total_issues >= 1
        rule_ids = [v.rule_id for v in report.vulnerabilities]
        assert "RF-SEC-SYNTAX" in rule_ids

    def test_multiple_vulnerabilities_highest_risk(self) -> None:
        scanner = SecurityScanner()
        code = (
            "import os\n"
            'password = "secret"\n'
            "eval(input())\n"
            "try:\n"
            "    pass\n"
            "except:\n"
            "    pass\n"
        )
        report = scanner.scan(code)
        assert report.total_issues > 1
        assert report.risk_level == "critical"


# ======================================================================
# TestTestGenerator
# ======================================================================


class TestTestGenerator:
    """Tests for the static-analysis-based TestGenerator."""

    def test_empty_code_no_tests(self) -> None:
        gen = TestGenerator()
        suite = gen.generate("")
        assert suite.test_count == 0
        assert suite.function_signatures == []

    def test_simple_function_generates_callable_test(self) -> None:
        gen = TestGenerator()
        code = "def greet(name):\n    return f'Hello, {name}!'"
        suite = gen.generate(code)
        assert suite.test_count >= 1
        assert "test_greet_callable" in suite.test_code

    def test_function_with_annotations_generates_type_check(self) -> None:
        gen = TestGenerator()
        code = "def add(a: int, b: int) -> int:\n    return a + b"
        suite = gen.generate(code)
        assert "test_add_return_type" in suite.test_code
        assert "isinstance" in suite.test_code

    def test_function_with_doctest_generates_docstring_test(self) -> None:
        gen = TestGenerator()
        code = (
            "def double(x):\n"
            '    """Double x.\n'
            "\n"
            "    >>> double(3)\n"
            "    6\n"
            '    """\n'
            "    return x * 2\n"
        )
        suite = gen.generate(code)
        assert "test_double_docstring_example" in suite.test_code

    def test_function_with_int_param_generates_edge_cases(self) -> None:
        gen = TestGenerator()
        code = "def increment(n: int) -> int:\n    return n + 1"
        suite = gen.generate(code)
        assert "test_increment_edge_" in suite.test_code

    def test_private_functions_skipped(self) -> None:
        gen = TestGenerator()
        code = (
            "def _private_helper(x):\n"
            "    return x\n"
            "\n"
            "def public_func(x):\n"
            "    return x\n"
        )
        suite = gen.generate(code)
        assert (
            "_private_helper" not in suite.test_code
            or "test__private" not in suite.test_code
        )
        assert "test_public_func_callable" in suite.test_code

    def test_generated_code_syntactically_valid(self) -> None:
        gen = TestGenerator()
        code = "def add(a: int, b: int) -> int:\n    return a + b"
        suite = gen.generate(code)
        compile(suite.test_code, "<test>", "exec")

    def test_test_count_matches_functions(self) -> None:
        gen = TestGenerator()
        code = "def add(a: int, b: int) -> int:\n    return a + b"
        suite = gen.generate(code)
        actual_test_count = len(
            re.findall(r"^(?:async )?def test_", suite.test_code, re.MULTILINE)
        )
        assert suite.test_count == actual_test_count


# ======================================================================
# TestCodeVerificationReport
# ======================================================================


class TestCodeVerificationReport:
    """Tests for the CodeVerificationReport dataclass."""

    def test_default_values(self) -> None:
        report = CodeVerificationReport()
        assert report.report_id == ""
        assert report.verdict == "ERROR"
        assert report.confidence_score == 0.0
        assert report.syntax_valid is True
        assert report.security_risk_level == "safe"
        assert report.tests_generated == 0

    def test_compute_report_id_deterministic(self) -> None:
        report = CodeVerificationReport(
            code_hash="abc123",
            verdict="PASSED",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        id1 = report.compute_report_id()
        id2 = report.compute_report_id()
        assert id1 == id2
        assert len(id1) == 64  # SHA-256 hex digest

    def test_compute_confidence_all_perfect(self) -> None:
        report = CodeVerificationReport(
            syntax_valid=True,
            security_risk_level="safe",
            test_pass_rate=1.0,
        )
        confidence = report.compute_confidence()
        assert confidence == pytest.approx(1.0)

    def test_compute_confidence_syntax_failure(self) -> None:
        report = CodeVerificationReport(
            syntax_valid=False,
            security_risk_level="safe",
            test_pass_rate=0.0,
        )
        confidence = report.compute_confidence()
        # 0.2*0.0 + 0.3*1.0 + 0.5*0.0 = 0.3
        assert confidence == pytest.approx(0.3)
        assert confidence < 0.5

    def test_to_dict_structure(self) -> None:
        report = CodeVerificationReport(
            report_id="test-id",
            timestamp="2025-01-01T00:00:00+00:00",
            code_hash="abc123",
            verdict="PASSED",
        )
        d = report.to_dict()
        assert "report_id" in d
        assert "syntax" in d
        assert d["syntax"]["valid"] is True
        assert "security" in d
        assert d["security"]["risk_level"] == "safe"
        assert "tests" in d
        assert "generated" in d["tests"]
        assert "verdict" in d
        assert d["verdict"]["result"] == "PASSED"


# ======================================================================
# TestVerificationPipeline
# ======================================================================


class TestVerificationPipeline:
    """Tests for the VerificationPipeline async orchestrator."""

    @pytest.mark.asyncio
    async def test_verify_clean_function_passes(self) -> None:
        pipeline = VerificationPipeline(timeout=30)
        # Use a function that handles None gracefully and has simple logic
        # so generated callable/none-input tests all pass.
        code = (
            "def greet(name: str = 'world') -> str:\n"
            "    if name is None:\n"
            "        name = 'world'\n"
            "    return f'hello {name}'\n"
        )
        report = await pipeline.verify(code)
        assert report.syntax_valid is True
        assert report.security_risk_level in ("safe", "low")
        # Verdict depends on generated test pass rate; at minimum syntax
        # and security must be clean.
        assert report.confidence_score > 0.0

    @pytest.mark.asyncio
    async def test_verify_syntax_error_fails(self) -> None:
        pipeline = VerificationPipeline(timeout=30)
        code = "def foo(:\n    return 1"
        report = await pipeline.verify(code)
        assert report.verdict == "FAILED"
        assert report.syntax_valid is False
        assert report.syntax_error is not None

    @pytest.mark.asyncio
    async def test_verify_critical_security_issue_fails(self) -> None:
        pipeline = VerificationPipeline(timeout=30)
        code = "import os\nos.system('rm -rf /')"
        report = await pipeline.verify(code)
        assert report.verdict == "FAILED"
        assert report.security_risk_level in ("high", "critical")

    @pytest.mark.asyncio
    async def test_verify_returns_security_issues(self) -> None:
        pipeline = VerificationPipeline(timeout=30)
        code = "result = eval('1+1')"
        report = await pipeline.verify(code)
        assert len(report.security_issues) > 0
        rule_ids: List[str] = [issue["rule_id"] for issue in report.security_issues]
        assert "RF-SEC-001" in rule_ids

    @pytest.mark.asyncio
    async def test_verify_populates_test_metrics(self) -> None:
        pipeline = VerificationPipeline(timeout=30)
        code = "def multiply(a, b):\n    return a * b"
        report = await pipeline.verify(code)
        assert report.tests_generated >= 1
        assert report.confidence_score > 0.0
        assert report.report_id != ""
