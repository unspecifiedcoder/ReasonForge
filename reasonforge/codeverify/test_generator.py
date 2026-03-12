# reasonforge/codeverify/test_generator.py
#
# Test Generation Engine for ReasonForge Code Verification
#
# Generates property-based pytest test code from Python source by statically
# analyzing function signatures, type hints, docstrings, and return types
# using the ast module.  No LLM or external API dependency is required.

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────


@dataclass
class ParameterInfo:
    """Describes a single parameter of a function."""

    name: str
    annotation: Optional[str] = None
    has_default: bool = False
    default_value: Optional[str] = None


@dataclass
class FunctionSignature:
    """Describes a parsed function extracted from source code."""

    name: str
    parameters: List[ParameterInfo] = field(default_factory=list)
    return_annotation: Optional[str] = None
    docstring: Optional[str] = None
    line_number: int = 0
    is_async: bool = False
    class_name: Optional[str] = None


@dataclass
class GeneratedTestSuite:
    """The output of the test generator."""

    test_code: str
    test_count: int
    function_signatures: List[FunctionSignature]
    generation_strategy: str


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Maps type-annotation strings to sensible zero/default values for callable
# tests.  Keys are lowercase canonical forms.
_TYPE_TO_DEFAULT: Dict[str, str] = {
    "int": "0",
    "float": "0.0",
    "str": '""',
    "bool": "False",
    "list": "[]",
    "dict": "{}",
    "tuple": "()",
    "set": "set()",
    "bytes": 'b""',
    "bytearray": 'bytearray(b"")',
    "complex": "0j",
    "frozenset": "frozenset()",
    "none": "None",
    "type[none]": "None",
}

# Maps return-type annotation strings to ``isinstance`` type expressions.
_TYPE_TO_ISINSTANCE: Dict[str, str] = {
    "int": "int",
    "float": "float",
    "str": "str",
    "bool": "bool",
    "list": "list",
    "dict": "dict",
    "tuple": "tuple",
    "set": "set",
    "bytes": "bytes",
    "bytearray": "bytearray",
    "complex": "complex",
    "frozenset": "frozenset",
    "none": "type(None)",
}

# Dunder method names to skip during generation.
_SKIP_DUNDERS: frozenset[str] = frozenset(
    {
        "__init__",
        "__repr__",
        "__str__",
        "__del__",
        "__enter__",
        "__exit__",
        "__len__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__iter__",
        "__next__",
        "__contains__",
        "__call__",
        "__hash__",
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__pow__",
        "__and__",
        "__or__",
        "__xor__",
        "__bool__",
        "__bytes__",
        "__format__",
        "__new__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__get__",
        "__set__",
        "__delete__",
        "__init_subclass__",
        "__class_getitem__",
        "__abs__",
        "__neg__",
        "__pos__",
        "__invert__",
        "__complex__",
        "__int__",
        "__float__",
        "__index__",
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__rtruediv__",
        "__rfloordiv__",
        "__rmod__",
        "__rpow__",
        "__rand__",
        "__ror__",
        "__rxor__",
        "__iadd__",
        "__isub__",
        "__imul__",
        "__itruediv__",
        "__ifloordiv__",
        "__imod__",
        "__ipow__",
        "__iand__",
        "__ior__",
        "__ixor__",
        "__reversed__",
        "__copy__",
        "__deepcopy__",
        "__reduce__",
        "__reduce_ex__",
        "__sizeof__",
    }
)

# Boundary values injected for numeric edge-case tests.
_EDGE_VALUES_INT: List[str] = ["0", "-1", "1", "10**9", "-(10**9)"]
_EDGE_VALUES_FLOAT: List[str] = ["0.0", "-1.0", "1.0", "1e9", "-1e9"]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _annotation_to_source(node: Optional[ast.expr]) -> Optional[str]:
    """Convert an AST annotation node back to a source-code string.

    Handles the common annotation forms produced by ``ast.parse``.  Returns
    ``None`` when *node* is ``None``.
    """
    if node is None:
        return None
    return ast.unparse(node)


def _normalize_type(annotation: Optional[str]) -> Optional[str]:
    """Return a lowercase, stripped version of an annotation string."""
    if annotation is None:
        return None
    return annotation.strip().lower()


def _resolve_optional(annotation: Optional[str]) -> Tuple[Optional[str], bool]:
    """Unwrap ``Optional[X]`` into ``(X, True)`` or return ``(annotation, False)``."""
    if annotation is None:
        return None, False

    stripped = annotation.strip()

    # Handle Optional[X]
    match = re.match(r"^Optional\[(.+)\]$", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip(), True

    # Handle Union[X, None] or Union[None, X]
    match = re.match(r"^Union\[(.+)\]$", stripped, re.IGNORECASE)
    if match:
        parts = [p.strip() for p in match.group(1).split(",")]
        non_none = [p for p in parts if p.lower() != "none"]
        has_none = len(non_none) < len(parts)
        if has_none and len(non_none) == 1:
            return non_none[0], True

    return stripped, False


def _default_value_for_annotation(annotation: Optional[str]) -> str:
    """Return a source-code literal suitable as a default value for *annotation*.

    Falls back to ``"None"`` when the annotation is unknown.
    """
    norm = _normalize_type(annotation)
    if norm is None:
        return "None"

    # Direct match
    if norm in _TYPE_TO_DEFAULT:
        return _TYPE_TO_DEFAULT[norm]

    # Parameterised generics: list[...], List[...], etc.
    base = norm.split("[", 1)[0].strip()
    if base in _TYPE_TO_DEFAULT:
        return _TYPE_TO_DEFAULT[base]

    return "None"


def _isinstance_expr_for_annotation(annotation: Optional[str]) -> Optional[str]:
    """Return a Python expression usable inside ``isinstance(result, ...)``.

    Returns ``None`` if the annotation cannot be mapped.
    """
    if annotation is None:
        return None

    inner, is_optional = _resolve_optional(annotation)
    norm = _normalize_type(inner)
    if norm is None:
        return None

    base = norm.split("[", 1)[0].strip()
    type_str = _TYPE_TO_ISINSTANCE.get(base)
    if type_str is None:
        return None

    if is_optional:
        return f"({type_str}, type(None))"
    return type_str


def _extract_doctest_examples(docstring: str) -> List[Tuple[str, str]]:
    """Parse ``>>>`` doctest examples from *docstring*.

    Returns a list of ``(code, expected_output)`` pairs.  Only simple
    single-line examples are extracted (continuation lines starting with
    ``...`` are appended to the code).
    """
    lines = docstring.splitlines()
    examples: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(">>>"):
            code = line[3:].strip()
            i += 1
            # Gather continuation lines
            while i < len(lines) and lines[i].strip().startswith("..."):
                code += "\n" + lines[i].strip()[3:].strip()
                i += 1
            # The next non-blank, non->>> line is the expected result
            if i < len(lines):
                expected = lines[i].strip()
                if expected and not expected.startswith(">>>"):
                    examples.append((code, expected))
                    i += 1
                else:
                    # No expected output line; skip this example
                    pass
            continue
        i += 1
    return examples


def _is_numeric_type(annotation: Optional[str]) -> Tuple[bool, bool]:
    """Return ``(is_int, is_float)`` based on *annotation*."""
    norm = _normalize_type(annotation)
    if norm is None:
        return False, False
    base = norm.split("[", 1)[0].strip()
    return base == "int", base == "float"


# ──────────────────────────────────────────────
# AST Extraction
# ──────────────────────────────────────────────


def _extract_functions(source_code: str) -> List[FunctionSignature]:
    """Parse *source_code* with ``ast.parse`` and return all public functions.

    Iterates top-level statements and recurses into class bodies so that
    every public function and method is captured exactly once.
    """
    tree = ast.parse(source_code)
    return _extract_from_body(tree.body, class_name=None)


def _extract_from_body(
    body: List[ast.stmt],
    class_name: Optional[str],
) -> List[FunctionSignature]:
    """Recursively extract function signatures from a list of AST statements."""
    results: List[FunctionSignature] = []

    for node in body:
        if isinstance(node, ast.ClassDef):
            results.extend(_extract_from_body(node.body, class_name=node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name

            # Skip private and dunder methods
            if name.startswith("_") and not name.startswith("__"):
                continue
            if name in _SKIP_DUNDERS:
                continue
            if name.startswith("__") and name.endswith("__"):
                # Catch any remaining dunders not in the explicit set
                continue

            params = _extract_parameters(node, class_name=class_name)
            return_ann = _annotation_to_source(node.returns)
            docstring = ast.get_docstring(node)

            sig = FunctionSignature(
                name=name,
                parameters=params,
                return_annotation=return_ann,
                docstring=docstring,
                line_number=node.lineno,
                is_async=isinstance(node, ast.AsyncFunctionDef),
                class_name=class_name,
            )
            results.append(sig)

    return results


def _extract_parameters(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    class_name: Optional[str],
) -> List[ParameterInfo]:
    """Extract parameter info from a function node's arguments."""
    args = func_node.args
    params: List[ParameterInfo] = []

    # Positional args (includes positional-only in 3.8+)
    all_args: List[ast.arg] = list(args.posonlyargs) + list(args.args)

    # Number of args that have defaults (aligned from the right)
    num_defaults = len(args.defaults)
    num_args = len(all_args)

    for idx, arg_node in enumerate(all_args):
        name = arg_node.arg

        # Skip 'self' and 'cls' for class methods
        if class_name is not None and name in ("self", "cls"):
            continue

        annotation = _annotation_to_source(arg_node.annotation)

        # Defaults are right-aligned with the positional arg list
        default_idx = idx - (num_args - num_defaults)
        has_default = default_idx >= 0
        default_value: Optional[str] = None
        if has_default:
            default_value = ast.unparse(args.defaults[default_idx])

        params.append(
            ParameterInfo(
                name=name,
                annotation=annotation,
                has_default=has_default,
                default_value=default_value,
            )
        )

    # *args
    if args.vararg is not None:
        params.append(
            ParameterInfo(
                name=f"*{args.vararg.arg}",
                annotation=_annotation_to_source(args.vararg.annotation),
                has_default=False,
                default_value=None,
            )
        )

    # keyword-only args
    for kw_idx, kw_arg in enumerate(args.kwonlyargs):
        annotation = _annotation_to_source(kw_arg.annotation)
        kw_default = args.kw_defaults[kw_idx]
        has_default = kw_default is not None
        default_value = ast.unparse(kw_default) if kw_default is not None else None
        params.append(
            ParameterInfo(
                name=kw_arg.arg,
                annotation=annotation,
                has_default=has_default,
                default_value=default_value,
            )
        )

    # **kwargs
    if args.kwarg is not None:
        params.append(
            ParameterInfo(
                name=f"**{args.kwarg.arg}",
                annotation=_annotation_to_source(args.kwarg.annotation),
                has_default=False,
                default_value=None,
            )
        )

    return params


# ──────────────────────────────────────────────
# Test Code Emitters
# ──────────────────────────────────────────────


def _build_call_args(sig: FunctionSignature) -> str:
    """Build an argument string to call the function with safe default values."""
    parts: List[str] = []
    for param in sig.parameters:
        # Skip *args and **kwargs — just pass nothing for them
        if param.name.startswith("*"):
            continue
        value = _default_value_for_annotation(param.annotation)
        parts.append(f"{param.name}={value}")
    return ", ".join(parts)


def _call_expression(sig: FunctionSignature) -> str:
    """Return the function call expression including class prefix if needed."""
    args = _build_call_args(sig)
    if sig.class_name is not None:
        # We cannot instantiate the class generically, so call as a
        # static/classmethod attempt, or note it requires an instance.
        return f"{sig.class_name}.{sig.name}({args})"
    return f"{sig.name}({args})"


def _callable_params(sig: FunctionSignature) -> List[ParameterInfo]:
    """Return only the parameters that should receive explicit arguments."""
    return [p for p in sig.parameters if not p.name.startswith("*")]


def _emit_callable_test(sig: FunctionSignature) -> str:
    """Strategy 1: call with safe defaults and assert no crash."""
    fn_name = sig.name
    call = _call_expression(sig)
    prefix = "await " if sig.is_async else ""
    async_kw = "async " if sig.is_async else ""
    decorator = "@pytest.mark.asyncio\n" if sig.is_async else ""

    return (
        f"{decorator}"
        f"{async_kw}def test_{fn_name}_callable():\n"
        f'    """Test that {fn_name} is callable with default-typed arguments."""\n'
        f"    result = {prefix}{call}\n"
        f"    assert result is not None or result is None  # did not raise\n"
    )


def _emit_type_check_test(sig: FunctionSignature) -> Optional[str]:
    """Strategy 2: assert return type matches annotation."""
    if sig.return_annotation is None:
        return None

    isinstance_expr = _isinstance_expr_for_annotation(sig.return_annotation)
    if isinstance_expr is None:
        return None

    fn_name = sig.name
    call = _call_expression(sig)
    prefix = "await " if sig.is_async else ""
    async_kw = "async " if sig.is_async else ""
    decorator = "@pytest.mark.asyncio\n" if sig.is_async else ""

    return (
        f"{decorator}"
        f"{async_kw}def test_{fn_name}_return_type():\n"
        f'    """Test that {fn_name} returns the annotated type."""\n'
        f"    result = {prefix}{call}\n"
        f"    assert isinstance(result, {isinstance_expr})\n"
    )


def _emit_none_input_tests(sig: FunctionSignature) -> List[str]:
    """Strategy 3: pass None for each parameter and handle TypeError."""
    tests: List[str] = []
    callable_params = _callable_params(sig)

    if not callable_params:
        return tests

    fn_name = sig.name
    prefix = "await " if sig.is_async else ""
    async_kw = "async " if sig.is_async else ""
    decorator = "@pytest.mark.asyncio\n" if sig.is_async else ""

    for param in callable_params:
        # Build arg list with this param set to None
        parts: List[str] = []
        for p in callable_params:
            if p.name == param.name:
                parts.append(f"{p.name}=None")
            else:
                parts.append(f"{p.name}={_default_value_for_annotation(p.annotation)}")
        args_str = ", ".join(parts)

        if sig.class_name is not None:
            call = f"{sig.class_name}.{fn_name}({args_str})"
        else:
            call = f"{fn_name}({args_str})"

        safe_param_name = param.name.replace("*", "")
        test_name = f"test_{fn_name}_none_{safe_param_name}"

        test = (
            f"{decorator}"
            f"{async_kw}def {test_name}():\n"
            f'    """Test {fn_name} with None for parameter \'{param.name}\'."""\n'
            f"    try:\n"
            f"        result = {prefix}{call}\n"
            f"        # Function accepted None without raising\n"
            f"        assert result is not None or result is None\n"
            f"    except TypeError:\n"
            f"        # Function correctly rejected None input\n"
            f"        pass\n"
        )
        tests.append(test)

    return tests


def _emit_docstring_tests(sig: FunctionSignature) -> List[str]:
    """Strategy 4: extract ``>>>`` doctest examples and generate assertions."""
    tests: List[str] = []
    if sig.docstring is None:
        return tests
    if ">>>" not in sig.docstring:
        return tests

    examples = _extract_doctest_examples(sig.docstring)
    fn_name = sig.name

    for idx, (code, expected) in enumerate(examples):
        test_name = f"test_{fn_name}_docstring_example_{idx}"
        test = (
            f"def {test_name}():\n"
            f'    """Test {fn_name} docstring example {idx}."""\n'
            f"    result = {code}\n"
            f"    assert repr(result) == repr({expected}) or str(result) == str({expected})\n"
        )
        tests.append(test)

    return tests


def _emit_edge_case_tests(sig: FunctionSignature) -> List[str]:
    """Strategy 5: boundary values for numeric parameters."""
    tests: List[str] = []
    callable_params = _callable_params(sig)

    if not callable_params:
        return tests

    fn_name = sig.name
    prefix = "await " if sig.is_async else ""
    async_kw = "async " if sig.is_async else ""
    decorator = "@pytest.mark.asyncio\n" if sig.is_async else ""

    for param in callable_params:
        is_int, is_float = _is_numeric_type(param.annotation)
        if not is_int and not is_float:
            continue

        edge_values = _EDGE_VALUES_INT if is_int else _EDGE_VALUES_FLOAT

        for ev_idx, edge_val in enumerate(edge_values):
            # Build arg list with this param set to the edge value
            parts: List[str] = []
            for p in callable_params:
                if p.name == param.name:
                    parts.append(f"{p.name}={edge_val}")
                else:
                    parts.append(
                        f"{p.name}={_default_value_for_annotation(p.annotation)}"
                    )
            args_str = ", ".join(parts)

            if sig.class_name is not None:
                call = f"{sig.class_name}.{fn_name}({args_str})"
            else:
                call = f"{fn_name}({args_str})"

            safe_param_name = param.name.replace("*", "")
            test_name = f"test_{fn_name}_edge_{safe_param_name}_{ev_idx}"

            test = (
                f"{decorator}"
                f"{async_kw}def {test_name}():\n"
                f'    """Test {fn_name} with edge value {edge_val} for \'{param.name}\'."""\n'
                f"    try:\n"
                f"        result = {prefix}{call}\n"
                f"        assert result is not None or result is None\n"
                f"    except (ValueError, OverflowError, ArithmeticError):\n"
                f"        # Edge value was correctly rejected or caused a math error\n"
                f"        pass\n"
            )
            tests.append(test)

    return tests


# ──────────────────────────────────────────────
# TestGenerator
# ──────────────────────────────────────────────


class TestGenerator:
    """Generates pytest test suites from Python source code via static analysis.

    Uses ``ast.parse`` to inspect function signatures, type hints, docstrings,
    and return annotations.  No LLM or network dependency is required.

    Usage::

        gen = TestGenerator()
        suite = gen.generate(source_code)
        print(suite.test_code)
    """

    def generate(self, source_code: str) -> GeneratedTestSuite:
        """Analyse *source_code* and return a ``GeneratedTestSuite``.

        Parameters
        ----------
        source_code:
            A string containing valid Python source code to analyse.

        Returns
        -------
        GeneratedTestSuite
            The complete generated test suite with metadata.
        """
        signatures = _extract_functions(source_code)
        test_functions: List[str] = []
        strategies_used: List[str] = []

        for sig in signatures:
            # Strategy 1: callable test
            test_functions.append(_emit_callable_test(sig))
            if "callable" not in strategies_used:
                strategies_used.append("callable")

            # Strategy 2: type check test
            type_test = _emit_type_check_test(sig)
            if type_test is not None:
                test_functions.append(type_test)
                if "type_check" not in strategies_used:
                    strategies_used.append("type_check")

            # Strategy 3: none input tests
            none_tests = _emit_none_input_tests(sig)
            if none_tests:
                test_functions.extend(none_tests)
                if "none_input" not in strategies_used:
                    strategies_used.append("none_input")

            # Strategy 4: docstring example tests
            doc_tests = _emit_docstring_tests(sig)
            if doc_tests:
                test_functions.extend(doc_tests)
                if "docstring_example" not in strategies_used:
                    strategies_used.append("docstring_example")

            # Strategy 5: edge case tests
            edge_tests = _emit_edge_case_tests(sig)
            if edge_tests:
                test_functions.extend(edge_tests)
                if "edge_case" not in strategies_used:
                    strategies_used.append("edge_case")

        # Assemble the full test file
        test_code = self._assemble_test_file(source_code, test_functions, signatures)
        strategy_description = (
            "Static AST analysis with strategies: " + ", ".join(strategies_used)
            if strategies_used
            else "No functions found; no tests generated."
        )

        return GeneratedTestSuite(
            test_code=test_code,
            test_count=len(test_functions),
            function_signatures=signatures,
            generation_strategy=strategy_description,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assemble_test_file(
        self,
        source_code: str,
        test_functions: List[str],
        signatures: List[FunctionSignature],
    ) -> str:
        """Combine imports, source, and test bodies into a complete test file."""
        has_async = any(sig.is_async for sig in signatures)
        lines: List[str] = [
            '"""Auto-generated tests produced by ReasonForge TestGenerator."""',
            "",
            "from __future__ import annotations",
            "",
            "import pytest",
        ]

        if has_async:
            lines.append("import pytest_asyncio  # noqa: F401")

        lines += [
            "",
            "",
            "# ---------------------------------------------------------------------------",
            "# Source code under test (embedded so the test file is self-contained)",
            "# ---------------------------------------------------------------------------",
            "",
        ]

        # Embed the original source, dedented to top level
        dedented_source = textwrap.dedent(source_code)
        lines.append(dedented_source.rstrip())

        lines += [
            "",
            "",
            "# ---------------------------------------------------------------------------",
            "# Generated tests",
            "# ---------------------------------------------------------------------------",
            "",
        ]

        for test_fn in test_functions:
            lines.append("")
            lines.append(test_fn.rstrip())
            lines.append("")

        return "\n".join(lines) + "\n"
