# security_scanner.py — AST-based security vulnerability scanner for ReasonForge.
#
# Parses Python source code into an abstract syntax tree and walks the tree to
# detect common security anti-patterns such as use of eval/exec, dangerous
# imports, SQL injection vectors, hardcoded secrets, and more.  Each finding is
# mapped to a numbered rule (RF-SEC-001 … RF-SEC-010) with severity metadata
# and the offending source line so that downstream tooling can render actionable
# reports.

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Vulnerability:
    """A single security finding produced by the scanner."""

    rule_id: str
    severity: str
    line_number: int
    column: int
    description: str
    code_snippet: str
    category: str


@dataclass
class SecurityReport:
    """Aggregated results of a security scan."""

    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    risk_level: str = "safe"
    total_issues: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEVERITY_ORDER: Dict[str, int] = {
    "safe": 0,
    "info": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "critical": 5,
}

_DANGEROUS_MODULES: Set[str] = {
    "os",
    "subprocess",
    "shutil",
    "socket",
    "ctypes",
    "pickle",
    "marshal",
}

_SECRET_VARIABLE_NAMES: Set[str] = {
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "api_secret",
    "apisecret",
    "token",
    "access_token",
    "secret_key",
    "secretkey",
    "private_key",
    "privatekey",
    "auth_token",
    "authtoken",
    "credentials",
}

_SQL_KEYWORDS: Tuple[str, ...] = (
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "select",
    "insert",
    "update",
    "delete",
    "drop",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_source_line(source_lines: List[str], lineno: int) -> str:
    """Return the source line at *lineno* (1-indexed), or ``""`` if out of range."""
    if 1 <= lineno <= len(source_lines):
        return source_lines[lineno - 1].rstrip("\n\r")
    return ""


def _highest_severity(vulnerabilities: List[Vulnerability]) -> str:
    """Return the highest severity string found across all *vulnerabilities*."""
    if not vulnerabilities:
        return "safe"
    return max(
        vulnerabilities, key=lambda v: _SEVERITY_ORDER.get(v.severity, 0)
    ).severity


def _call_name(node: ast.Call) -> Optional[str]:
    """Extract a simple function name from a Call node, if possible."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _full_call_name(node: ast.Call) -> Optional[str]:
    """Return a dotted name like ``pickle.loads`` from a Call node, if possible."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    if isinstance(func, ast.Name):
        return func.id
    return None


def _collect_with_targets(tree: ast.Module) -> Set[int]:
    """Return the set of AST node ids for ``open()`` calls used as with-item context expressions."""
    ids: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                if (
                    isinstance(item.context_expr, ast.Call)
                    and _call_name(item.context_expr) == "open"
                ):
                    ids.add(id(item.context_expr))
        if isinstance(node, ast.AsyncWith):
            for item in node.items:
                if (
                    isinstance(item.context_expr, ast.Call)
                    and _call_name(item.context_expr) == "open"
                ):
                    ids.add(id(item.context_expr))
    return ids


def _joinedstr_has_sql(node: ast.JoinedStr) -> bool:
    """Return True if an f-string contains any SQL keyword in its literal parts."""
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for kw in _SQL_KEYWORDS:
                if kw in value.value:
                    return True
    return False


def _format_call_has_sql(node: ast.Call) -> bool:
    """Return True if a ``.format()`` call's receiver string contains SQL keywords."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "format":
        return False
    receiver = func.value
    if isinstance(receiver, ast.Constant) and isinstance(receiver.value, str):
        for kw in _SQL_KEYWORDS:
            if kw in receiver.value:
                return True
    return False


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class SecurityScanner:
    """AST-based security vulnerability scanner.

    Usage::

        scanner = SecurityScanner()
        report = scanner.scan(source_code)
        for vuln in report.vulnerabilities:
            print(vuln.rule_id, vuln.description)
    """

    def scan(self, code: str) -> SecurityReport:
        """Parse *code* and return a :class:`SecurityReport`."""
        source_lines: List[str] = code.splitlines()

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            lineno = exc.lineno if exc.lineno is not None else 1
            col = exc.offset if exc.offset is not None else 0
            vuln = Vulnerability(
                rule_id="RF-SEC-SYNTAX",
                severity="critical",
                line_number=lineno,
                column=col,
                description=f"Syntax error prevents analysis: {exc.msg}",
                code_snippet=_get_source_line(source_lines, lineno),
                category="parse_error",
            )
            return SecurityReport(
                vulnerabilities=[vuln],
                risk_level="critical",
                total_issues=1,
                summary="Code contains a syntax error and could not be fully analysed.",
            )

        findings: List[Vulnerability] = []

        # Pre-compute set of open() calls used inside with-statements so we
        # can skip them in RF-SEC-006.
        safe_open_ids = _collect_with_targets(tree)

        for node in ast.walk(tree):
            findings.extend(self._check_node(node, source_lines, safe_open_ids))

        # Sort by line number for deterministic output.
        findings.sort(key=lambda v: (v.line_number, v.column))

        risk = _highest_severity(findings)
        total = len(findings)

        if total == 0:
            summary = "No security issues detected."
        else:
            summary = (
                f"Found {total} security issue{'s' if total != 1 else ''} "
                f"(highest severity: {risk})."
            )

        return SecurityReport(
            vulnerabilities=findings,
            risk_level=risk,
            total_issues=total,
            summary=summary,
        )

    # -- per-node dispatch --------------------------------------------------

    def _check_node(
        self,
        node: ast.AST,
        source_lines: List[str],
        safe_open_ids: Set[int],
    ) -> List[Vulnerability]:
        """Run all rule checks against a single AST *node*."""
        hits: List[Vulnerability] = []

        if isinstance(node, ast.Call):
            hits.extend(self._check_call(node, source_lines, safe_open_ids))

        if isinstance(node, ast.Import):
            hits.extend(self._check_import(node, source_lines))

        if isinstance(node, ast.ImportFrom):
            hits.extend(self._check_import_from(node, source_lines))

        if isinstance(node, ast.ExceptHandler):
            hits.extend(self._check_bare_except(node, source_lines))

        if isinstance(node, ast.JoinedStr):
            hits.extend(self._check_fstring_sql(node, source_lines))

        if isinstance(node, ast.Assign):
            hits.extend(self._check_hardcoded_secret_assign(node, source_lines))

        if isinstance(node, ast.AnnAssign):
            hits.extend(self._check_hardcoded_secret_annassign(node, source_lines))

        if isinstance(node, ast.Assert):
            hits.extend(self._check_assert(node, source_lines))

        return hits

    # -- rule implementations -----------------------------------------------

    def _check_call(
        self,
        node: ast.Call,
        source_lines: List[str],
        safe_open_ids: Set[int],
    ) -> List[Vulnerability]:
        hits: List[Vulnerability] = []
        name = _call_name(node)
        full_name = _full_call_name(node)
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = _get_source_line(source_lines, lineno)

        # RF-SEC-001: eval / exec
        if name in ("eval", "exec"):
            hits.append(
                Vulnerability(
                    rule_id="RF-SEC-001",
                    severity="critical",
                    line_number=lineno,
                    column=col,
                    description=f"Use of {name}() allows arbitrary code execution.",
                    code_snippet=snippet,
                    category="dangerous_builtin",
                )
            )

        # RF-SEC-002: __import__
        if name == "__import__":
            hits.append(
                Vulnerability(
                    rule_id="RF-SEC-002",
                    severity="high",
                    line_number=lineno,
                    column=col,
                    description="Dynamic __import__() call can load arbitrary modules.",
                    code_snippet=snippet,
                    category="dangerous_builtin",
                )
            )

        # RF-SEC-004: getattr / setattr / delattr with dynamic args
        if name in ("getattr", "setattr", "delattr") and node.args:
            # The second argument (attribute name) is dynamic if it is not a
            # constant string.
            attr_arg: Optional[ast.expr] = node.args[1] if len(node.args) >= 2 else None
            is_dynamic = attr_arg is not None and not (
                isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str)
            )
            if is_dynamic:
                hits.append(
                    Vulnerability(
                        rule_id="RF-SEC-004",
                        severity="medium",
                        line_number=lineno,
                        column=col,
                        description=(
                            f"Use of {name}() with a non-literal attribute name "
                            "enables reflection abuse."
                        ),
                        code_snippet=snippet,
                        category="injection",
                    )
                )

        # RF-SEC-005: .format() with SQL keywords
        if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            if _format_call_has_sql(node):
                hits.append(
                    Vulnerability(
                        rule_id="RF-SEC-005",
                        severity="medium",
                        line_number=lineno,
                        column=col,
                        description=(
                            "String .format() call contains SQL keywords — "
                            "potential SQL injection vector."
                        ),
                        code_snippet=snippet,
                        category="injection",
                    )
                )

        # RF-SEC-006: open() outside a with-statement
        if name == "open" and id(node) not in safe_open_ids:
            hits.append(
                Vulnerability(
                    rule_id="RF-SEC-006",
                    severity="medium",
                    line_number=lineno,
                    column=col,
                    description="open() called without a context manager may leak file handles.",
                    code_snippet=snippet,
                    category="resource_abuse",
                )
            )

        # RF-SEC-008: pickle.loads / marshal.loads
        if full_name in ("pickle.loads", "marshal.loads"):
            hits.append(
                Vulnerability(
                    rule_id="RF-SEC-008",
                    severity="high",
                    line_number=lineno,
                    column=col,
                    description=(
                        f"{full_name}() deserialises untrusted data and can "
                        "execute arbitrary code."
                    ),
                    code_snippet=snippet,
                    category="dangerous_builtin",
                )
            )

        return hits

    # -- import rules -------------------------------------------------------

    def _check_import(
        self,
        node: ast.Import,
        source_lines: List[str],
    ) -> List[Vulnerability]:
        hits: List[Vulnerability] = []
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = _get_source_line(source_lines, lineno)
        for alias in node.names:
            top_level = alias.name.split(".")[0]
            if top_level in _DANGEROUS_MODULES:
                hits.append(
                    Vulnerability(
                        rule_id="RF-SEC-003",
                        severity="high",
                        line_number=lineno,
                        column=col,
                        description=(
                            f"Import of dangerous module '{alias.name}' — "
                            "may allow system-level access."
                        ),
                        code_snippet=snippet,
                        category="dangerous_builtin",
                    )
                )
        return hits

    def _check_import_from(
        self,
        node: ast.ImportFrom,
        source_lines: List[str],
    ) -> List[Vulnerability]:
        hits: List[Vulnerability] = []
        if node.module is None:
            return hits
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = _get_source_line(source_lines, lineno)
        top_level = node.module.split(".")[0]
        if top_level in _DANGEROUS_MODULES:
            hits.append(
                Vulnerability(
                    rule_id="RF-SEC-003",
                    severity="high",
                    line_number=lineno,
                    column=col,
                    description=(
                        f"Import from dangerous module '{node.module}' — "
                        "may allow system-level access."
                    ),
                    code_snippet=snippet,
                    category="dangerous_builtin",
                )
            )
        return hits

    # -- bare except --------------------------------------------------------

    def _check_bare_except(
        self,
        node: ast.ExceptHandler,
        source_lines: List[str],
    ) -> List[Vulnerability]:
        if node.type is not None:
            return []
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = _get_source_line(source_lines, lineno)
        return [
            Vulnerability(
                rule_id="RF-SEC-007",
                severity="low",
                line_number=lineno,
                column=col,
                description=(
                    "Bare except clause catches all exceptions including "
                    "SystemExit and KeyboardInterrupt."
                ),
                code_snippet=snippet,
                category="information_leak",
            )
        ]

    # -- f-string SQL -------------------------------------------------------

    def _check_fstring_sql(
        self,
        node: ast.JoinedStr,
        source_lines: List[str],
    ) -> List[Vulnerability]:
        if not _joinedstr_has_sql(node):
            return []
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = _get_source_line(source_lines, lineno)
        return [
            Vulnerability(
                rule_id="RF-SEC-005",
                severity="medium",
                line_number=lineno,
                column=col,
                description=(
                    "f-string contains SQL keywords — potential SQL injection vector."
                ),
                code_snippet=snippet,
                category="injection",
            )
        ]

    # -- hardcoded secrets --------------------------------------------------

    def _check_hardcoded_secret_assign(
        self,
        node: ast.Assign,
        source_lines: List[str],
    ) -> List[Vulnerability]:
        hits: List[Vulnerability] = []
        # Only flag when the value is a string literal.
        if not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            return hits
        for target in node.targets:
            hits.extend(self._secret_from_target(target, source_lines))
        return hits

    def _check_hardcoded_secret_annassign(
        self,
        node: ast.AnnAssign,
        source_lines: List[str],
    ) -> List[Vulnerability]:
        if node.value is None:
            return []
        if not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            return []
        if node.target is None:
            return []
        return self._secret_from_target(node.target, source_lines)

    def _secret_from_target(
        self,
        target: ast.expr,
        source_lines: List[str],
    ) -> List[Vulnerability]:
        hits: List[Vulnerability] = []
        names: List[str] = []
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    names.append(elt.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)

        for var_name in names:
            if var_name.lower() in _SECRET_VARIABLE_NAMES:
                lineno = getattr(target, "lineno", 1)
                col = getattr(target, "col_offset", 0)
                snippet = _get_source_line(source_lines, lineno)
                hits.append(
                    Vulnerability(
                        rule_id="RF-SEC-009",
                        severity="medium",
                        line_number=lineno,
                        column=col,
                        description=(
                            f"Variable '{var_name}' appears to contain a "
                            "hardcoded secret."
                        ),
                        code_snippet=snippet,
                        category="information_leak",
                    )
                )
        return hits

    # -- assert used for validation -----------------------------------------

    def _check_assert(
        self,
        node: ast.Assert,
        source_lines: List[str],
    ) -> List[Vulnerability]:
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = _get_source_line(source_lines, lineno)
        return [
            Vulnerability(
                rule_id="RF-SEC-010",
                severity="low",
                line_number=lineno,
                column=col,
                description=(
                    "Assert statement used for validation — asserts are "
                    "stripped when Python runs with -O."
                ),
                code_snippet=snippet,
                category="information_leak",
            )
        ]
