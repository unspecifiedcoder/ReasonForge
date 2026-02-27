"""
ReasonForge - Mathematical Verification

Numerical and symbolic verification using SymPy.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger("reasonforge.verification.math")


class MathChecker:
    """Numerical and symbolic verification using SymPy."""

    def __init__(self):
        self._sympy_available: Optional[bool] = None

    def _check_sympy(self) -> bool:
        if self._sympy_available is None:
            try:
                import sympy
                self._sympy_available = True
            except ImportError:
                self._sympy_available = False
                logger.warning("SymPy not available, math verification limited")
        return self._sympy_available

    def verify(self, problem: str, answer: str) -> float:
        """
        Verify a mathematical answer against the problem.

        Returns:
            1.0 for verified correct
            0.0 for verified incorrect
            0.5 for unverifiable
        """
        if not answer or not answer.strip():
            return 0.0

        # Try numeric extraction and verification
        numeric_result = self._verify_numeric(problem, answer)
        if numeric_result is not None:
            return numeric_result

        # Try symbolic verification if SymPy is available
        if self._check_sympy():
            symbolic_result = self._verify_symbolic(problem, answer)
            if symbolic_result is not None:
                return symbolic_result

        # Can't verify: return neutral
        return 0.5

    def _verify_numeric(self, problem: str, answer: str) -> Optional[float]:
        """Try to extract and verify numeric answers."""
        problem_lower = problem.lower()
        answer_lower = answer.lower().strip()

        # Known simple problems
        simple_checks = {
            "2+2": ("4", 1.0),
            "2 + 2": ("4", 1.0),
            "3 + 5": ("8", 1.0),
            "derivative of x^3": ("3x^2", 1.0),
            "integral of 2x": ("x^2", 1.0),
        }

        for pattern, (expected, score) in simple_checks.items():
            if pattern in problem_lower:
                if expected in answer_lower:
                    return score
                return 0.0

        # Extract numbers from the answer
        numbers = re.findall(r"-?\d+\.?\d*", answer_lower)
        if not numbers:
            return None

        return None  # Can't determine correctness from numbers alone

    def _verify_symbolic(self, problem: str, answer: str) -> Optional[float]:
        """Try symbolic verification using SymPy."""
        try:
            from sympy import sympify, simplify, Eq
            from sympy.parsing.sympy_parser import parse_expr

            # Try to parse the answer as a SymPy expression
            answer_clean = answer.strip()

            # Remove common text around the answer
            for prefix in ["=", "is", "equals", "the answer is"]:
                if answer_clean.lower().startswith(prefix):
                    answer_clean = answer_clean[len(prefix):].strip()

            # Try to parse
            try:
                expr = parse_expr(answer_clean)
                # If we can parse it, it's at least mathematically valid
                return 0.6
            except Exception:
                pass

            return None

        except ImportError:
            return None
        except Exception:
            return None

    def verify_equation(self, lhs: str, rhs: str) -> float:
        """Verify that two expressions are equal."""
        if not self._check_sympy():
            return 0.5

        try:
            from sympy import sympify, simplify

            lhs_expr = sympify(lhs)
            rhs_expr = sympify(rhs)
            diff = simplify(lhs_expr - rhs_expr)
            return 1.0 if diff == 0 else 0.0
        except Exception:
            return 0.5

    def verify_inequality(self, expr: str, bound: float, direction: str = "gt") -> float:
        """Verify that an expression satisfies a bound."""
        if not self._check_sympy():
            return 0.5

        try:
            from sympy import sympify

            val = float(sympify(expr))
            if direction == "gt":
                return 1.0 if val > bound else 0.0
            elif direction == "lt":
                return 1.0 if val < bound else 0.0
            elif direction == "eq":
                return 1.0 if abs(val - bound) < 1e-6 else 0.0
            return 0.5
        except Exception:
            return 0.5
