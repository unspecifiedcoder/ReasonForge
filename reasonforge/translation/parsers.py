"""Parsers that extract formal code from LLM markdown responses."""

from __future__ import annotations

import re
from typing import Optional


def _extract_fenced_block(raw_text: str, lang_pattern: str) -> Optional[str]:
    """Extract the first fenced code block whose info-string matches *lang_pattern*.

    Parameters
    ----------
    raw_text:
        The full LLM response (may contain markdown).
    lang_pattern:
        A regex pattern to match the language tag after the opening ````` ```.

    Returns
    -------
    str | None
        The content inside the code fence, or ``None`` if no matching block
        was found.
    """
    pattern = rf"```{lang_pattern}\s*\n(.*?)```"
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_lean4_output(raw_text: str) -> str:
    """Extract Lean 4 code from an LLM response.

    Looks for a fenced code block tagged ``lean4`` or ``lean``.  Falls back to
    returning the raw text (stripped) when no code block is found.

    Parameters
    ----------
    raw_text:
        Raw LLM output that may contain markdown formatting.

    Returns
    -------
    str
        Extracted Lean 4 source code.
    """
    if not raw_text:
        return ""

    result = _extract_fenced_block(raw_text, r"(?:lean4|lean)")
    if result is not None:
        return result

    return raw_text.strip()


def parse_code_output(raw_text: str) -> str:
    """Extract Python code from an LLM response.

    Looks for a fenced code block tagged ``python`` or ``py``.  Falls back to
    returning the raw text (stripped) when no code block is found.

    Parameters
    ----------
    raw_text:
        Raw LLM output that may contain markdown formatting.

    Returns
    -------
    str
        Extracted Python source code.
    """
    if not raw_text:
        return ""

    result = _extract_fenced_block(raw_text, r"(?:python|py)")
    if result is not None:
        return result

    return raw_text.strip()


def parse_smt_output(raw_text: str) -> str:
    """Extract SMT-LIB code from an LLM response.

    Looks for a fenced code block tagged ``smt2``, ``smt-lib``, or ``smtlib``.
    Falls back to returning the raw text (stripped) when no code block is found.

    Parameters
    ----------
    raw_text:
        Raw LLM output that may contain markdown formatting.

    Returns
    -------
    str
        Extracted SMT-LIB source code.
    """
    if not raw_text:
        return ""

    result = _extract_fenced_block(raw_text, r"(?:smt2|smt-lib|smtlib)")
    if result is not None:
        return result

    return raw_text.strip()
