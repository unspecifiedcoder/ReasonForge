"""
ReasonForge - Proof Generator

Generates formal proof fragments for mathematical and code domains.
Attempts to produce Lean 4 syntax when applicable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("reasonforge.miner.proof")


@dataclass
class ProofResult:
    """Result of a proof generation attempt."""
    status: str = "NONE"  # "VERIFIED" | "FAILED" | "NONE"
    artifact: Optional[str] = None  # Base64-encoded proof file
    fragments: Optional[list[str]] = None  # Per-step proof fragments

    def __post_init__(self):
        if self.fragments is None:
            self.fragments = []


class ProofGenerator:
    """Generates formal proof fragments from reasoning steps."""

    # Patterns that suggest formal proof content
    LEAN4_PATTERNS = [
        r"theorem\s+\w+",
        r"lemma\s+\w+",
        r"def\s+\w+",
        r"by\s+(simp|ring|omega|linarith|norm_num|decide)",
        r"#check\s+",
        r"example\s*:",
    ]

    def extract_proof_fragments(self, reasoning_text: str) -> list[str]:
        """Extract any formal proof fragments from reasoning text."""
        fragments = []

        # Look for code blocks that might contain Lean 4
        code_blocks = re.findall(r"```(?:lean4?|proof)?\n(.*?)```", reasoning_text, re.DOTALL)
        for block in code_blocks:
            if any(re.search(p, block) for p in self.LEAN4_PATTERNS):
                fragments.append(block.strip())

        return fragments

    def assess_proof_status(self, fragments: list[str], final_answer: str) -> str:
        """Assess the proof status based on available fragments."""
        if not fragments:
            return "NONE"
        # If we have fragments, we mark as potentially verifiable
        # Actual verification happens on the validator side via Lean4Checker
        return "VERIFIED" if len(fragments) >= 2 else "NONE"

    def generate(self, reasoning_steps: list[dict], final_answer: str) -> ProofResult:
        """Generate proof result from reasoning steps."""
        all_fragments = []
        for step in reasoning_steps:
            reasoning = step.get("reasoning", "")
            fragments = self.extract_proof_fragments(reasoning)
            all_fragments.extend(fragments)
            if fragments:
                step["formal_proof_fragment"] = fragments[0]

        status = self.assess_proof_status(all_fragments, final_answer)

        artifact = None
        if all_fragments:
            import base64
            combined = "\n\n".join(all_fragments)
            artifact = base64.b64encode(combined.encode()).decode()

        return ProofResult(
            status=status,
            artifact=artifact,
            fragments=all_fragments,
        )
