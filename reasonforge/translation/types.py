"""Data types for the NL-to-Formal translation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepTranslation:
    """A single reasoning step translated into a formal representation."""

    step_id: int
    original_content: str
    formal_representation: str
    dependencies: List[int] = field(default_factory=list)
    translation_confidence: float = 0.5
    compilation_check: bool = False
    notes: Optional[str] = None


@dataclass
class TranslationRequest:
    """A request to translate a reasoning chain into formal proofs."""

    task_id: str
    original_query: str
    reasoning_chain: List[dict] = field(default_factory=list)
    domain: str = "mathematics"
    difficulty: int = 5
    proof_level: str = "standard"


@dataclass
class TranslationResult:
    """The result of translating an entire reasoning chain."""

    task_id: str
    translations: List[StepTranslation] = field(default_factory=list)
    compilation_status: str = "FAILED"
    full_proof: Optional[str] = None
