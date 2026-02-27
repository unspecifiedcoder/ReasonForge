"""
ReasonForge - Domain Router

Maps reasoning domains to specialized system prompts and output parsers.
"""

from __future__ import annotations

from typing import Dict

from ..types import Domain

# ──────────────────────────────────────────────
# Domain-Specific System Prompts
# ──────────────────────────────────────────────

DOMAIN_PROMPTS: Dict[str, str] = {
    Domain.MATHEMATICS: """You are a mathematical reasoning engine. For each step in your solution:
1. State your approach clearly
2. Show all formal work with explicit derivations
3. Verify each step's correctness
4. If possible, express proofs in Lean 4 syntax fragments

Structure your response as a series of numbered reasoning steps, each with:
- reasoning: Your detailed work for this step
- evidence: Mathematical justification or references
- confidence: Your confidence in this step (0.0 to 1.0)
- formal_proof_fragment: Optional Lean 4 proof fragment

Conclude with a clear final answer.""",

    Domain.CODE: """You are a code reasoning engine. For each step:
1. Analyze the requirements and constraints
2. Design the solution approach with complexity analysis
3. Implement with clean, well-documented code
4. Include test cases that verify correctness

Structure your response as reasoning steps covering:
- Problem analysis and edge cases
- Algorithm design with time/space complexity
- Implementation with inline comments
- Test cases and verification

Your code_artifact should contain the complete, runnable solution.""",

    Domain.SCIENTIFIC: """You are a scientific reasoning engine. For each step:
1. Formulate hypotheses based on known principles
2. Design the analytical approach or simulation
3. Execute calculations with proper units
4. Validate results against known benchmarks or constraints

Structure your response with steps covering:
- Problem formulation and relevant theory
- Methodology and approach
- Calculations and results
- Validation and uncertainty analysis""",

    Domain.STRATEGIC: """You are a strategic reasoning engine specializing in game theory and optimization. For each step:
1. Model the problem formally (players, strategies, payoffs)
2. Identify equilibrium concepts or optimization framework
3. Solve using appropriate methods (LP, Nash, etc.)
4. Verify the solution and analyze sensitivity

Structure your response with steps covering:
- Problem formalization
- Solution methodology
- Detailed computation
- Solution verification and interpretation""",

    Domain.CAUSAL: """You are a causal reasoning engine. For each step:
1. Construct or analyze the causal DAG
2. Identify confounders, mediators, and instruments
3. Apply do-calculus or appropriate identification strategy
4. Derive the causal estimand

Structure your response with steps covering:
- Causal graph specification
- Identification strategy (backdoor, frontdoor, IV)
- Formal derivation
- Interpretation and assumptions""",

    Domain.ETHICAL: """You are an ethical reasoning engine. For each step:
1. Identify the key moral dimensions and stakeholders
2. Apply multiple ethical frameworks (utilitarian, deontological, virtue ethics, etc.)
3. Analyze tensions and trade-offs between perspectives
4. Synthesize a nuanced conclusion

Structure your response with steps covering:
- Stakeholder and issue analysis
- Framework application (minimum 3 frameworks)
- Comparative analysis of perspectives
- Balanced conclusion with justified reasoning""",
}

# Output schema for structured reasoning
REASONING_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "integer"},
                    "reasoning": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "formal_proof_fragment": {"type": "string"},
                },
                "required": ["step_id", "reasoning", "confidence"],
            },
        },
        "final_answer": {"type": "string"},
        "proof_status": {"type": "string", "enum": ["VERIFIED", "FAILED", "NONE"]},
    },
    "required": ["steps", "final_answer"],
}


class DomainRouter:
    """Routes tasks to domain-specialized prompts and parsers."""

    def __init__(self, supported_domains: list[str] | None = None):
        self.supported_domains = supported_domains or list(DOMAIN_PROMPTS.keys())

    def get_system_prompt(self, domain: str) -> str:
        """Get the system prompt for a domain."""
        # Handle both Domain enum and string
        domain_key = domain if isinstance(domain, str) else domain.value
        for key, prompt in DOMAIN_PROMPTS.items():
            key_val = key.value if hasattr(key, "value") else key
            if key_val == domain_key:
                return prompt
        # Default fallback
        return DOMAIN_PROMPTS[Domain.MATHEMATICS]

    def build_prompt(self, problem: str, domain: str, difficulty: int,
                     context: str | None = None, constraints: str | None = None) -> str:
        """Build the full user prompt for a task."""
        parts = [f"**Problem (Difficulty {difficulty}/10):**\n{problem}"]
        if context:
            parts.append(f"\n**Context:**\n{context}")
        if constraints:
            parts.append(f"\n**Constraints:**\n{constraints}")
        parts.append(
            "\n\nProvide your solution as a structured chain of reasoning steps. "
            "Each step should include your reasoning, supporting evidence, and confidence level."
        )
        return "\n".join(parts)

    def get_schema(self) -> dict:
        """Get the output schema for structured responses."""
        return REASONING_SCHEMA

    def supports_domain(self, domain: str) -> bool:
        """Check if this router supports a given domain."""
        domain_val = domain.value if hasattr(domain, "value") else domain
        supported_vals = [
            d.value if hasattr(d, "value") else d
            for d in self.supported_domains
        ]
        return domain_val in supported_vals
