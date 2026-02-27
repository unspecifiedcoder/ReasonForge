"""
ReasonForge - Reasoning Engine

Orchestrates multi-step reasoning using pluggable LLM backends.
Routes tasks through domain-specific prompts and parses structured output.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .backends.base import LLMBackend
from .domain_router import DomainRouter
from .proof_generator import ProofGenerator

logger = logging.getLogger("reasonforge.miner.reasoning")


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""

    step_id: int = 0
    reasoning: str = ""
    evidence: str = ""
    confidence: float = 0.0
    formal_proof_fragment: Optional[str] = None


@dataclass
class ReasoningResult:
    """Complete result of a reasoning task."""

    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    proof_status: Optional[str] = None
    proof_artifact: Optional[str] = None
    code_artifact: Optional[str] = None
    time_taken_ms: int = 0


class ReasoningEngine:
    """
    Orchestrates multi-step reasoning:
    1. Build domain-specific system prompt
    2. Request chain-of-thought from LLM
    3. Parse structured reasoning steps
    4. Attempt formal proof generation (math/code domains)
    5. Return structured result
    """

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o",
        domains: list[str] | None = None,
        api_key: str | None = None,
    ):
        self.domain_router = DomainRouter(domains)
        self.proof_generator = ProofGenerator()
        self.backend = self._create_backend(backend, model, api_key)

    def _create_backend(
        self, backend_type: str, model: str, api_key: str | None = None
    ) -> LLMBackend:
        """Create the appropriate LLM backend."""
        if backend_type == "openai":
            from .backends.openai_backend import OpenAIBackend

            return OpenAIBackend(model=model, api_key=api_key)
        elif backend_type == "anthropic":
            from .backends.anthropic_backend import AnthropicBackend

            return AnthropicBackend(model=model, api_key=api_key)
        elif backend_type == "local":
            from .backends.local_backend import LocalBackend

            return LocalBackend(model=model)
        elif backend_type == "agent":
            from .backends.agent_backend import AgentBackend

            return AgentBackend(model=model, api_key=api_key)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    async def solve(
        self,
        problem: str,
        domain: str,
        difficulty: int = 5,
        context: str | None = None,
        constraints: str | None = None,
        timeout: int = 300,
    ) -> ReasoningResult:
        """Execute multi-step reasoning for a task."""
        start_time = time.time_ns()

        # 1. Build prompts
        system_prompt = self.domain_router.get_system_prompt(domain)
        user_prompt = self.domain_router.build_prompt(
            problem, domain, difficulty, context, constraints
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 2. Try structured generation first
        schema = self.domain_router.get_schema()
        parsed = await self.backend.generate_structured(messages, schema, timeout=timeout)

        # 3. Parse response
        steps = []
        final_answer = ""
        code_artifact = None

        if parsed and "steps" in parsed:
            for i, step_data in enumerate(parsed["steps"]):
                steps.append(
                    ReasoningStep(
                        step_id=step_data.get("step_id", i),
                        reasoning=step_data.get("reasoning", ""),
                        evidence=step_data.get("evidence", ""),
                        confidence=float(step_data.get("confidence", 0.5)),
                        formal_proof_fragment=step_data.get("formal_proof_fragment"),
                    )
                )
            final_answer = parsed.get("final_answer", "")
        else:
            # Fallback: generate free-form and parse
            raw_text = await self.backend.generate(messages, timeout=timeout)
            steps, final_answer = self._parse_freeform(raw_text)

        # 4. Extract code artifact if applicable
        domain_val = domain.value if hasattr(domain, "value") else domain
        if domain_val == "code":
            code_artifact = self._extract_code_artifact(
                [s.reasoning for s in steps] + [final_answer]
            )

        # 5. Attempt proof generation
        steps_dicts = [
            {
                "step_id": s.step_id,
                "reasoning": s.reasoning,
                "evidence": s.evidence,
                "confidence": s.confidence,
                "formal_proof_fragment": s.formal_proof_fragment,
            }
            for s in steps
        ]
        proof_result = self.proof_generator.generate(steps_dicts, final_answer)

        # Update steps with any proof fragments found
        for i, step_dict in enumerate(steps_dicts):
            if i < len(steps) and step_dict.get("formal_proof_fragment"):
                steps[i].formal_proof_fragment = str(step_dict["formal_proof_fragment"])

        elapsed_ms = int((time.time_ns() - start_time) / 1_000_000)

        return ReasoningResult(
            steps=steps,
            final_answer=final_answer,
            proof_status=proof_result.status,
            proof_artifact=proof_result.artifact,
            code_artifact=code_artifact,
            time_taken_ms=elapsed_ms,
        )

    def _parse_freeform(self, text: str) -> tuple[list[ReasoningStep], str]:
        """Parse free-form LLM output into structured steps."""
        if not text:
            return [], ""

        lines = text.strip().split("\n")
        steps = []
        current_step: list[str] = []
        step_id = 0

        for line in lines:
            # Detect step boundaries (numbered lines, "Step X:", etc.)
            stripped = line.strip()
            if (
                stripped and stripped[0].isdigit() and "." in stripped[:4]
            ) or stripped.lower().startswith("step "):
                if current_step:
                    steps.append(
                        ReasoningStep(
                            step_id=step_id,
                            reasoning="\n".join(current_step),
                            confidence=0.5,
                        )
                    )
                    step_id += 1
                    current_step = []
            current_step.append(line)

        # Last step becomes final answer if no explicit answer section
        if current_step:
            text_block = "\n".join(current_step)
            if any(
                marker in text_block.lower()
                for marker in ["final answer", "therefore", "conclusion", "answer:"]
            ):
                final_answer = text_block
                if not steps:
                    steps.append(
                        ReasoningStep(
                            step_id=0,
                            reasoning=text_block,
                            confidence=0.5,
                        )
                    )
            else:
                steps.append(
                    ReasoningStep(
                        step_id=step_id,
                        reasoning=text_block,
                        confidence=0.5,
                    )
                )
                final_answer = text_block

        if not steps:
            steps = [ReasoningStep(step_id=0, reasoning=text, confidence=0.3)]
            final_answer = text

        return steps, final_answer if "final_answer" in dir() else steps[-1].reasoning

    def _extract_code_artifact(self, texts: list[str]) -> str | None:
        """Extract code blocks from reasoning text."""
        import base64
        import re

        for text in texts:
            code_blocks = re.findall(
                r"```(?:python|javascript|java|cpp|c\+\+|rust|go)?\n(.*?)```", text, re.DOTALL
            )
            if code_blocks:
                # Return the longest code block as the artifact
                longest = max(code_blocks, key=len)
                return base64.b64encode(longest.strip().encode()).decode()
        return None
