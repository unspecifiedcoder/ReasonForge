"""Translation engine that orchestrates NL-to-Formal translation."""

from __future__ import annotations

import logging
from typing import List, Optional

from .prompts import CODE_SYSTEM_PROMPT, LOGIC_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT
from .types import StepTranslation, TranslationRequest, TranslationResult

logger = logging.getLogger("reasonforge.translation.engine")

_DOMAIN_PROMPTS = {
    "mathematics": MATH_SYSTEM_PROMPT,
    "code": CODE_SYSTEM_PROMPT,
    "logic": LOGIC_SYSTEM_PROMPT,
}


class TranslationEngine:
    """Translates natural-language reasoning chains into formal representations.

    The engine selects the appropriate system prompt based on the domain and
    delegates to an LLM backend for the actual translation.  When no backend
    is connected the engine returns deterministic placeholder translations so
    the rest of the pipeline can be developed and tested independently.
    """

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o",
        domains: Optional[List[str]] = None,
    ) -> None:
        self.backend_name = backend
        self.model = model
        self.domains: List[str] = domains or ["mathematics", "code", "logic"]
        self._backend = None  # lazy-loaded when a real LLM is wired up

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def translate_step(
        self,
        step: dict,
        domain: str,
        proof_level: str,
        previous_steps: List[StepTranslation],
        original_query: str,
    ) -> StepTranslation:
        """Translate a single reasoning step into its formal representation.

        Parameters
        ----------
        step:
            A dict with at least ``"step_id"`` (int) and ``"content"`` (str).
        domain:
            One of the supported domains (mathematics, code, logic).
        proof_level:
            The requested proof rigour level (e.g. "standard", "detailed").
        previous_steps:
            Already-translated steps for building context.
        original_query:
            The user's original question / problem statement.

        Returns
        -------
        StepTranslation
            The translated step.  Until a real LLM backend is connected this
            returns a placeholder so the pipeline remains functional.
        """
        system_prompt = self._get_system_prompt(domain)
        context = self._build_context(previous_steps, original_query)

        step_id: int = step.get("step_id", 0)
        content: str = step.get("content", "")

        user_message = (
            f"Original query: {original_query}\n\n"
            f"Context from previous steps:\n{context}\n\n"
            f"Current step (step {step_id}):\n{content}\n\n"
            f"Proof level: {proof_level}\n"
            f"Domain: {domain}\n\n"
            f"Translate the current step into the appropriate formal representation."
        )

        logger.debug(
            "translate_step  step_id=%s  domain=%s  system_prompt_len=%d  "
            "user_message_len=%d",
            step_id,
            domain,
            len(system_prompt),
            len(user_message),
        )

        # ----- placeholder until a real backend is connected -----
        if self._backend is None:
            logger.info(
                "No LLM backend connected — returning placeholder for step %s",
                step_id,
            )
            return StepTranslation(
                step_id=step_id,
                original_content=content,
                formal_representation="-- placeholder: LLM backend not connected",
                dependencies=[s.step_id for s in previous_steps],
                translation_confidence=0.0,
                compilation_check=False,
                notes="Placeholder — wire up an LLM backend to get real translations.",
            )

        # Future: call self._backend with system_prompt + user_message
        raise NotImplementedError("LLM backend dispatch is not yet implemented")

    async def translate_chain(
        self,
        request: TranslationRequest,
    ) -> TranslationResult:
        """Translate every step in a reasoning chain.

        Parameters
        ----------
        request:
            A ``TranslationRequest`` containing the full reasoning chain.

        Returns
        -------
        TranslationResult
            Aggregated translations for the entire chain.
        """
        logger.info(
            "translate_chain  task_id=%s  steps=%d  domain=%s",
            request.task_id,
            len(request.reasoning_chain),
            request.domain,
        )

        translated: List[StepTranslation] = []

        for step in request.reasoning_chain:
            result = await self.translate_step(
                step=step,
                domain=request.domain,
                proof_level=request.proof_level,
                previous_steps=translated,
                original_query=request.original_query,
            )
            translated.append(result)

        # Determine an overall compilation status
        if not translated:
            compilation_status = "EMPTY"
        elif all(t.compilation_check for t in translated):
            compilation_status = "PASSED"
        else:
            compilation_status = "FAILED"

        full_proof = (
            "\n\n".join(t.formal_representation for t in translated)
            if translated
            else None
        )

        return TranslationResult(
            task_id=request.task_id,
            translations=translated,
            compilation_status=compilation_status,
            full_proof=full_proof,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        previous_steps: List[StepTranslation],
        original_query: str,
    ) -> str:
        """Build a context string from previously translated steps."""
        if not previous_steps:
            return "(no previous steps)"

        parts: List[str] = []
        for step in previous_steps:
            parts.append(
                f"Step {step.step_id} "
                f"(confidence={step.translation_confidence:.2f}, "
                f"compiled={step.compilation_check}):\n"
                f"{step.formal_representation}"
            )
        return "\n\n".join(parts)

    def _get_system_prompt(self, domain: str) -> str:
        """Return the system prompt for the given domain."""
        prompt = _DOMAIN_PROMPTS.get(domain)
        if prompt is None:
            logger.warning(
                "Unknown domain %r — falling back to mathematics prompt", domain
            )
            return MATH_SYSTEM_PROMPT
        return prompt
