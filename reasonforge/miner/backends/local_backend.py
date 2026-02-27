"""
ReasonForge - Local LLM Backend

Supports HuggingFace transformers and vLLM for direct GPU inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from .base import LLMBackend

logger = logging.getLogger("reasonforge.miner.local")


class LocalBackend(LLMBackend):
    """Local transformers/vLLM backend for miner reasoning."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "auto",
        use_vllm: bool = False,
    ):
        self.model_name = model
        self.device = device
        self.use_vllm = use_vllm
        self._pipeline = None
        self._vllm_model = None

    def _load_model(self):
        """Lazy-load the model."""
        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self):
        if self._pipeline is not None:
            return
        try:
            import torch
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16,
            )
            logger.info("Loaded transformers model: %s", self.model_name)
        except ImportError:
            raise ImportError(
                "transformers package not installed. "
                "Install with: pip install transformers torch"
            )

    def _load_vllm(self):
        if self._vllm_model is not None:
            return
        try:
            from vllm import LLM

            self._vllm_model = LLM(model=self.model_name)
            logger.info("Loaded vLLM model: %s", self.model_name)
        except ImportError:
            raise ImportError(
                "vllm package not installed. "
                "Install with: pip install vllm"
            )

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt string."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> str:
        self._load_model()
        prompt = self._format_messages(messages)

        loop = asyncio.get_event_loop()

        try:
            if self.use_vllm:
                from vllm import SamplingParams

                params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: self._vllm_model.generate([prompt], params)
                    ),
                    timeout=timeout,
                )
                return result[0].outputs[0].text
            else:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._pipeline(
                            prompt,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            do_sample=True,
                            return_full_text=False,
                        ),
                    ),
                    timeout=timeout,
                )
                return result[0]["generated_text"]
        except asyncio.TimeoutError:
            logger.warning("Local generation timed out after %ds", timeout)
            return ""
        except Exception as e:
            logger.error("Local generation failed: %s", e)
            return ""

    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        timeout: int = 300,
    ) -> Dict[str, Any]:
        messages_copy = messages.copy()
        messages_copy[-1] = {
            **messages_copy[-1],
            "content": (
                messages_copy[-1]["content"]
                + "\n\nRespond with ONLY valid JSON:\n"
                + json.dumps(schema, indent=2)
            ),
        }
        text = await self.generate(messages_copy, temperature=0.3, timeout=timeout)
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return {}
        except json.JSONDecodeError:
            return {}

    async def health_check(self) -> bool:
        try:
            self._load_model()
            return True
        except Exception:
            return False
