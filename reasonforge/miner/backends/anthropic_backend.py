"""
ReasonForge - Anthropic LLM Backend

Supports Claude models via the Anthropic API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import LLMBackend

logger = logging.getLogger("reasonforge.miner.anthropic")


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend for miner reasoning."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic>=0.20.0"
                )
        return self._client

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> str:
        client = self._get_client()

        # Extract system message if present
        system = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)

        try:
            kwargs = {
                "model": self.model,
                "messages": user_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system:
                kwargs["system"] = system

            response = await asyncio.wait_for(
                client.messages.create(**kwargs),
                timeout=timeout,
            )
            return response.content[0].text if response.content else ""
        except asyncio.TimeoutError:
            logger.warning("Anthropic request timed out after %ds", timeout)
            return ""
        except Exception as e:
            logger.error("Anthropic request failed: %s", e)
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
                + "\n\nRespond with ONLY valid JSON matching this schema:\n"
                + json.dumps(schema, indent=2)
            ),
        }

        text = await self.generate(messages_copy, temperature=0.3, timeout=timeout)
        try:
            # Try to extract JSON from the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse Anthropic response as JSON")
            return {}

    async def health_check(self) -> bool:
        try:
            result = await self.generate(
                [{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=10,
            )
            return bool(result)
        except Exception:
            return False
