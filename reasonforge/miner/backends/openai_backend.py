"""
ReasonForge - OpenAI/Compatible LLM Backend

Supports OpenAI API, Azure OpenAI, DeepSeek, local vLLM, and any
OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import LLMBackend

logger = logging.getLogger("reasonforge.miner.openai")


class OpenAIBackend(LLMBackend):
    """OpenAI API backend for miner reasoning."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai>=1.0.0"
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
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
            return response.choices[0].message.content or ""
        except asyncio.TimeoutError:
            logger.warning("OpenAI request timed out after %ds", timeout)
            return ""
        except Exception as e:
            logger.error("OpenAI request failed: %s", e)
            return ""

    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        timeout: int = 300,
    ) -> Dict[str, Any]:
        client = self._get_client()
        try:
            # Request JSON mode
            messages_with_json = messages.copy()
            messages_with_json[-1]["content"] += (
                "\n\nRespond with valid JSON matching this schema:\n"
                + json.dumps(schema, indent=2)
            )

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=messages_with_json,
                    temperature=0.3,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                ),
                timeout=timeout,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except asyncio.TimeoutError:
            logger.warning("OpenAI structured request timed out")
            return {}
        except json.JSONDecodeError:
            logger.warning("Failed to parse structured response as JSON")
            return {}
        except Exception as e:
            logger.error("OpenAI structured request failed: %s", e)
            return {}

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                ),
                timeout=10,
            )
            return bool(response.choices)
        except Exception:
            return False
