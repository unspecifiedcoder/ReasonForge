"""
ReasonForge - Abstract LLM Backend

All LLM backends must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> str:
        """Generate a text completion from the LLM.

        Args:
            messages: List of {role, content} message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout: Timeout in seconds.

        Returns:
            Generated text string.
        """
        ...

    @abstractmethod
    async def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """Generate a structured JSON response conforming to schema.

        Args:
            messages: List of {role, content} message dicts.
            schema: JSON schema for the expected response.
            timeout: Timeout in seconds.

        Returns:
            Parsed JSON dict matching the schema.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is available and responding."""
        ...
