"""
ReasonForge - Agent LLM Backend

Supports LangGraph/LangChain multi-agent reasoning pipelines.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import LLMBackend

logger = logging.getLogger("reasonforge.miner.agent")


class AgentBackend(LLMBackend):
    """LangGraph/LangChain agent backend for multi-step reasoning."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        max_iterations: int = 10,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_iterations = max_iterations
        self._agent: Any = None

    def _build_agent(self):
        """Build a LangGraph reasoning agent."""
        if self._agent is not None:
            return

        try:
            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent

            llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=0.7,
            )

            # Create a basic ReAct agent with reasoning tools
            self._agent = create_react_agent(llm, tools=[])
            logger.info("Built LangGraph agent with model: %s", self.model)

        except ImportError:
            raise ImportError(
                "langchain/langgraph not installed. "
                "Install with: pip install langchain-openai langgraph"
            )

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> str:
        self._build_agent()

        try:
            # Convert messages to LangGraph format
            from langchain_core.messages import HumanMessage, SystemMessage

            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                else:
                    lc_messages.append(HumanMessage(content=msg["content"]))

            result = await asyncio.wait_for(
                self._agent.ainvoke(
                    {"messages": lc_messages},
                ),
                timeout=timeout,
            )

            # Extract final message
            if result and "messages" in result:
                return result["messages"][-1].content
            return ""

        except asyncio.TimeoutError:
            logger.warning("Agent execution timed out after %ds", timeout)
            return ""
        except Exception as e:
            logger.error("Agent execution failed: %s", e)
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
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return {}
        except json.JSONDecodeError:
            return {}

    async def health_check(self) -> bool:
        try:
            self._build_agent()
            return True
        except Exception:
            return False
