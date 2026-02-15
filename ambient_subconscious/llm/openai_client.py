"""
OpenAI-compatible LLM client for llama.cpp / vLLM / any OpenAI-API server.

Targets the /v1/chat/completions endpoint (e.g. llama-server on port 8080).
Drop-in replacement for OllamaClient in the executive agent.
"""

import logging
from typing import Dict, Any, List, Optional

import aiohttp

from .base import LLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """LLM client using the OpenAI chat completions API."""

    def __init__(
        self,
        host: str = "http://localhost:8080",
        model: str = "qwen2.5-7b",
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 60,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        logger.info(f"OpenAIClient initialized: host={host}, model={model}")

    async def complete(self, prompt: str, **kwargs) -> str:
        """Send a prompt via chat completions and return the response text."""
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        system_prompt = kwargs.get("system_prompt")

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self._chat(messages, model, temperature, max_tokens)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """Send a full message list and return the assistant reply."""
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        return await self._chat(messages, model, temperature, max_tokens)

    async def _chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.host}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    choices = result.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    return ""

        except aiohttp.ClientError as e:
            logger.error(f"OpenAI API connection error: {e}")
            return f"[LLM unavailable: {e}]"
        except Exception as e:
            logger.error(f"OpenAI API completion error: {e}")
            return f"[Error: {e}]"

    async def is_available(self) -> bool:
        """Check if the server is reachable via GET /v1/models."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.host}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"OpenAI API availability check failed: {e}")
            return False
