"""
Ollama LLM client for deep reasoning.

Deployment: Local Ollama installation
Model: User-configurable (llama3, mistral, etc.)
Role: Strategic thinking, complex analysis
"""

import logging
from typing import Dict, Any

import aiohttp

from .base import LLMClient

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """
    Ollama reasoning LLM client.

    Handles deep analysis and strategic thinking when triggered.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.7,
        timeout: int = 60
    ):
        """
        Initialize Ollama client.

        Args:
            host: Ollama server URL
            model: Model name (llama3, mistral, codellama, etc.)
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

        logger.info(f"OllamaClient initialized: host={host}, model={model}")

    async def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate reasoning response.

        Args:
            prompt: Text prompt
            **kwargs: Optional overrides (temperature, model)

        Returns:
            Generated analysis/reasoning
        """
        model = kwargs.get('model', self.model)
        temperature = kwargs.get('temperature', self.temperature)

        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result.get("response", "")

        except aiohttp.ClientError as e:
            logger.error(f"Ollama connection error: {e}")
            return f"[Ollama unavailable: {e}]"
        except Exception as e:
            logger.error(f"Ollama completion error: {e}")
            return f"[Error: {e}]"

    async def analyze(self, context: Dict[str, Any], conversation: str) -> str:
        """
        Deep reasoning/analysis mode.

        Args:
            context: Executive agent's side context
            conversation: Recent conversational output

        Returns:
            Strategic analysis
        """
        prompt = f"""You are a strategic reasoning agent. Provide deep analysis of the current situation.

Context:
{self._format_context(context)}

Recent Conversation:
{conversation}

Provide:
1. Strategic analysis of what's happening
2. Patterns or insights
3. Recommended focus or actions

Keep your analysis concise but insightful.
"""
        return await self.complete(prompt)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        lines = []
        for key, value in context.items():
            if value:  # Only include non-empty values
                lines.append(f"  {key}: {value}")
        return "\n".join(lines) if lines else "  (No context available)"

    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False
