"""
Base LLM client interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class LLMClient(ABC):
    """Base interface for LLM clients."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """
        Send prompt and get completion.

        Args:
            prompt: Text prompt to send to LLM
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if LLM is available/healthy.

        Returns:
            True if LLM is ready to use
        """
        pass

    async def cleanup(self) -> None:
        """
        Cleanup resources (optional).

        Override if client needs cleanup (close connections, etc.)
        """
        pass
