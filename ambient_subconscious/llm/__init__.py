"""
LLM client interfaces for executive agent.

Provides base interface and implementations for:
- Personaplex (conversational)
- Ollama (reasoning)
"""

from .base import LLMClient
from .personaplex_client import PersonaplexClient
from .ollama_client import OllamaClient

__all__ = ['LLMClient', 'PersonaplexClient', 'OllamaClient']
