"""
LLM client interfaces for executive agent.

Provides base interface and implementations for:
- Personaplex (conversational)
- Ollama (reasoning)
- OpenAI-compatible (llama.cpp, vLLM, etc.)
"""

from .base import LLMClient
from .personaplex_client import PersonaplexClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient

__all__ = ['LLMClient', 'PersonaplexClient', 'OllamaClient', 'OpenAIClient']
