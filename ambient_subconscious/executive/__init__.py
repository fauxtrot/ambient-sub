"""
Executive layer for high-level reasoning and decision making.

Dual-LLM architecture:
- Personaplex: Conversational, natural dialogue
- Ollama: Deep reasoning, strategic analysis

Stage integration:
- StageClient: HTTP client for VTuber avatar control (emotes, speech)
- TTSEngine: Pocket TTS wrapper for voice synthesis
"""

from .executive_agent import ExecutiveAgent
from .stage_client import StageClient
from .tts import TTSEngine

__all__ = ['ExecutiveAgent', 'StageClient', 'TTSEngine']
