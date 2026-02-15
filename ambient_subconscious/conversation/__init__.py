"""Conversation engine — ZMQ audio → Whisper → LLM → Skills → Game Client."""

from .engine import ConversationEngine
from .skills import Skill, SkillRegistry

__all__ = ["ConversationEngine", "Skill", "SkillRegistry"]
