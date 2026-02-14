"""
Provider system for ambient subconscious.

This module implements the capability-based provider architecture for
multimodal data processing. Providers can be native M4 (speak the token
language) or adapted (external models with translation layers).
"""

from .base import Provider, ProviderAdapter
from .registry import CapabilityRegistry
from .router import CapabilityRouter
from .m4_token import M4Token

__all__ = [
    "Provider",
    "ProviderAdapter",
    "CapabilityRegistry",
    "CapabilityRouter",
    "M4Token",
]
