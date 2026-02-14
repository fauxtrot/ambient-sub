"""
Agent system for ambient-subconscious.

This module provides the core agent infrastructure for building
an agent swarm coordinated through SpacetimeDB.
"""

from .base import Agent, ProviderAgent, EnrichmentAgent, TrainingAgent, AgentStatus
from .events import Event, EventType, EntryCreatedEvent, FrameCreatedEvent, EnrichmentAddedEvent, CorrectionReceivedEvent

__all__ = [
    'Agent',
    'ProviderAgent',
    'EnrichmentAgent',
    'TrainingAgent',
    'AgentStatus',
    'Event',
    'EventType',
    'EntryCreatedEvent',
    'FrameCreatedEvent',
    'EnrichmentAddedEvent',
    'CorrectionReceivedEvent',
]
