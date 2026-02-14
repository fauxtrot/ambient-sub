"""
Event system for agent communication.

Events are the primary mechanism for agents to communicate with each other
and react to changes in SpacetimeDB.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """Types of events in the system."""
    # Data events (from SpacetimeDB)
    ENTRY_CREATED = "entry_created"
    ENTRY_UPDATED = "entry_updated"
    ENTRY_DELETED = "entry_deleted"
    FRAME_CREATED = "frame_created"
    FRAME_UPDATED = "frame_updated"
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"

    # Enrichment events
    ENRICHMENT_ADDED = "enrichment_added"
    SPEAKER_IDENTIFIED = "speaker_identified"
    SENTIMENT_DETECTED = "sentiment_detected"
    INTENT_DETECTED = "intent_detected"
    OBJECTS_DETECTED = "objects_detected"

    # User interaction events
    CORRECTION_RECEIVED = "correction_received"
    UI_INTERACTION = "ui_interaction"
    VOICE_COMMAND = "voice_command"

    # Agent control events
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"

    # Prediction events
    PREDICTION_MADE = "prediction_made"
    CONTEXT_UPDATED = "context_updated"
    CLCT_HEARTBEAT = "clct_heartbeat"

    # Training events
    MODEL_RETRAINED = "model_retrained"
    AB_TEST_STARTED = "ab_test_started"
    AB_TEST_COMPLETED = "ab_test_completed"


class EventPriority(Enum):
    """Event processing priority."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """
    Base event class.

    All events in the system inherit from this class.
    """
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    source_agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'source_agent_id': self.source_agent_id,
            'data': self.data,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            event_type=EventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            priority=EventPriority(data['priority']),
            source_agent_id=data.get('source_agent_id'),
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
        )


@dataclass
class EntryCreatedEvent(Event):
    """Event fired when a new TranscriptEntry is created."""

    def __init__(
        self,
        entry_id: str,
        session_id: str,
        timestamp: float,
        transcript: str,
        confidence: float,
        audio_clip_path: Optional[str] = None,
        source_agent_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            event_type=EventType.ENTRY_CREATED,
            source_agent_id=source_agent_id,
            data={
                'entry_id': entry_id,
                'session_id': session_id,
                'timestamp': timestamp,
                'transcript': transcript,
                'confidence': confidence,
                'audio_clip_path': audio_clip_path,
                **kwargs
            }
        )

    @property
    def entry_id(self) -> str:
        return self.data['entry_id']

    @property
    def session_id(self) -> str:
        return self.data['session_id']

    @property
    def transcript(self) -> str:
        return self.data['transcript']

    @property
    def confidence(self) -> float:
        return self.data['confidence']


@dataclass
class FrameCreatedEvent(Event):
    """Event fired when a new Frame is created (vision data)."""

    def __init__(
        self,
        frame_id: str,
        session_id: str,
        timestamp: float,
        frame_type: str,  # "webcam", "screen"
        image_path: str,
        detections: Optional[list] = None,
        source_agent_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            event_type=EventType.FRAME_CREATED,
            source_agent_id=source_agent_id,
            data={
                'frame_id': frame_id,
                'session_id': session_id,
                'timestamp': timestamp,
                'frame_type': frame_type,
                'image_path': image_path,
                'detections': detections or [],
                **kwargs
            }
        )

    @property
    def frame_id(self) -> str:
        return self.data['frame_id']

    @property
    def session_id(self) -> str:
        return self.data['session_id']

    @property
    def frame_type(self) -> str:
        return self.data['frame_type']

    @property
    def image_path(self) -> str:
        return self.data['image_path']

    @property
    def detections(self) -> list:
        return self.data['detections']


@dataclass
class EnrichmentAddedEvent(Event):
    """Event fired when enrichment is added to an entry."""

    def __init__(
        self,
        entry_id: str,
        enrichment_type: str,  # "sentiment", "intent", "speaker", etc.
        enrichment_value: Any,
        confidence: Optional[float] = None,
        source_agent_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            event_type=EventType.ENRICHMENT_ADDED,
            source_agent_id=source_agent_id,
            data={
                'entry_id': entry_id,
                'enrichment_type': enrichment_type,
                'enrichment_value': enrichment_value,
                'confidence': confidence,
                **kwargs
            }
        )

    @property
    def entry_id(self) -> str:
        return self.data['entry_id']

    @property
    def enrichment_type(self) -> str:
        return self.data['enrichment_type']

    @property
    def enrichment_value(self) -> Any:
        return self.data['enrichment_value']


@dataclass
class CorrectionReceivedEvent(Event):
    """Event fired when user corrects a prediction."""

    def __init__(
        self,
        entry_id: str,
        correction_type: str,  # "transcript", "speaker", "sentiment", "object", etc.
        original_value: Any,
        corrected_value: Any,
        source_agent_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            event_type=EventType.CORRECTION_RECEIVED,
            priority=EventPriority.HIGH,  # Corrections are important for training
            source_agent_id=source_agent_id,
            data={
                'entry_id': entry_id,
                'correction_type': correction_type,
                'original_value': original_value,
                'corrected_value': corrected_value,
                **kwargs
            }
        )

    @property
    def entry_id(self) -> str:
        return self.data['entry_id']

    @property
    def correction_type(self) -> str:
        return self.data['correction_type']

    @property
    def original_value(self) -> Any:
        return self.data['original_value']

    @property
    def corrected_value(self) -> Any:
        return self.data['corrected_value']


@dataclass
class UIInteractionEvent(Event):
    """Event fired when user interacts with UI."""

    def __init__(
        self,
        interaction_type: str,  # "edit", "click", "hover", etc.
        target: str,  # What was interacted with
        details: Optional[Dict[str, Any]] = None,
        source_agent_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            event_type=EventType.UI_INTERACTION,
            source_agent_id=source_agent_id,
            data={
                'interaction_type': interaction_type,
                'target': target,
                'details': details or {},
                **kwargs
            }
        )

    @property
    def interaction_type(self) -> str:
        return self.data['interaction_type']

    @property
    def target(self) -> str:
        return self.data['target']


@dataclass
class CLCTHeartbeatEvent(Event):
    """Event fired by CLCT heartbeat with current context."""

    def __init__(
        self,
        latent_embedding: Any,  # Current latent representation
        current_context: Dict[str, Any],  # Current speaker, objects, etc.
        working_memory_size: int,
        predictions: Dict[str, Any],
        source_agent_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            event_type=EventType.CLCT_HEARTBEAT,
            source_agent_id=source_agent_id,
            data={
                'latent_embedding': latent_embedding,
                'current_context': current_context,
                'working_memory_size': working_memory_size,
                'predictions': predictions,
                **kwargs
            }
        )

    @property
    def current_context(self) -> Dict[str, Any]:
        return self.data['current_context']

    @property
    def predictions(self) -> Dict[str, Any]:
        return self.data['predictions']


@dataclass
class PredictionMadeEvent(Event):
    """Event fired when a model makes a prediction."""

    def __init__(
        self,
        model_name: str,
        model_version: str,
        prediction_type: str,  # "speaker_id", "sentiment", "objects", etc.
        prediction_value: Any,
        confidence: float,
        input_data: Optional[Dict[str, Any]] = None,
        source_agent_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            event_type=EventType.PREDICTION_MADE,
            source_agent_id=source_agent_id,
            data={
                'model_name': model_name,
                'model_version': model_version,
                'prediction_type': prediction_type,
                'prediction_value': prediction_value,
                'confidence': confidence,
                'input_data': input_data,
                **kwargs
            }
        )

    @property
    def model_name(self) -> str:
        return self.data['model_name']

    @property
    def prediction_type(self) -> str:
        return self.data['prediction_type']

    @property
    def prediction_value(self) -> Any:
        return self.data['prediction_value']

    @property
    def confidence(self) -> float:
        return self.data['confidence']


@dataclass
class ModelRetrainedEvent(Event):
    """Event fired when a model is retrained."""

    def __init__(
        self,
        model_name: str,
        old_version: str,
        new_version: str,
        training_samples: int,
        performance_metrics: Dict[str, float],
        ab_line: Optional[str] = None,  # "a" or "b"
        source_agent_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            event_type=EventType.MODEL_RETRAINED,
            priority=EventPriority.HIGH,
            source_agent_id=source_agent_id,
            data={
                'model_name': model_name,
                'old_version': old_version,
                'new_version': new_version,
                'training_samples': training_samples,
                'performance_metrics': performance_metrics,
                'ab_line': ab_line,
                **kwargs
            }
        )

    @property
    def model_name(self) -> str:
        return self.data['model_name']

    @property
    def new_version(self) -> str:
        return self.data['new_version']

    @property
    def performance_metrics(self) -> Dict[str, float]:
        return self.data['performance_metrics']
