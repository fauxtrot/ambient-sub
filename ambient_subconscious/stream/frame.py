"""
Frame dataclass - atomic unit of the continuous audio stream.

Each frame captures the state at a specific timepoint:
- Audio position (timestamp + sample position)
- Speaker prediction (from diarization)
- Visual context (from YOLO, arrives async)
- Text hypothesis (from Whisper, arrives async)
- Confidence and weight for decay
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class Frame:
    """
    Represents a single timepoint in the continuous audio stream.

    Frames are the atomic unit of storage. Audio provides the timebase,
    and visual/text data anchors to audio events.
    """

    # Timebase (audio is the foundation)
    timestamp: float  # seconds from stream start
    sample_position: int  # sample index in continuous audio stream

    # Speaker prediction (from diart real-time diarization)
    speaker_prediction: Optional[str] = None  # "speaker_0", "speaker_1", etc.

    # Visual context (from YOLO, triggered on audio events)
    visual_context: Optional[Dict[str, Any]] = None  # {"detections": [...], "embedding": [...]}

    # Text hypothesis (from Whisper, arrives async)
    text_hypothesis: Optional[str] = None

    # Confidence and weight
    confidence: float = 0.0  # prediction confidence (0.0-1.0)
    weight: float = 1.0  # decays over time for ephemeral predictions

    # Metadata
    frame_id: Optional[str] = None  # Auto-generated: "{session_id}_{timestamp}"
    session_id: Optional[str] = None  # Which session this belongs to
    event_type: Optional[str] = None  # "speaker_change", "speech_start", "update", etc.
    created_at: Optional[str] = None  # ISO timestamp when frame was created

    # Enrichment tracking (for future SpacetimeDB migration)
    enriched_at: Optional[str] = None  # Set when enriched with visual/text data

    def __post_init__(self):
        """Auto-generate missing fields"""
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

        if self.frame_id is None and self.session_id:
            self.frame_id = f"{self.session_id}_{self.timestamp:.3f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string (for JSONL storage)"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Frame":
        """Create Frame from dictionary"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Frame":
        """Create Frame from JSON string"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_diarization_event(
        cls,
        event,  # DiarizationEvent
        session_id: str,
        stream_start_time: float = 0.0
    ) -> "Frame":
        """
        Create Frame from AudioListener DiarizationEvent.

        This is the primary integration point with the existing listener.

        Args:
            event: DiarizationEvent from AudioListener
            session_id: Current session ID
            stream_start_time: Stream start timestamp (for offset calculation)

        Returns:
            Frame object
        """
        return cls(
            timestamp=event.timestamp,
            sample_position=event.sample_position,
            speaker_prediction=event.speaker,
            confidence=event.confidence,
            event_type=event.event_type,
            session_id=session_id,
        )

    def add_visual_context(self, detections: list, embedding: Optional[list] = None):
        """
        Add visual context from YOLO (called async after frame creation).

        Args:
            detections: List of YOLO detections
            embedding: Optional visual embedding vector
        """
        self.visual_context = {
            "detections": detections,
            "embedding": embedding,
            "captured_at": datetime.utcnow().isoformat()
        }
        self.enriched_at = datetime.utcnow().isoformat()

    def add_text_hypothesis(self, text: str, confidence: float = 0.0):
        """
        Add text from Whisper (called async after frame creation).

        Args:
            text: Transcribed text
            confidence: Transcription confidence
        """
        self.text_hypothesis = text
        self.confidence = max(self.confidence, confidence)  # Take max confidence
        self.enriched_at = datetime.utcnow().isoformat()
