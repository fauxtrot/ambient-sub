"""
M4 Token: Unified multimodal token format for ambient subconscious.

M4 tokens are the universal interchange format between layers:
- Reflexive layer produces M4 tokens with annotations
- Subconscious layer consumes M4 tokens
- Executive layer reads aggregated M4 token streams

Tokens preserve both structure (features, annotations) and dense
representations (embeddings) for flexibility.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import torch
import json


@dataclass
class M4Token:
    """
    Multimodal token format for the ambient listener system.

    Universal format for representing processed input across modalities.
    Providers produce these tokens, layers consume them.
    """

    # === Identity ===
    modality: str  # "audio", "vision_rgb", "vision_depth", "text", "system"
    source: str  # Provider identifier (e.g., "diart_speaker_id", "whisper_stt")
    timestamp: float  # Unix timestamp (seconds)
    duration_ms: float  # How long this token spans

    # === Dense Representation ===
    embedding: Optional[torch.Tensor] = None  # [latent_dim] - from modality encoder

    # === Structured Features ===
    # Preserved for interpretability and specialized processing
    # Modality-specific feature dictionaries
    features: Dict[str, float] = field(default_factory=dict)
    # Examples:
    #   Audio: {"rms": 0.23, "spectral_centroid": 1420.5, "zcr": 0.12, ...}
    #   Vision: {"motion": 0.1, "faces_detected": 1, "brightness": 0.7, ...}

    # === Reflexive Annotations ===
    # Added by reflexive providers, consumed by subconscious/executive
    # Each annotation has value, confidence, and optional metadata
    annotations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Examples:
    #   {"is_sound": {"value": True, "confidence": 0.94, "latency_ms": 12}}
    #   {"speaker_id": {"value": "Todd", "confidence": 0.87, "source": "diart"}}
    #   {"object_detected": {"class": "person", "bbox": [x, y, w, h], "confidence": 0.92}}

    # === Raw Data References ===
    # Optional references to raw data for training/replay
    raw_data: Optional[Dict[str, Any]] = field(default_factory=dict)
    # Examples:
    #   {"encodec_tokens": [234, 567, ...]}
    #   {"image_path": "data/sessions/session_123/frame_045.jpg"}
    #   {"depth_map_path": "data/sessions/session_123/depth_045.npy"}

    # === Metadata ===
    session_id: Optional[str] = None  # Session this token belongs to
    provider_version: Optional[str] = None  # Version of provider that generated this

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Note: torch.Tensor fields are not JSON-serializable,
        so they're converted to lists.
        """
        data = asdict(self)

        # Convert tensors to lists for JSON serialization
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()

        return data

    def to_json(self) -> str:
        """Convert to JSON string for storage."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'M4Token':
        """
        Create M4Token from dictionary.

        Reconstructs tensor fields from lists.
        """
        # Convert embedding list back to tensor
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = torch.tensor(data['embedding'])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'M4Token':
        """Create M4Token from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        """
        Convert to tensor format for model consumption.

        Used by subconscious layer to feed M4 tokens into neural networks.
        """
        # Convert features to tensor
        if self.features:
            feature_vec = torch.tensor(list(self.features.values()), dtype=torch.float32)
        else:
            feature_vec = torch.zeros(8, dtype=torch.float32)  # Default feature dim

        # Convert annotations to tensor
        annotation_vec = self._encode_annotations()

        # Ensure embedding exists
        if self.embedding is None:
            embedding = torch.zeros(512, dtype=torch.float32)  # Default embedding dim
        else:
            embedding = self.embedding

        return {
            "embedding": embedding,
            "features": feature_vec,
            "annotations": annotation_vec,
            "timestamp": torch.tensor([self.timestamp], dtype=torch.float32),
            "duration": torch.tensor([self.duration_ms], dtype=torch.float32),
        }

    def _encode_annotations(self) -> torch.Tensor:
        """
        Flatten annotations to fixed-size vector for neural network input.

        This is a simplified encoding. In production, you'd want a more
        sophisticated schema that handles variable annotations dynamically.
        """
        vec = []

        # Schema-dependent encoding
        # Expand as capabilities grow
        for key in ["is_sound", "is_speech", "speaker_detected", "object_detected"]:
            if key in self.annotations:
                ann = self.annotations[key]
                # Binary value + confidence
                value = float(ann.get("value", False)) if isinstance(ann.get("value"), bool) else float(ann.get("value", 0))
                confidence = float(ann.get("confidence", 0.0))
                vec.extend([value, confidence])
            else:
                vec.extend([0.0, 0.0])

        return torch.tensor(vec, dtype=torch.float32)

    def add_annotation(
        self,
        capability: str,
        value: Any,
        confidence: float,
        **metadata
    ):
        """
        Add an annotation from a provider.

        Args:
            capability: The capability this annotation represents (e.g., "speaker_id")
            value: The annotation value (e.g., "Todd", True, {"class": "person"})
            confidence: Confidence score (0.0 to 1.0)
            **metadata: Additional metadata (source, latency_ms, etc.)
        """
        self.annotations[capability] = {
            "value": value,
            "confidence": confidence,
            **metadata
        }

    def get_annotation(self, capability: str, default=None) -> Optional[Dict[str, Any]]:
        """Get annotation for a capability, or default if not present."""
        return self.annotations.get(capability, default)

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"M4Token(modality={self.modality}, source={self.source}, "
            f"timestamp={self.timestamp:.2f}s, "
            f"annotations={list(self.annotations.keys())})"
        )
