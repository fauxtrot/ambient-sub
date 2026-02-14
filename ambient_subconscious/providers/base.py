"""
Base interfaces for providers and adapters.

Providers implement capabilities (modality, capability) pairs.
They can be native M4 (speak the token language) or adapted (external models).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
from .m4_token import M4Token


class Provider(ABC):
    """
    Base interface for all capability providers.

    Providers register with (modality, capability) pairs and process
    input data to produce M4 tokens with annotations.

    Examples:
        - (audio, speaker_id): Identifies speaker from audio
        - (audio, transcription): Transcribes speech to text
        - (vision_rgb, object_detection): Detects objects in images
        - (vision_depth, depth_map): Generates depth maps
    """

    def __init__(self, provider_id: str, version: str = "1.0"):
        """
        Initialize provider.

        Args:
            provider_id: Unique identifier for this provider (e.g., "diart_speaker_id")
            version: Version string for tracking provider evolution
        """
        self.provider_id = provider_id
        self.version = version

    @abstractmethod
    def get_capabilities(self) -> List[Tuple[str, str]]:
        """
        Returns the (modality, capability) pairs this provider supports.

        A provider can support multiple capabilities.

        Example:
            return [("audio", "speaker_id"), ("audio", "is_speech")]
        """
        pass

    @abstractmethod
    def is_native_m4(self) -> bool:
        """
        Returns True if provider speaks M4 token language natively.

        Native M4 providers:
            - Built/trained by us
            - Directly produce M4 tokens
            - Understand M4 token inputs

        Adapted providers:
            - External models (Whisper, YOLO, etc.)
            - Need ProviderAdapter wrapper
            - Use external input/output formats
        """
        pass

    @abstractmethod
    def process(self, input_data: Any) -> M4Token:
        """
        Process input data and produce M4 token with annotations.

        Args:
            input_data: Raw data (audio, image, etc.) or M4 token

        Returns:
            M4Token with this provider's annotations added

        The input_data format depends on the provider:
            - Native M4: Can accept M4 tokens or raw data
            - Adapted: Adapter translates before calling this
        """
        pass

    def update_from_correction(self, token: M4Token, correction: dict):
        """
        Optional: Update provider based on executive correction.

        This enables online learning / feedback loops.
        Providers that support this should override.

        Args:
            token: Original M4Token that was corrected
            correction: Correction dictionary from executive
                Example: {"speaker_id": "YouTuber", "confidence": 1.0}
        """
        pass  # Default: no-op

    def get_metadata(self) -> dict:
        """
        Return provider metadata for tracking/debugging.

        Returns:
            Dict with provider info (id, version, capabilities, etc.)
        """
        return {
            "provider_id": self.provider_id,
            "version": self.version,
            "capabilities": self.get_capabilities(),
            "is_native_m4": self.is_native_m4(),
        }


class ProviderAdapter(ABC):
    """
    Adapter for wrapping external providers that don't speak M4.

    Adapters translate between external provider formats and M4 tokens.
    They also capture domain knowledge (e.g., "this is what Todd's voice looks like").

    The adapter pattern allows us to:
        1. Use mature external models (Whisper, YOLO) without retraining
        2. Translate their outputs to M4 tokens with rich annotations
        3. Capture domain-specific features as reflexive shortcuts
        4. Maintain consistent M4 interface across all providers
    """

    def __init__(self, provider_id: str, external_provider: Any, version: str = "1.0"):
        """
        Initialize adapter.

        Args:
            provider_id: Unique identifier (e.g., "whisper_stt")
            external_provider: The external model/API being wrapped
            version: Adapter version string
        """
        self.provider_id = provider_id
        self.external_provider = external_provider
        self.version = version

    @abstractmethod
    def get_capabilities(self) -> List[Tuple[str, str]]:
        """
        Returns the (modality, capability) pairs this adapter provides.

        Example:
            return [("audio", "transcription")]
        """
        pass

    def is_native_m4(self) -> bool:
        """Adapters are not native M4 by definition."""
        return False

    @abstractmethod
    def translate_input(self, input_data: Any) -> Any:
        """
        Translate M4 token or raw data to external provider's format.

        Args:
            input_data: M4Token or raw data (audio array, image, etc.)

        Returns:
            Data in format expected by external_provider

        Example (Whisper adapter):
            - Input: M4Token with raw audio data
            - Extract: audio array from token.raw_data
            - Return: numpy array for Whisper
        """
        pass

    @abstractmethod
    def translate_output(
        self,
        provider_output: Any,
        original_input: Any,
        timestamp: float,
        duration_ms: float
    ) -> M4Token:
        """
        Translate external provider's output to M4 token.

        This is where domain knowledge is captured as features/annotations.

        Args:
            provider_output: Output from external_provider
            original_input: Original input (for extracting metadata)
            timestamp: Timestamp for the M4 token
            duration_ms: Duration for the M4 token

        Returns:
            M4Token with annotations from provider output

        Example (Whisper adapter):
            - provider_output: {"text": "hello", "language": "en"}
            - Extract audio characteristics (pitch, cadence, etc.)
            - Create M4Token with:
                - modality: "text" (translated from audio)
                - annotations: {"transcription": ..., "language": ...}
                - features: {"voice_pitch_hz": ..., "cadence_words_per_min": ...}
        """
        pass

    def process(
        self,
        input_data: Any,
        timestamp: Optional[float] = None,
        duration_ms: Optional[float] = None
    ) -> M4Token:
        """
        Process input through external provider and translate to M4 token.

        This is the main entry point for adapted providers.

        Args:
            input_data: Raw data or M4 token
            timestamp: Optional timestamp (extracted from M4Token if not provided)
            duration_ms: Optional duration (extracted from M4Token if not provided)

        Returns:
            M4Token with provider annotations
        """
        # Extract timestamp/duration from M4Token if input is a token
        if isinstance(input_data, M4Token):
            if timestamp is None:
                timestamp = input_data.timestamp
            if duration_ms is None:
                duration_ms = input_data.duration_ms

        # Default timestamp/duration if not provided
        if timestamp is None:
            import time
            timestamp = time.time()
        if duration_ms is None:
            duration_ms = 50.0  # Default 50ms frame

        # Translate input to external format
        external_input = self.translate_input(input_data)

        # Call external provider
        external_output = self.external_provider(external_input)

        # Translate output to M4 token
        m4_token = self.translate_output(
            provider_output=external_output,
            original_input=input_data,
            timestamp=timestamp,
            duration_ms=duration_ms
        )

        # Set provider metadata
        m4_token.source = self.provider_id
        m4_token.provider_version = self.version

        return m4_token

    def get_metadata(self) -> dict:
        """Return adapter metadata."""
        return {
            "provider_id": self.provider_id,
            "version": self.version,
            "capabilities": self.get_capabilities(),
            "is_native_m4": False,
            "external_provider": str(type(self.external_provider).__name__),
        }
