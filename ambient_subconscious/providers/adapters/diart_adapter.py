"""
Diart adapter for speaker diarization.

Wraps the existing diart/pyannote speaker diarization pipeline
as a provider in the M4 token system.
"""

import os
import numpy as np
import torch
from typing import Any, List, Tuple, Optional
from diart import SpeakerDiarization
from ..base import ProviderAdapter
from ..m4_token import M4Token

# Configure model cache
os.environ['HF_HOME'] = os.path.abspath('.models/huggingface')
os.environ['TORCH_HOME'] = os.path.abspath('.models/torch')


class DiartAdapter(ProviderAdapter):
    """
    Adapter for diart speaker diarization.

    Provides capabilities:
        - (audio, speaker_id): Identifies active speaker(s)
        - (audio, is_speech): Detects if speech is present

    This wraps the existing audio_source.py AudioListener functionality
    into the provider system.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        version: str = "1.0"
    ):
        """
        Initialize Diart adapter.

        Args:
            device: 'cuda' or 'cpu'. Auto-detected if None.
            sample_rate: Expected audio sample rate (Hz)
            version: Adapter version
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.sample_rate = sample_rate

        # Initialize diart pipeline
        print(f"Initializing Diart speaker diarization on {device}...")
        pipeline = SpeakerDiarization()

        super().__init__(
            provider_id="diart_speaker_diarization",
            external_provider=pipeline,
            version=version
        )

        print("Diart adapter ready")

    def get_capabilities(self) -> List[Tuple[str, str]]:
        """Diart provides speaker ID and speech detection."""
        return [
            ("audio", "speaker_id"),
            ("audio", "is_speech"),
        ]

    def translate_input(self, input_data: Any) -> Any:
        """
        Translate input to format expected by diart.

        Args:
            input_data: Can be:
                - numpy array of audio samples
                - M4Token with raw_data["audio"]
                - torch.Tensor of audio

        Returns:
            numpy array for diart
        """
        # If M4Token, extract audio
        if isinstance(input_data, M4Token):
            if "audio" in input_data.raw_data:
                audio = input_data.raw_data["audio"]
            else:
                raise ValueError("M4Token missing 'audio' in raw_data")
        else:
            audio = input_data

        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        elif not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        return audio

    def translate_output(
        self,
        provider_output: Any,
        original_input: Any,
        timestamp: float,
        duration_ms: float
    ) -> M4Token:
        """
        Translate diart output to M4 token.

        Args:
            provider_output: Annotation object from diart
            original_input: Original audio or M4Token
            timestamp: Timestamp for token
            duration_ms: Duration for token

        Returns:
            M4Token with speaker_id and is_speech annotations
        """
        # Extract speaker information from annotation
        annotation = provider_output

        # Get active speakers at this timestamp
        active_speakers = set()
        for segment, _, label in annotation.itertracks(yield_label=True):
            if segment.start <= timestamp <= segment.end:
                active_speakers.add(label)

        # Determine primary speaker (first one if multiple)
        speaker_id = list(active_speakers)[0] if active_speakers else None
        is_speech = len(active_speakers) > 0

        # Create M4Token
        token = M4Token(
            modality="audio",
            source=self.provider_id,
            timestamp=timestamp,
            duration_ms=duration_ms,
            embedding=None,  # Diart doesn't provide embeddings directly
            features={},  # Could extract from audio if needed
            raw_data={},
            provider_version=self.version,
        )

        # Add speaker_id annotation
        if speaker_id is not None:
            token.add_annotation(
                capability="speaker_id",
                value=speaker_id,
                confidence=1.0,  # Diart doesn't provide confidence scores directly
                num_speakers=len(active_speakers),
                all_speakers=list(active_speakers),
            )

        # Add is_speech annotation
        token.add_annotation(
            capability="is_speech",
            value=is_speech,
            confidence=1.0 if is_speech else 0.95,
        )

        return token


class DiartStreamAdapter:
    """
    Streaming adapter for diart.

    This wraps the existing AudioListener functionality for streaming audio.
    For now, this is a placeholder for future streaming integration.

    The current DiartAdapter works on audio chunks. For streaming integration,
    we'd need to:
        1. Use diart's StreamingInference
        2. Emit M4Tokens as diarization events occur
        3. Maintain speaker state across chunks
    """

    def __init__(self, device: Optional[str] = None, audio_device: Optional[int] = None):
        """
        Initialize streaming adapter.

        Args:
            device: 'cuda' or 'cpu'
            audio_device: Audio input device index
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.audio_device = audio_device
        self.sample_rate = 16000

        # TODO: Initialize streaming pipeline
        # This would use diart.inference.StreamingInference
        # and diart.sources.MicrophoneAudioSource

        print("DiartStreamAdapter initialized (streaming not yet implemented)")

    def start_stream(self):
        """Start streaming audio and emitting M4Tokens."""
        # TODO: Implement streaming
        raise NotImplementedError("Streaming support coming soon")

    def stop_stream(self):
        """Stop streaming."""
        # TODO: Implement
        pass
