"""
Utterance buffering for audio capture with rich metadata tracking.

Buffers audio chunks and tracks metadata (energy profiles, speaker events)
during an active utterance.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Represents a speaker segment within an utterance."""
    speaker: str
    start_ms: int
    end_ms: int


class UtteranceBuffer:
    """
    Buffers audio and metadata during an active utterance.

    Tracks:
    - Audio chunks
    - Energy profile (RMS over time)
    - Speaker diarization events
    - Timing information
    """

    def __init__(
        self,
        start_sample: int,
        device_label: str,
        sample_rate: int = 16000
    ):
        """
        Initialize utterance buffer.

        Args:
            start_sample: Starting sample position in audio stream
            device_label: Label for audio device (e.g., "desk_mic")
            sample_rate: Audio sample rate (Hz)
        """
        self.start_sample = start_sample
        self.device_label = device_label
        self.sample_rate = sample_rate
        self.start_time = time.time()

        # Audio buffer
        self.audio_chunks: List[np.ndarray] = []

        # Metadata
        self.energy_profile: List[Dict[str, float]] = []
        self.speaker_events: List[Any] = []  # DiarizationEvent from diart

        # Current sample position (updated as audio appended)
        self.current_sample = start_sample

    def append_audio(self, chunk: np.ndarray):
        """
        Add audio chunk to buffer.

        Args:
            chunk: Audio samples to append
        """
        self.audio_chunks.append(chunk.copy())
        self.current_sample += len(chunk)

    def add_energy_sample(self, time_ms: int, rms: float):
        """
        Add energy measurement to profile.

        Args:
            time_ms: Timestamp in milliseconds
            rms: RMS energy value
        """
        self.energy_profile.append({
            "time_ms": time_ms,
            "rms": float(rms)
        })

    def add_speaker_event(self, event: Any):
        """
        Add speaker diarization event.

        Args:
            event: DiarizationEvent from diart AudioListener
        """
        self.speaker_events.append(event)

    def get_audio(self) -> np.ndarray:
        """
        Get concatenated audio buffer.

        Returns:
            Numpy array of all audio samples
        """
        if not self.audio_chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(self.audio_chunks)

    def get_duration_seconds(self) -> float:
        """Get utterance duration in seconds."""
        return (self.current_sample - self.start_sample) / self.sample_rate

    def build_metadata(
        self,
        transcript: str,
        whisper_confidence: float
    ) -> Dict[str, Any]:
        """
        Build rich metadata structure for TranscriptEntry.

        Args:
            transcript: Whisper transcription text
            whisper_confidence: Whisper confidence score

        Returns:
            Dictionary with complete utterance metadata
        """
        audio = self.get_audio()
        end_sample = self.start_sample + len(audio)

        # Extract speaker segments from diart events
        speakers = self._extract_speaker_segments()

        # Calculate conversation context
        unique_speakers = set(s['speaker'] for s in speakers if s['speaker'])
        conversation_context = {
            "multi_speaker": len(unique_speakers) > 1,
            "speaker_count": len(unique_speakers),
            "likely_conversation": len(speakers) > 1,  # Multiple speaker transitions
            "dominant_speaker": self._get_dominant_speaker(speakers) if speakers else None
        }

        # Build metadata structure
        metadata = {
            "utterance_id": f"utt_{int(time.time() * 1000)}",
            "device_label": self.device_label,
            "recording_start_ms": int(self.start_sample * 1000 / self.sample_rate),
            "recording_end_ms": int(end_sample * 1000 / self.sample_rate),
            "duration_ms": int(self.get_duration_seconds() * 1000),
            "transcript": transcript,
            "speakers": speakers,
            "energy_profile": self.energy_profile,
            "speaker_transitions": max(0, len(speakers) - 1) if speakers else 0,
            "conversation_context": conversation_context,
            "whisper_confidence": float(whisper_confidence),
            "is_meaningful": True  # Only meaningful speech reaches this point
        }

        logger.debug(
            f"Built metadata for utterance: {metadata['utterance_id']}, "
            f"duration={metadata['duration_ms']}ms, "
            f"speakers={len(speakers)}, "
            f"transcript_len={len(transcript)}"
        )

        return metadata

    def _extract_speaker_segments(self) -> List[Dict[str, Any]]:
        """
        Build speaker segment list from diart events.

        Returns:
            List of speaker segments with timing
        """
        if not self.speaker_events:
            return []

        segments = []
        current_speaker = None
        current_start_ms = None

        for event in self.speaker_events:
            # Skip events without speaker
            if event.speaker is None:
                continue

            event_time_ms = int(event.timestamp * 1000)

            if event.speaker != current_speaker:
                # Speaker change
                if current_speaker is not None:
                    # Close previous segment
                    segments.append({
                        "speaker": current_speaker,
                        "start_ms": current_start_ms,
                        "end_ms": event_time_ms
                    })

                # Start new segment
                current_speaker = event.speaker
                current_start_ms = event_time_ms

        # Close final segment
        if current_speaker is not None:
            end_ms = int((self.start_sample + len(self.get_audio())) * 1000 / self.sample_rate)
            segments.append({
                "speaker": current_speaker,
                "start_ms": current_start_ms,
                "end_ms": end_ms
            })

        return segments

    def _get_dominant_speaker(self, segments: List[Dict[str, Any]]) -> Optional[str]:
        """
        Get the speaker who spoke for the longest duration.

        Args:
            segments: List of speaker segments

        Returns:
            Speaker ID of dominant speaker
        """
        if not segments:
            return None

        # Calculate duration for each speaker
        speaker_durations = {}
        for seg in segments:
            speaker = seg['speaker']
            duration = seg['end_ms'] - seg['start_ms']
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

        # Return speaker with longest total duration
        return max(speaker_durations, key=speaker_durations.get)

    def __repr__(self) -> str:
        return (
            f"UtteranceBuffer(device={self.device_label}, "
            f"duration={self.get_duration_seconds():.2f}s, "
            f"chunks={len(self.audio_chunks)}, "
            f"energy_samples={len(self.energy_profile)}, "
            f"speaker_events={len(self.speaker_events)})"
        )
