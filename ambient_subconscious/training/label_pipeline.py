"""
Multi-source label pipeline for generating ground truth.

Updated to use VAD + Whisper + Diart alignment for complete coverage.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import soundfile as sf

from ..stream.frame import Frame


@dataclass
class GroundTruthSegment:
    """
    Ground truth labels for a single segment.

    This represents what the model will learn to predict.
    Segments are VAD-based (complete coverage), with speaker attribution
    aligned from diart.
    """
    # Identification
    segment_id: str
    session_id: str

    # Temporal
    start_time: float
    end_time: float
    duration: float

    # Stage 0: Binary detection
    has_sound: bool

    # Stage 1: Speaker awareness
    has_speaker: Optional[bool] = None
    speaker_changed: Optional[bool] = None

    # Stage 2: Speaker identity
    speaker_id: Optional[str] = None
    speaker_confidence: float = 0.0  # Overlap ratio with diart

    # Stage 4: Rich recognition
    transcription: Optional[str] = None
    language: Optional[str] = None

    # Quality metrics
    detected_by_diart: bool = False
    gap_from_diart: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, ensuring JSON-serializable types"""
        d = asdict(self)
        # Convert numpy types to native Python types for JSON
        for key, value in d.items():
            if isinstance(value, (np.bool_, np.integer, np.floating)):
                d[key] = value.item()
        return d


class Session:
    """Container for session data"""
    def __init__(self, session_id: str, frames: List[Frame], audio: np.ndarray, metadata: Dict):
        self.session_id = session_id
        self.frames = frames
        self.audio = audio
        self.metadata = metadata
        self.sample_rate = 16000  # diart default


class GroundTruth:
    """
    Container for ground truth labels.

    Updated to hold segment-based labels instead of frame-based.
    """
    def __init__(self, segments: List[GroundTruthSegment]):
        self.segments = segments

    def __getitem__(self, index: int) -> GroundTruthSegment:
        """Get segment by index"""
        return self.segments[index]

    def __len__(self) -> int:
        """Number of segments"""
        return len(self.segments)

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert all segments to list of dicts"""
        return [seg.to_dict() for seg in self.segments]


class LabelSource(ABC):
    """Abstract base class for label sources"""

    @abstractmethod
    def generate_labels(
        self,
        session_path: Path,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[GroundTruthSegment]:
        """
        Generate labels for a session.

        Args:
            session_path: Path to session directory
            audio: Audio samples
            sample_rate: Sample rate

        Returns:
            List of GroundTruthSegment objects
        """
        pass


class WhisperVADSource(LabelSource):
    """
    Generate labels using Whisper's VAD + transcription.

    This is the PRIMARY source - provides complete coverage.
    """

    def __init__(self, whisper_model=None, model_size="base"):
        self.whisper_model = whisper_model
        self.model_size = model_size

    def generate_labels(
        self,
        session_path: Path,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[GroundTruthSegment]:
        """Generate VAD-based segments with transcription"""
        import whisper

        # Load model if needed
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model(self.model_size)

        # Transcribe with VAD
        result = self.whisper_model.transcribe(
            audio.astype('float32'),
            fp16=False,
            word_timestamps=True
        )

        # Create segments
        segments = []
        for i, seg in enumerate(result.get('segments', [])):
            segment = GroundTruthSegment(
                segment_id=f"vad_{i:04d}",
                session_id=session_path.name,
                start_time=seg['start'],
                end_time=seg['end'],
                duration=seg['end'] - seg['start'],
                has_sound=True,
                has_speaker=True,  # Assume speech has speaker
                transcription=seg['text'].strip(),
                language=result.get('language', 'unknown'),
            )
            segments.append(segment)

        return segments


class DiartLabelSource(LabelSource):
    """
    Generate speaker labels from diart frames.

    This is the SECONDARY source - provides speaker attribution.
    """

    def __init__(self, overlap_threshold=1.0, min_duration=1.0):
        self.overlap_threshold = overlap_threshold
        self.min_duration = min_duration

    def generate_labels(
        self,
        session_path: Path,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[GroundTruthSegment]:
        """Generate speaker-attributed utterances"""
        from .utterance_builder import UtteranceBuilder

        builder = UtteranceBuilder(
            overlap_threshold=self.overlap_threshold,
            min_duration=self.min_duration,
            sample_rate=sample_rate
        )

        utterances = builder.build_from_session(session_path)

        # Convert to segments
        segments = []
        prev_speaker = None

        for i, utt in enumerate(utterances):
            speaker_changed = (prev_speaker is not None and
                             utt.speaker_id != prev_speaker)

            segment = GroundTruthSegment(
                segment_id=f"diart_{i:04d}",
                session_id=session_path.name,
                start_time=utt.start_time,
                end_time=utt.end_time,
                duration=utt.duration(),
                has_sound=True,
                has_speaker=True,
                speaker_changed=speaker_changed,
                speaker_id=utt.speaker_id,
                speaker_confidence=utt.confidence,
                detected_by_diart=True,
            )
            segments.append(segment)
            prev_speaker = utt.speaker_id

        return segments


class LabelPipeline:
    """
    Orchestrates VAD + Whisper + Diart alignment.

    Strategy:
    1. Use Whisper VAD as primary (complete coverage)
    2. Align diart speakers with VAD segments
    3. Create enriched training samples
    """

    def __init__(self, whisper_model=None):
        self.whisper_vad = WhisperVADSource(whisper_model=whisper_model)
        self.diart_source = DiartLabelSource()

    def generate_ground_truth(self, session_path: Path) -> GroundTruth:
        """
        Generate enriched ground truth.

        Args:
            session_path: Path to session directory

        Returns:
            GroundTruth with aligned VAD + speaker labels
        """
        # Load audio
        audio_file = session_path / "audio.wav"
        audio, sample_rate = sf.read(audio_file)

        # Get segments from both sources
        vad_segments = self.whisper_vad.generate_labels(session_path, audio, sample_rate)
        diart_segments = self.diart_source.generate_labels(session_path, audio, sample_rate)

        # Align them
        enriched_segments = self._align_segments(vad_segments, diart_segments)

        return GroundTruth(enriched_segments)

    def _align_segments(
        self,
        vad_segments: List[GroundTruthSegment],
        diart_segments: List[GroundTruthSegment]
    ) -> List[GroundTruthSegment]:
        """
        Align VAD segments with diart speakers.

        For each VAD segment, find overlapping diart segment
        and copy speaker attribution.
        """
        enriched = []

        for vad in vad_segments:
            # Find best overlapping diart segment
            best_diart = None
            best_overlap = 0

            for diart in diart_segments:
                # Calculate temporal overlap
                overlap_start = max(vad.start_time, diart.start_time)
                overlap_end = min(vad.end_time, diart.end_time)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_diart = diart

            # Create enriched segment
            enriched_segment = GroundTruthSegment(
                segment_id=vad.segment_id,
                session_id=vad.session_id,

                # From VAD (complete coverage)
                start_time=vad.start_time,
                end_time=vad.end_time,
                duration=vad.duration,
                has_sound=vad.has_sound,
                transcription=vad.transcription,
                language=vad.language,

                # From diart (speaker attribution)
                speaker_id=best_diart.speaker_id if best_diart else "unknown",
                has_speaker=best_diart is not None,
                speaker_changed=best_diart.speaker_changed if best_diart else False,
                speaker_confidence=best_overlap / vad.duration if vad.duration > 0 else 0.0,

                # Quality metrics
                detected_by_diart=best_overlap > 0,
                gap_from_diart=best_overlap == 0,
            )

            enriched.append(enriched_segment)

        return enriched


# Backward compatibility with old API
class AudioFeatureLabelSource(LabelSource):
    """Simple amplitude-based sound detection (legacy)"""

    def __init__(self, threshold: float = 0.01, frame_duration: float = 0.5):
        self.threshold = threshold
        self.frame_duration = frame_duration

    def generate_labels(
        self,
        session_path: Path,
        audio: np.ndarray,
        sample_rate: int
    ) -> List[GroundTruthSegment]:
        """Generate simple sound detection labels"""
        frame_samples = int(self.frame_duration * sample_rate)
        segments = []

        for i in range(0, len(audio), frame_samples):
            frame = audio[i:i + frame_samples]
            rms = np.sqrt(np.mean(frame ** 2))
            has_sound = rms > self.threshold

            segment = GroundTruthSegment(
                segment_id=f"audio_{i // frame_samples:04d}",
                session_id=session_path.name,
                start_time=i / sample_rate,
                end_time=(i + len(frame)) / sample_rate,
                duration=len(frame) / sample_rate,
                has_sound=has_sound,
                speaker_confidence=float(rms / self.threshold) if has_sound else 0.0,
            )
            segments.append(segment)

        return segments
