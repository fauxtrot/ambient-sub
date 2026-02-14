"""Data pipeline for loading sessions and creating training samples"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import numpy as np
import soundfile as sf
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio

from ..stream.frame import Frame
from .label_pipeline import Session, GroundTruth
from .recognition_frame import RecognitionFrame


class SessionLoader:
    """Load session data from disk"""

    def __init__(self, base_path: str = "data/sessions"):
        self.base_path = Path(base_path)

    def load_session(self, session_id: str) -> Session:
        """
        Load a session from disk.

        Args:
            session_id: Session ID (directory name)

        Returns:
            Session object with frames, audio, and metadata
        """
        session_path = self.base_path / session_id

        if not session_path.exists():
            raise FileNotFoundError(f"Session {session_id} not found at {session_path}")

        # Load metadata
        metadata_path = session_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load frames
        frames = []
        frames_path = session_path / "frames.jsonl"

        if frames_path.exists():
            with open(frames_path, 'r') as f:
                for line in f:
                    if line.strip():
                        frames.append(Frame.from_json(line))

        # Load audio
        audio = None
        audio_path = session_path / "audio.wav"

        if audio_path.exists():
            audio, sr = sf.read(str(audio_path))
            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio[:, 0]
        else:
            # Create silent audio if no file (for testing)
            audio = np.zeros(16000 * 10)  # 10 seconds of silence

        return Session(session_id, frames, audio, metadata)

    def list_sessions(self) -> List[str]:
        """List all available session IDs"""
        sessions = []
        for path in self.base_path.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                sessions.append(path.name)
        return sorted(sessions)


class EncodecTokenizer:
    """Tokenize audio using Encodec"""

    def __init__(self, bandwidth: float = 6.0):
        """
        Args:
            bandwidth: Encodec bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0)
        """
        self.bandwidth = bandwidth
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()  # Inference mode

    def tokenize(self, audio: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Convert audio to encodec tokens.

        Args:
            audio: Audio waveform (mono)
            sample_rate: Sample rate of input audio

        Returns:
            Tensor of encodec tokens [codebooks, seq_len]
        """
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Add channel dimension if needed
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)  # [1, samples]

        # Convert to model's expected format (24kHz, 1 channel)
        audio = convert_audio(audio, sample_rate, self.model.sample_rate, self.model.channels)

        # Add batch dimension
        audio = audio.unsqueeze(0)  # [batch=1, channels=1, samples]

        # Encode
        with torch.no_grad():
            encoded = self.model.encode(audio)

        # Extract codes [batch=1, n_codebooks, seq_len]
        codes = encoded[0][0]  # [n_codebooks, seq_len]

        return codes

    def get_frame_tokens(self, audio: np.ndarray, frame_position: int, window_size: int = 4000) -> List[int]:
        """
        Get encodec tokens for a specific frame position.

        Args:
            audio: Full audio array
            frame_position: Sample position of frame
            window_size: Window size around frame (samples)

        Returns:
            List of token IDs (flattened from codebooks)
        """
        # Extract audio window around frame
        start = max(0, frame_position - window_size // 2)
        end = min(len(audio), frame_position + window_size // 2)
        window = audio[start:end]

        # Tokenize
        codes = self.tokenize(window)

        # Flatten codebooks: [n_codebooks, seq_len] -> [n_codebooks * seq_len]
        # Use first codebook for Stage 0 simplicity
        tokens = codes[0].tolist()  # First codebook

        return tokens


class TrainingSample:
    """Single training sample"""

    def __init__(
        self,
        encodec_tokens: List[int],
        ground_truth: Dict[str, Any],
        where: str = "unknown",
        timestamp: float = 0.0,
        session_id: str = "",
    ):
        self.encodec_tokens = encodec_tokens
        self.ground_truth = ground_truth
        self.where = where
        self.timestamp = timestamp
        self.session_id = session_id


class TrainingSampleGenerator:
    """Generate training samples from sessions"""

    def __init__(self, tokenizer: Optional[EncodecTokenizer] = None):
        """
        Args:
            tokenizer: EncodecTokenizer instance (created if None)
        """
        self.tokenizer = tokenizer or EncodecTokenizer()

    def generate_samples(
        self,
        session: Session,
        ground_truth: GroundTruth,
        context_window: int = 0  # Stage 0: no context
    ) -> List[TrainingSample]:
        """
        Generate training samples from a session.

        Args:
            session: Session data
            ground_truth: Ground truth labels
            context_window: Number of previous frames to include (0 for Stage 0)

        Returns:
            List of TrainingSamples
        """
        samples = []

        # If no frames, create samples at fixed intervals
        if not session.frames:
            # Create samples every 0.5 seconds
            sample_rate = session.sample_rate
            interval = int(0.5 * sample_rate)  # 0.5 second windows

            for i, frame_pos in enumerate(range(0, len(session.audio), interval)):
                if i >= len(ground_truth):
                    break

                # Get tokens for this position
                tokens = self.tokenizer.get_frame_tokens(session.audio, frame_pos)

                # Create sample
                sample = TrainingSample(
                    encodec_tokens=tokens,
                    ground_truth=ground_truth[i],
                    where=session.metadata.get("device_name", "unknown"),
                    timestamp=frame_pos / sample_rate,
                    session_id=session.session_id,
                )
                samples.append(sample)

            return samples

        # Generate samples from frames
        for i, frame in enumerate(session.frames):
            if i >= len(ground_truth):
                break

            # Get encodec tokens for this frame
            tokens = self.tokenizer.get_frame_tokens(session.audio, frame.sample_position)

            # Create sample
            sample = TrainingSample(
                encodec_tokens=tokens,
                ground_truth=ground_truth[i],
                where=session.metadata.get("device_name", "unknown"),
                timestamp=frame.timestamp,
                session_id=session.session_id,
            )

            samples.append(sample)

        return samples
