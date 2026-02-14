"""Build clean utterances from overlapping diart segments"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import soundfile as sf


@dataclass
class Utterance:
    """
    A clean utterance with speaker attribution.

    Built from potentially multiple overlapping diart frames,
    represents a single continuous speech segment.
    """
    audio: np.ndarray              # Audio samples for this utterance
    speaker_id: str                # Speaker attribution (e.g., "speaker0")
    start_time: float              # Start timestamp (seconds)
    end_time: float                # End timestamp (seconds)
    start_sample: int              # Start sample position
    end_sample: int                # End sample position
    num_frames: int                # How many overlapping frames contributed
    frame_ids: List[str]           # Frame IDs that contributed to this utterance
    confidence: float              # Average confidence across frames

    def duration(self) -> float:
        """Duration of utterance in seconds"""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "speaker_id": self.speaker_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "num_frames": self.num_frames,
            "frame_ids": self.frame_ids,
            "confidence": self.confidence,
            "duration": self.duration()
        }


class UtteranceBuilder:
    """
    Builds clean utterances from overlapping diart segments.

    Diart uses overlapping sliding windows (500ms steps), so the same speech
    segment appears in multiple frames. This class merges overlapping frames
    into clean utterances with speaker attribution.

    Strategy (Option A): Use final diart prediction after session completes.
    """

    def __init__(
        self,
        overlap_threshold: float = 1.0,
        min_duration: float = 1.0,
        sample_rate: int = 16000
    ):
        """
        Args:
            overlap_threshold: Time threshold (seconds) to consider frames overlapping
                              Default 1.0s works well for diart's 500ms sliding window
            min_duration: Minimum duration (seconds) for standalone utterances
                         Shorter utterances are merged with adjacent ones
                         Default 1.0s provides good quality for Whisper
            sample_rate: Audio sample rate
        """
        self.overlap_threshold = overlap_threshold
        self.min_duration = min_duration
        self.sample_rate = sample_rate

    def build_utterances(
        self,
        frames: List[Dict[str, Any]],
        audio: np.ndarray
    ) -> List[Utterance]:
        """
        Merge overlapping segments into utterances.

        Args:
            frames: Raw frames from diart (overlapping)
            audio: Audio data (samples)

        Returns:
            List of clean utterances with speaker attribution
        """
        if not frames:
            return []

        # Sort frames by timestamp
        sorted_frames = sorted(frames, key=lambda f: f['timestamp'])

        # Group overlapping segments
        segment_groups = self._group_overlapping_segments(sorted_frames)

        # Build utterances from groups
        utterances = []
        for group in segment_groups:
            utterance = self._build_utterance_from_group(group, audio)
            if utterance:
                utterances.append(utterance)

        # Merge short utterances with adjacent ones
        utterances = self._merge_short_utterances(utterances, audio)

        return utterances

    def _merge_short_utterances(
        self,
        utterances: List[Utterance],
        audio: np.ndarray
    ) -> List[Utterance]:
        """
        Merge short utterances with adjacent ones.

        Strategy:
        - If utterance < min_duration AND same speaker as previous: merge with previous
        - Else if utterance < min_duration AND same speaker as next: merge with next
        - Else: keep as-is (different speakers or acceptable duration)

        Args:
            utterances: List of utterances to process
            audio: Full audio array

        Returns:
            List of utterances with short ones merged
        """
        if not utterances or len(utterances) == 1:
            return utterances

        merged = []
        skip_next = False

        for i, utt in enumerate(utterances):
            if skip_next:
                skip_next = False
                continue

            # Check if utterance is too short
            if utt.duration() < self.min_duration:
                # Try to merge with previous (if same speaker)
                if merged and merged[-1].speaker_id == utt.speaker_id:
                    # Merge with previous
                    prev = merged[-1]
                    merged_utt = self._merge_two_utterances(prev, utt, audio)
                    merged[-1] = merged_utt
                    continue

                # Try to merge with next (if same speaker)
                elif i + 1 < len(utterances) and utterances[i + 1].speaker_id == utt.speaker_id:
                    # Merge with next
                    next_utt = utterances[i + 1]
                    merged_utt = self._merge_two_utterances(utt, next_utt, audio)
                    merged.append(merged_utt)
                    skip_next = True
                    continue

            # Keep utterance as-is (meets duration or can't merge)
            merged.append(utt)

        return merged

    def _merge_two_utterances(
        self,
        utt1: Utterance,
        utt2: Utterance,
        audio: np.ndarray
    ) -> Utterance:
        """
        Merge two adjacent utterances into one.

        Args:
            utt1: First utterance (earlier in time)
            utt2: Second utterance (later in time)
            audio: Full audio array

        Returns:
            Merged utterance
        """
        # Time bounds: from first to second
        start_time = utt1.start_time
        end_time = utt2.end_time

        # Sample bounds: from first to second
        start_sample = utt1.start_sample
        end_sample = utt2.end_sample

        # Extract merged audio segment
        audio_segment = audio[start_sample:end_sample]

        # Merge metadata
        merged_frame_ids = utt1.frame_ids + utt2.frame_ids
        total_frames = utt1.num_frames + utt2.num_frames
        avg_confidence = (utt1.confidence * utt1.num_frames + utt2.confidence * utt2.num_frames) / total_frames

        return Utterance(
            audio=audio_segment,
            speaker_id=utt1.speaker_id,  # Should be same as utt2
            start_time=start_time,
            end_time=end_time,
            start_sample=start_sample,
            end_sample=end_sample,
            num_frames=total_frames,
            frame_ids=merged_frame_ids,
            confidence=avg_confidence
        )

    def _group_overlapping_segments(
        self,
        frames: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group frames that overlap temporally.

        Two frames are considered overlapping if they're within
        overlap_threshold seconds of each other.

        Args:
            frames: Sorted list of frames

        Returns:
            List of frame groups (each group = one potential utterance)
        """
        if not frames:
            return []

        groups = []
        current_group = [frames[0]]

        for frame in frames[1:]:
            # Check if this frame overlaps with current group
            last_timestamp = current_group[-1]['timestamp']

            # Also check speaker consistency
            last_speaker = current_group[-1]['speaker_prediction']
            current_speaker = frame['speaker_prediction']

            if (abs(frame['timestamp'] - last_timestamp) < self.overlap_threshold and
                current_speaker == last_speaker):
                # Overlapping with same speaker - add to current group
                current_group.append(frame)
            else:
                # Not overlapping OR different speaker - start new group
                groups.append(current_group)
                current_group = [frame]

        # Don't forget the last group
        if current_group:
            groups.append(current_group)

        return groups

    def _build_utterance_from_group(
        self,
        group: List[Dict[str, Any]],
        audio: np.ndarray
    ) -> Utterance:
        """
        Build a single utterance from a group of overlapping frames.

        Strategy: Use final prediction (most stable) for speaker attribution,
        but include audio from first to last frame.

        Args:
            group: List of overlapping frames
            audio: Full audio array

        Returns:
            Utterance object
        """
        if not group:
            return None

        # Use final frame for speaker attribution (most stable prediction)
        final_frame = group[-1]
        speaker_id = final_frame['speaker_prediction']

        # Time bounds: from first frame to last frame
        start_time = group[0]['timestamp']
        end_time = final_frame['timestamp']

        # Sample bounds: from first frame's sample position
        # to last frame's sample position + small buffer (500ms)
        start_sample = group[0]['sample_position']
        buffer_samples = int(0.5 * self.sample_rate)  # 500ms buffer
        end_sample = final_frame['sample_position'] + buffer_samples

        # Ensure we don't go past audio bounds
        end_sample = min(end_sample, len(audio))

        # Extract audio segment
        audio_segment = audio[start_sample:end_sample]

        # Collect metadata
        frame_ids = [f['frame_id'] for f in group]
        avg_confidence = np.mean([f['confidence'] for f in group])

        return Utterance(
            audio=audio_segment,
            speaker_id=speaker_id,
            start_time=start_time,
            end_time=end_time,
            start_sample=start_sample,
            end_sample=end_sample,
            num_frames=len(group),
            frame_ids=frame_ids,
            confidence=avg_confidence
        )

    def build_from_session(self, session_path: Path) -> List[Utterance]:
        """
        Build utterances from a session directory.

        Args:
            session_path: Path to session directory containing:
                - frames.jsonl
                - audio.wav

        Returns:
            List of utterances
        """
        # Load frames
        frames = []
        frames_file = session_path / "frames.jsonl"
        with open(frames_file, 'r') as f:
            for line in f:
                frames.append(json.loads(line))

        # Load audio
        audio_file = session_path / "audio.wav"
        audio, sr = sf.read(audio_file)

        # Ensure sample rate matches
        if sr != self.sample_rate:
            print(f"Warning: Audio sample rate {sr} != expected {self.sample_rate}")

        # Build utterances
        return self.build_utterances(frames, audio)

    def save_utterances(
        self,
        utterances: List[Utterance],
        output_dir: Path,
        save_audio: bool = True
    ):
        """
        Save utterances to directory.

        Args:
            utterances: List of utterances to save
            output_dir: Directory to save to
            save_audio: Whether to save audio segments
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = []
        for i, utt in enumerate(utterances):
            utt_dict = utt.to_dict()
            utt_dict['utterance_id'] = f"utt_{i:04d}"
            metadata.append(utt_dict)

            # Save audio if requested
            if save_audio:
                audio_path = output_dir / f"utt_{i:04d}.wav"
                sf.write(audio_path, utt.audio, self.sample_rate)
                utt_dict['audio_path'] = str(audio_path)

        # Save metadata as JSON
        metadata_path = output_dir / "utterances.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(utterances)} utterances to {output_dir}")
        print(f"  - Metadata: {metadata_path}")
        if save_audio:
            print(f"  - Audio files: utt_*.wav")


def load_utterances(output_dir: Path) -> List[Utterance]:
    """
    Load utterances from directory.

    Args:
        output_dir: Directory containing utterances.json and audio files

    Returns:
        List of utterances
    """
    output_dir = Path(output_dir)
    metadata_path = output_dir / "utterances.json"

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    utterances = []
    for utt_dict in metadata:
        # Load audio if available
        audio = None
        if 'audio_path' in utt_dict:
            audio_path = Path(utt_dict['audio_path'])
            if audio_path.exists():
                audio, _ = sf.read(audio_path)

        utterance = Utterance(
            audio=audio,
            speaker_id=utt_dict['speaker_id'],
            start_time=utt_dict['start_time'],
            end_time=utt_dict['end_time'],
            start_sample=utt_dict['start_sample'],
            end_sample=utt_dict['end_sample'],
            num_frames=utt_dict['num_frames'],
            frame_ids=utt_dict['frame_ids'],
            confidence=utt_dict['confidence']
        )
        utterances.append(utterance)

    return utterances
