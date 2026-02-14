"""Verify diarization by playing audio excerpts from detected segments"""

import json
import soundfile as sf
import sounddevice as sd
from pathlib import Path
import time
import whisper
import numpy as np

def find_latest_session(base_path="data/sessions"):
    """Find the most recent session directory"""
    sessions_path = Path(base_path)
    if not sessions_path.exists():
        print(f"Sessions directory not found: {sessions_path}")
        return None

    # Get all session directories
    session_dirs = [d for d in sessions_path.iterdir() if d.is_dir()]

    if not session_dirs:
        print("No sessions found")
        return None

    # Sort by modification time (most recent first)
    latest = max(session_dirs, key=lambda d: d.stat().st_mtime)
    return latest

def load_session_data(session_path):
    """Load frames and audio from session"""
    frames_file = session_path / "frames.jsonl"
    audio_file = session_path / "audio.wav"
    metadata_file = session_path / "metadata.json"

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load frames
    frames = []
    with open(frames_file, 'r') as f:
        for line in f:
            frames.append(json.loads(line))

    # Load audio
    audio_data, sample_rate = sf.read(audio_file)

    return metadata, frames, audio_data, sample_rate

def deduplicate_segments(frames, overlap_threshold=0.3):
    """
    Deduplicate overlapping segments from diart's sliding window processing.

    Diart processes audio in overlapping windows, so the same speech segment
    can appear in multiple frames. This function groups overlapping segments
    and keeps one representative per group.

    Args:
        frames: List of frame dicts with 'timestamp' field
        overlap_threshold: Time threshold in seconds to consider segments as duplicates

    Returns:
        List of unique frames (deduplicated)
    """
    if not frames:
        return []

    # Sort frames by timestamp
    sorted_frames = sorted(frames, key=lambda f: f['timestamp'])

    # Group overlapping segments
    unique_frames = []
    current_group = [sorted_frames[0]]

    for frame in sorted_frames[1:]:
        # Check if this frame overlaps with the current group
        last_timestamp = current_group[-1]['timestamp']

        if abs(frame['timestamp'] - last_timestamp) < overlap_threshold:
            # Overlapping - add to current group
            current_group.append(frame)
        else:
            # Not overlapping - save current group and start new one
            # Keep the first frame from each group
            unique_frames.append(current_group[0])
            current_group = [frame]

    # Don't forget the last group
    if current_group:
        unique_frames.append(current_group[0])

    return unique_frames

def play_excerpt(audio_data, sample_rate, timestamp, duration=2.0, context=0.5):
    """
    Play an audio excerpt around a timestamp.

    Args:
        audio_data: Audio samples
        sample_rate: Sample rate
        timestamp: Center timestamp in seconds
        duration: Total duration to play
        context: How much before the timestamp to include
    """
    # Calculate sample positions
    start_time = max(0, timestamp - context)
    end_time = start_time + duration

    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Ensure we don't go past the end
    end_sample = min(end_sample, len(audio_data))

    # Extract excerpt
    excerpt = audio_data[start_sample:end_sample]

    # Play
    print(f"  Playing {duration}s excerpt from {start_time:.2f}s to {end_time:.2f}s")
    sd.play(excerpt, sample_rate)
    sd.wait()

def main():
    print("=" * 70)
    print("SESSION VERIFICATION - AUDIO EXCERPT PLAYBACK")
    print("=" * 70)

    # Find latest session
    session_path = find_latest_session()
    if not session_path:
        return

    print(f"\nLatest session: {session_path.name}")

    # Load session data
    print("Loading session data...")
    metadata, frames, audio_data, sample_rate = load_session_data(session_path)

    # Load Whisper model
    print("Loading Whisper model (base)...")
    whisper_model = whisper.load_model("base")

    print(f"\nSession info:")
    print(f"  Duration: {metadata['duration_seconds']:.1f}s")
    print(f"  Total frames: {len(frames)}")
    print(f"  Audio: {len(audio_data)} samples @ {sample_rate}Hz")

    # Transcribe entire audio clip
    print("\nTranscribing entire audio clip...")
    # Convert to float32 for Whisper
    audio_float32 = audio_data.astype(np.float32)
    full_result = whisper_model.transcribe(audio_float32, fp16=False)
    full_transcription = full_result['text'].strip()
    print(f"\nFULL CLIP TRANSCRIPTION:")
    print(f'  "{full_transcription}"')

    # Deduplicate overlapping segments
    unique_frames = deduplicate_segments(frames, overlap_threshold=0.3)

    print(f"\n  Unique segments: {len(unique_frames)} (removed {len(frames) - len(unique_frames)} duplicates)")

    print(f"\n" + "=" * 70)
    print(f"Playing excerpts for {len(unique_frames)} unique speaker segments")
    print(f"Each excerpt is 2 seconds (0.5s before segment end, 1.5s after)")
    print("=" * 70)

    # Play excerpt for each unique frame
    for i, frame in enumerate(unique_frames, 1):
        timestamp = frame['timestamp']
        speaker = frame['speaker_prediction']

        print(f"\n[{i}/{len(unique_frames)}] Segment at {timestamp:.2f}s")
        print(f"  Speaker: {speaker}")
        print(f"  Event: {frame['event_type']}")

        # Extract audio excerpt for transcription
        start_time = max(0, timestamp - 0.5)
        end_time = start_time + 2.0
        start_sample = int(start_time * sample_rate)
        end_sample = min(int(end_time * sample_rate), len(audio_data))
        excerpt = audio_data[start_sample:end_sample]

        # Transcribe with Whisper
        try:
            # Convert to float32 for Whisper
            excerpt_float32 = excerpt.astype(np.float32)
            result = whisper_model.transcribe(excerpt_float32, fp16=False)
            transcription = result['text'].strip()
            print(f"  Segment transcription: \"{transcription}\"")
        except Exception as e:
            print(f"  Transcription error: {e}")

        # Play excerpt
        print(f"  Playing {2.0}s excerpt from {start_time:.2f}s to {end_time:.2f}s")
        sd.play(excerpt, sample_rate)
        sd.wait()

        # Brief pause between segments
        if i < len(unique_frames):
            time.sleep(0.5)

    print("\n" + "=" * 70)
    print("Verification complete!")
    print("=" * 70)
    print(f"\nSession files:")
    print(f"  {session_path / 'audio.wav'}")
    print(f"  {session_path / 'frames.jsonl'}")
    print(f"  {session_path / 'metadata.json'}")

if __name__ == "__main__":
    main()
