"""Test stream state and frame storage"""

import sys
import os
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))

from ambient_subconscious.stream.stream_state import StreamState
from ambient_subconscious.stream.frame import Frame


def test_frame_creation():
    """Test Frame dataclass"""
    print("Testing Frame creation...")

    frame = Frame(
        timestamp=1.234,
        sample_position=19744,
        speaker_prediction="speaker_0",
        confidence=0.95,
        event_type="speaker_change",
        session_id="test_session"
    )

    print(f"  Created frame: {frame.frame_id}")

    # Test serialization
    json_str = frame.to_json()
    frame2 = Frame.from_json(json_str)
    assert frame2.timestamp == frame.timestamp
    assert frame2.speaker_prediction == frame.speaker_prediction

    print("  Serialization: OK")

    # Test enrichment
    frame.add_visual_context(detections=["person", "laptop"], embedding=[0.1, 0.2, 0.3])
    assert frame.visual_context is not None
    assert frame.enriched_at is not None
    print("  Visual enrichment: OK")

    frame.add_text_hypothesis("Hello world", confidence=0.9)
    assert frame.text_hypothesis == "Hello world"
    assert frame.confidence == 0.95  # Should keep max confidence
    print("  Text enrichment: OK")


def test_session_lifecycle():
    """Test session start/stop"""
    print("\nTesting session lifecycle...")

    with tempfile.TemporaryDirectory() as tmpdir:
        stream = StreamState(
            storage_format="jsonl",
            base_storage_path=tmpdir,
            device_name="TestDevice"
        )

        # Start session
        session_id = stream.start_session()
        print(f"  Started session: {session_id}")

        # Create some frames manually
        for i in range(5):
            frame = Frame(
                timestamp=i * 0.5,
                sample_position=i * 8000,
                speaker_prediction=f"speaker_{i % 2}",
                session_id=session_id,
                event_type="update"
            )
            stream.store_frame(frame)

        # End session
        stream.end_session()

        # Load frames back
        frames = stream.load_session_frames(session_id)
        assert len(frames) == 5
        print(f"  Loaded {len(frames)} frames: OK")

        # Check metadata
        metadata = stream.get_session_metadata(session_id)
        assert metadata.frame_count == 5
        print(f"  Metadata: {metadata.frame_count} frames, {metadata.duration_seconds:.1f}s")

        # Verify JSONL file is human-readable
        frames_path = Path(tmpdir) / session_id / "frames.jsonl"
        with open(frames_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 5
            print(f"  JSONL file: {len(lines)} lines, human-readable: OK")


def test_listener_integration():
    """Test with actual AudioListener (10s interactive)"""
    print("\nTesting AudioListener integration...")
    print("Speak into your microphone for 10 seconds!\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        stream = StreamState(
            storage_format="jsonl",
            base_storage_path=tmpdir
        )

        frame_count = [0]  # Use list for closure

        def count_frames(frame):
            frame_count[0] += 1
            if frame_count[0] <= 3:
                print(f"  [{frame.timestamp:6.2f}s] {frame.event_type} - {frame.speaker_prediction}")

        stream.on_frame_created = count_frames

        session_id = stream.start_session()
        stream.start_listening(duration=10)
        time.sleep(11)
        stream.end_session()

        print(f"\n  Total frames: {frame_count[0]}")

        # Verify storage
        frames = stream.load_session_frames(session_id)
        assert len(frames) == frame_count[0]
        print(f"  Verified {len(frames)} frames in storage")

        # Show some stats
        speaker_changes = [f for f in frames if f.event_type == "speaker_change"]
        speakers = set(f.speaker_prediction for f in frames if f.speaker_prediction)
        print(f"  Speaker changes: {len(speaker_changes)}")
        print(f"  Unique speakers: {speakers}")


if __name__ == "__main__":
    test_frame_creation()
    test_session_lifecycle()
    test_listener_integration()

    print("\nAll tests passed!")
