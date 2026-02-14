"""Example: Record continuous stream with Frame storage"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

from ambient_subconscious.stream.stream_state import StreamState
from ambient_subconscious.stream.config import StreamConfig


def main():
    print("Ambient Subconscious - Stream Recording Example")
    print("=" * 60)

    # Initialize stream state with config
    stream = StreamState(
        storage_format=StreamConfig.get_storage_format(),
        base_storage_path=StreamConfig.get_storage_path(),
        device_name=StreamConfig.get_device_name(),
        recording_mode=StreamConfig.get_recording_mode(),
        device=StreamConfig.get_inference_device(),
    )

    # Register enrichment hooks
    frame_count = [0]  # Use list for closure

    def on_frame_created(frame):
        """Called every time a frame is created"""
        frame_count[0] += 1
        if frame.event_type == "speaker_change":
            print(f"[{frame.timestamp:6.2f}s] Speaker change: {frame.speaker_prediction}")
        elif frame_count[0] % 20 == 0:  # Print every 20th frame
            print(f"[{frame.timestamp:6.2f}s] Update (total frames: {frame_count[0]})")

    def on_speaker_change(frame):
        """Called on speaker change events - could trigger visual capture"""
        print(f"[TRIGGER] Visual capture opportunity for speaker: {frame.speaker_prediction}")
        # Future: trigger screenshot, YOLO detection
        # frame.add_visual_context(detections=[...])

    stream.on_frame_created = on_frame_created
    stream.on_speaker_change = on_speaker_change

    # Start session and listen
    print("\nStarting 30-second recording session...")
    print("Speak into your microphone to test speaker detection!\n")

    session_id = stream.start_session()

    try:
        stream.start_listening(duration=30)  # 30 seconds
        time.sleep(31)
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        stream.end_session()

    # Load and inspect frames
    print("\n" + "=" * 60)
    print("Loading session frames...")
    frames = stream.load_session_frames(session_id)
    print(f"Loaded {len(frames)} frames")

    # Show some stats
    speaker_changes = [f for f in frames if f.event_type == "speaker_change"]
    print(f"Speaker changes: {len(speaker_changes)}")

    speakers = set(f.speaker_prediction for f in frames if f.speaker_prediction)
    print(f"Unique speakers: {speakers}")

    # Show first few frames
    if frames:
        print("\nFirst 5 frames:")
        for i, frame in enumerate(frames[:5]):
            print(f"  {i+1}. [{frame.timestamp:.2f}s] {frame.event_type} - {frame.speaker_prediction}")

    print(f"\nSession saved to: {stream.base_storage_path / session_id}")
    print(f"Frames file: {stream.base_storage_path / session_id / 'frames.jsonl'}")


if __name__ == "__main__":
    main()
