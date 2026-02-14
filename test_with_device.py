"""Test audio capture with device 5"""
from ambient_subconscious.stream import StreamState
import time

print("=" * 60)
print("TESTING WITH DEVICE 5 (Realtek Microphone)")
print("=" * 60)

# Create stream with device 5
stream = StreamState(
    storage_format="jsonl",
    base_storage_path="data/sessions",
    device_name="Desktop",
    device=5  # Realtek Microphone
)

print(f"\nDevice: {stream.listener.device}")
print(f"Audio device: {stream.listener.audio_device}")

frame_count = [0]

def count_frames(frame):
    frame_count[0] += 1
    if frame_count[0] % 5 == 0:
        print(f"[{frame.timestamp:.1f}s] Frames: {frame_count[0]}, Speaker: {frame.speaker_prediction}")

stream.on_frame_created = count_frames

print("\n" + "=" * 60)
print("Recording 10 seconds - SPEAK INTO YOUR MICROPHONE!")
print("=" * 60 + "\n")

session_id = stream.start_session()
try:
    stream.start_listening(duration=10)
    time.sleep(11)
except KeyboardInterrupt:
    print("\nStopped")
finally:
    stream.end_session()

print(f"\n{'=' * 60}")
print(f"RESULTS")
print(f"{'=' * 60}")
print(f"Total frames captured: {frame_count[0]}")
print(f"Session: {session_id}")
print(f"\nCheck data/sessions/{session_id}/ for stored frames")
