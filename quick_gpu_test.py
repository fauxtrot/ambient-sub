"""Quick GPU test with StreamState"""
from ambient_subconscious.stream import StreamState, StreamConfig
import time

print("Testing GPU-accelerated streaming...")
stream = StreamState(
    storage_format="jsonl",
    base_storage_path="data/sessions",
    device_name="Desktop",
)

print(f"Device: {stream.listener.device}")
print("\nThis should now use CUDA!")
print("Press Ctrl+C to stop early\n")

frame_count = [0]
def count_frames(frame):
    frame_count[0] += 1
    if frame_count[0] % 5 == 0:
        print(f"[{frame.timestamp:.1f}s] Frames: {frame_count[0]}, Speaker: {frame.speaker_prediction}")

stream.on_frame_created = count_frames

session_id = stream.start_session()
try:
    stream.start_listening(duration=10)  # 10 seconds
    time.sleep(11)
except KeyboardInterrupt:
    print("\nStopped")
finally:
    stream.end_session()

print(f"\nTotal frames captured: {frame_count[0]}")
