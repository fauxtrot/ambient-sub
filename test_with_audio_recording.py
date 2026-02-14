"""Test diarization + audio recording together"""
from ambient_subconscious.stream import StreamState, StreamConfig
import time

print("=" * 70)
print("DIARIZATION + AUDIO RECORDING TEST")
print("=" * 70)

# Load config from .env
device = StreamConfig.get_inference_device()
audio_device = StreamConfig.get_audio_input_device()
record_audio = StreamConfig.get_record_audio()
storage_path = StreamConfig.get_storage_path()
device_name = StreamConfig.get_device_name()
recording_duration = StreamConfig.get_recording_duration()

print(f"\nConfiguration:")
print(f"  Inference device: {device}")
print(f"  Audio input device: {audio_device}")
print(f"  Record audio: {record_audio}")
print(f"  Storage path: {storage_path}")
if recording_duration:
    print(f"  Recording duration: {recording_duration}s ({recording_duration/60:.1f} min)")
else:
    print(f"  Recording duration: Unlimited (Ctrl+C to stop)")

# Create stream with audio recording enabled
stream = StreamState(
    storage_format="jsonl",
    base_storage_path=storage_path,
    device_name=device_name,
    record_audio=record_audio,  # Enable audio recording
    device=device,  # Inference device (CUDA/CPU)
    audio_device=audio_device  # Audio input device
)

frame_count = [0]

def count_frames(frame):
    """Count frames as they're created"""
    frame_count[0] += 1
    if frame_count[0] % 5 == 0:
        print(f"  [{frame.timestamp:.1f}s] Frame #{frame_count[0]}: {frame.event_type}, speaker={frame.speaker_prediction}")

stream.on_frame_created = count_frames

print("\n" + "=" * 70)
if recording_duration:
    print(f"RECORDING {recording_duration/60:.1f} MINUTES - SPEAK INTO YOUR MICROPHONE!")
else:
    print("RECORDING (UNLIMITED) - SPEAK INTO YOUR MICROPHONE!")
print("=" * 70)
print("\nThis will:")
print("  1. Run speaker diarization and save frames to JSONL")
print("  2. Record raw audio and save to WAV file")
print("  3. Link the audio file in session metadata")
if not recording_duration:
    print("  4. Press Ctrl+C to stop recording")
print()

session_id = stream.start_session()

try:
    stream.start_listening(duration=recording_duration)
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    stream.end_session()

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Session: {session_id}")
print(f"Frames captured: {frame_count[0]}")
print(f"\nCheck the session directory:")
print(f"  data/sessions/{session_id}/")
print(f"    - metadata.json (includes audio_path)")
print(f"    - frames.jsonl (diarization events)")
if record_audio:
    print(f"    - audio.wav (raw recording)")
