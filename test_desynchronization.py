"""Test for desynchronization between saved audio and analyzed frames"""

import sys
from pathlib import Path
import json
import soundfile as sf
import numpy as np

print("=" * 80)
print("DESYNCHRONIZATION TEST")
print("=" * 80)
print("\nThis test re-runs diarization on saved audio.wav and compares")
print("with the original frames.jsonl to detect any timing mismatches.\n")

# Find latest session
sessions_dir = Path("data/sessions")
latest_session = max([d for d in sessions_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime)

print(f"Session: {latest_session.name}")
print(f"Location: {latest_session}\n")

# Load original frames
original_frames = []
frames_file = latest_session / "frames.jsonl"
with open(frames_file, 'r') as f:
    for line in f:
        original_frames.append(json.loads(line))

print(f"Original analysis: {len(original_frames)} frames")

# Load saved audio
audio_file = latest_session / "audio.wav"
audio, sample_rate = sf.read(audio_file)

print(f"Saved audio: {len(audio)} samples @ {sample_rate}Hz")
print(f"Duration: {len(audio) / sample_rate:.2f}s\n")

print("=" * 80)
print("RE-RUNNING DIARIZATION ON SAVED AUDIO")
print("=" * 80)

# Try to re-run diarization (optional, might not work in all environments)
new_frames = []

print("\nAttempting to re-run diart on saved audio...")
print("(This is optional - will proceed with timing analysis if this fails)\n")

try:
    from diart import SpeakerDiarization
    from diart.sources import AudioSource
    from diart.inference import StreamingInference

    # Create audio source from saved file
    class FileAudioSource(AudioSource):
        """Audio source that reads from a file in chunks"""
        def __init__(self, file_path, sample_rate=16000, chunk_duration=5.0):
            self.audio, self.sample_rate = sf.read(file_path)
            self.chunk_size = int(chunk_duration * sample_rate)
            self.position = 0

        def read(self):
            """Read next chunk"""
            if self.position >= len(self.audio):
                return None

            end_position = min(self.position + self.chunk_size, len(self.audio))
            chunk = self.audio[self.position:end_position]
            self.position = end_position

            return chunk

    # Collect new frames
    frame_counter = [0]  # Use list to avoid nonlocal issues

    # Initialize diarization pipeline
    pipeline = SpeakerDiarization()

    # Create source from saved file
    source = FileAudioSource(audio_file, sample_rate=sample_rate)

    def collect_frame(prediction):
        # Extract speaker prediction
        # prediction is an Annotation object from pyannote
        if prediction and len(prediction.labels()) > 0:
            # Get the most prominent speaker at this time
            speaker = max(prediction.labels(),
                         key=lambda s: prediction.label_duration(s))

            frame = {
                "frame_id": f"retest_{frame_counter[0]:04d}",
                "timestamp": frame_counter[0] * 0.5,  # Assuming 500ms steps
                "speaker_prediction": speaker,
                "event_type": "speaker_active"
            }
            new_frames.append(frame)
            frame_counter[0] += 1
            print(f"\r  Processed {frame_counter[0]} frames", end="", flush=True)

    # Run inference
    inference = StreamingInference(pipeline, source, batch_size=1)
    inference.attach_hooks(
        lambda prediction: collect_frame(prediction)
    )

    # This will process the audio
    try:
        for prediction in inference:
            pass
    except StopIteration:
        pass

    print(f"\n\nRe-analysis generated: {len(new_frames)} frames\n")

except Exception as e:
    print(f"Could not re-run diart: {e}")
    print("\nSkipping re-analysis, proceeding with timing analysis...\n")
    new_frames = []

print("=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

if not new_frames:
    print("\nCould not re-run diarization (pipeline issues).")
    print("However, we can still analyze the original frames:\n")

    # Analyze original frames for timing consistency
    print("Original Frame Analysis:")
    print(f"  Total frames: {len(original_frames)}")
    print(f"  Time span: {original_frames[0]['timestamp']:.2f}s - {original_frames[-1]['timestamp']:.2f}s")
    print(f"  Audio duration: {len(audio) / sample_rate:.2f}s")

    # Check if timestamps align with audio length
    last_timestamp = original_frames[-1]['timestamp']
    audio_duration = len(audio) / sample_rate
    time_diff = abs(last_timestamp - audio_duration)

    print(f"\n  Timing check:")
    print(f"    Last frame timestamp: {last_timestamp:.2f}s")
    print(f"    Audio duration: {audio_duration:.2f}s")
    print(f"    Difference: {time_diff:.2f}s")

    if time_diff < 1.0:
        print("    [OK] Timestamps align with audio duration")
    else:
        print("    [WARNING] Significant time difference - possible desync!")

    # Check sample positions
    print(f"\n  Sample position check:")
    for i in [0, len(original_frames)//2, -1]:
        frame = original_frames[i]
        expected_sample = int(frame['timestamp'] * sample_rate)
        actual_sample = frame['sample_position']
        sample_diff = abs(expected_sample - actual_sample)

        print(f"    Frame {i}: timestamp={frame['timestamp']:.2f}s")
        print(f"      Expected sample: {expected_sample}")
        print(f"      Actual sample: {actual_sample}")
        print(f"      Difference: {sample_diff} samples ({sample_diff/sample_rate:.3f}s)")

        if sample_diff > sample_rate * 0.1:  # >100ms difference
            print(f"      [WARNING] Sample position mismatch!")
        else:
            print(f"      [OK]")

else:
    # Compare original vs new frames
    print(f"\nOriginal frames: {len(original_frames)}")
    print(f"Re-analysis frames: {len(new_frames)}")
    print(f"Difference: {abs(len(original_frames) - len(new_frames))} frames")

    if abs(len(original_frames) - len(new_frames)) > 2:
        print("\n[WARNING] Significant difference in frame count!")
        print("This suggests possible desynchronization.\n")
    else:
        print("\n[OK] Frame counts are similar.\n")

    # Compare timestamps
    print("Timestamp comparison (first 5 frames):")
    print(f"{'Frame':<8} {'Original':<12} {'Re-analysis':<12} {'Diff (s)':<10}")
    print("-" * 50)

    for i in range(min(5, len(original_frames), len(new_frames))):
        orig_time = original_frames[i]['timestamp']
        new_time = new_frames[i]['timestamp']
        diff = abs(orig_time - new_time)

        status = "[OK]" if diff < 0.1 else "[WARN]"
        print(f"{i:<8} {orig_time:<12.2f} {new_time:<12.2f} {diff:<10.3f} {status}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("""
Potential causes of short utterances:

1. Desynchronization (timing mismatch):
   - Audio saved doesn't match what was analyzed
   - Sample positions are incorrect
   - Check: Compare timestamps and sample positions above

2. Diart behavior (inherent to algorithm):
   - Diart uses 500ms sliding windows
   - Speaker changes detected even for brief pauses
   - Creates many short segments naturally
   - Check: If re-analysis produces same pattern, it's inherent

3. Merging threshold too strict:
   - Current threshold: 1.0s
   - May not merge brief pauses in same speaker's speech
   - Check: Test with higher threshold (1.5s, 2.0s)

4. Recording quality:
   - Brief utterances might actually be real
   - Background noise, interruptions, short responses
   - Check: Listen to audio file manually

Recommendations:
- If desynchronized: Fix audio capture/frame saving logic
- If diart behavior: Implement smart merging (append short to adjacent)
- If threshold issue: Increase overlap_threshold
- If real data: Filter by minimum duration (1.0-1.5s) before Whisper
""")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print("""
Based on the test results above:

1. If timing is OK:
   - Problem is likely diart's inherent behavior or merging threshold
   - Solution: Implement smart merging (append short utterances to adjacent ones)
   - Add minimum duration filter before feeding to Whisper

2. If timing is off:
   - Problem is desynchronization between audio and frames
   - Solution: Fix the audio capture and frame saving in stream_state.py
   - Ensure sample_position in frames matches actual audio position

3. Test smart merging:
   - Create new UtteranceBuilder.merge_short_utterances() method
   - Append utterances <1.0s to adjacent utterances
   - Re-test Whisper transcription quality
""")
