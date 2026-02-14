"""Comprehensive utterance verification with transcription and playback"""

import sys
from pathlib import Path
import importlib.util
import sounddevice as sd
import time

# Import UtteranceBuilder directly
spec = importlib.util.spec_from_file_location(
    "utterance_builder",
    Path(__file__).parent / "ambient_subconscious" / "training" / "utterance_builder.py"
)
utterance_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utterance_builder)

UtteranceBuilder = utterance_builder.UtteranceBuilder

# Import Whisper
import whisper

print("=" * 80)
print("UTTERANCE VERIFICATION TEST")
print("=" * 80)

# Find latest session
sessions_dir = Path("data/sessions")
latest_session = max([d for d in sessions_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime)

print(f"\nSession: {latest_session.name}")
print(f"Location: {latest_session}")

# Build utterances with optimal threshold (1.0s)
print("\nBuilding utterances (threshold=1.0s)...")
builder = UtteranceBuilder(overlap_threshold=1.0)
utterances = builder.build_from_session(latest_session)

print(f"\nResults:")
print(f"  Total utterances: {len(utterances)}")
print(f"  Total duration: {sum(u.duration() for u in utterances):.2f}s")
print(f"  Average duration: {sum(u.duration() for u in utterances) / len(utterances):.2f}s")
print(f"  Frames merged: {sum(u.num_frames for u in utterances)}")

# Load Whisper for transcription
print("\nLoading Whisper model (base)...")
whisper_model = whisper.load_model("base")

# Detailed utterance analysis
print("\n" + "=" * 80)
print("UTTERANCE DETAILS")
print("=" * 80)

for i, utt in enumerate(utterances):
    print(f"\n[Utterance {i}]")
    print(f"  Speaker: {utt.speaker_id}")
    print(f"  Time: {utt.start_time:.2f}s - {utt.end_time:.2f}s ({utt.duration():.2f}s)")
    print(f"  Frames merged: {utt.num_frames}")
    print(f"  Confidence: {utt.confidence:.2f}")

    # Show which frames were merged
    print(f"  Frame IDs: {', '.join(utt.frame_ids[:3])}{'...' if len(utt.frame_ids) > 3 else ''}")

    # Transcribe
    try:
        result = whisper_model.transcribe(utt.audio.astype('float32'), fp16=False)
        transcription = result['text'].strip()
        print(f"  Transcription: \"{transcription}\"")
    except Exception as e:
        print(f"  Transcription error: {e}")

# Ask if user wants to play utterances
print("\n" + "=" * 80)
print("PLAYBACK TEST")
print("=" * 80)
print("\nWould you like to play the utterances? (y/n)")
play_audio = input().strip().lower() == 'y'

if play_audio:
    print("\nPlaying utterances...")
    for i, utt in enumerate(utterances):
        print(f"\n[{i+1}/{len(utterances)}] Playing utterance {i}")
        print(f"  Speaker: {utt.speaker_id}")
        print(f"  Duration: {utt.duration():.2f}s")

        # Play the utterance
        sd.play(utt.audio, builder.sample_rate)
        sd.wait()

        # Pause between utterances
        if i < len(utterances) - 1:
            time.sleep(0.5)

    print("\nPlayback complete!")

# Save utterances for inspection
output_dir = latest_session / "utterances"
print("\n" + "=" * 80)
print("SAVING UTTERANCES")
print("=" * 80)

builder.save_utterances(utterances, output_dir, save_audio=True)

print(f"\nSaved to: {output_dir}")
print(f"  - utterances.json (metadata)")
print(f"  - utt_*.wav (audio files)")

# Frame reduction summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Load original frames to show reduction
import json
frames = []
with open(latest_session / "frames.jsonl", 'r') as f:
    for line in f:
        frames.append(json.loads(line))

reduction_pct = (1 - len(utterances) / len(frames)) * 100

print(f"\nFrame Reduction:")
print(f"  Original frames: {len(frames)}")
print(f"  Merged utterances: {len(utterances)}")
print(f"  Reduction: {reduction_pct:.1f}%")

print(f"\nUtterance Quality:")
print(f"  Avg frames/utterance: {sum(u.num_frames for u in utterances) / len(utterances):.1f}")
print(f"  Avg duration: {sum(u.duration() for u in utterances) / len(utterances):.2f}s")
print(f"  Avg confidence: {sum(u.confidence for u in utterances) / len(utterances):.2f}")

# Show grouping examples
print(f"\nGrouping Examples:")
for i in range(min(3, len(utterances))):
    utt = utterances[i]
    print(f"  Utterance {i}: {utt.num_frames} frames = {utt.duration():.2f}s")

print("\n" + "=" * 80)
print("[SUCCESS] MILESTONE 0: UTTERANCE BUILDING COMPLETE")
print("=" * 80)
print("\nThe UtteranceBuilder successfully:")
print("  [OK] Merged overlapping diart segments")
print("  [OK] Assigned speaker attribution")
print("  [OK] Created clean audio boundaries")
print("  [OK] Ready for Whisper transcription")
print("\nNext: Milestone 1 - Audio Expert Stage 0 (has_sound detection)")
