"""Test the UtteranceBuilder on existing session data (standalone)"""

import sys
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import whisper

# Import the utterance_builder module directly
utterance_builder_path = Path(__file__).parent / "ambient_subconscious" / "training" / "utterance_builder.py"
spec = __import__('importlib.util').util.spec_from_file_location("utterance_builder", utterance_builder_path)
utterance_builder = __import__('importlib.util').util.module_from_spec(spec)
spec.loader.exec_module(utterance_builder)

UtteranceBuilder = utterance_builder.UtteranceBuilder

print("=" * 70)
print("UTTERANCE BUILDER TEST")
print("=" * 70)

# Find latest session
sessions_dir = Path("data/sessions")
session_dirs = sorted([d for d in sessions_dir.iterdir() if d.is_dir()],
                     key=lambda d: d.stat().st_mtime,
                     reverse=True)

if not session_dirs:
    print("No sessions found!")
    exit(1)

latest_session = session_dirs[0]
print(f"\nLatest session: {latest_session.name}")

# Build utterances
print("\nBuilding utterances from overlapping segments...")
builder = UtteranceBuilder(overlap_threshold=0.5)
utterances = builder.build_from_session(latest_session)

print(f"\nResults:")
print(f"  Total utterances: {len(utterances)}")

# Print summary
print("\nUtterance Summary:")
print(f"{'ID':<6} {'Speaker':<12} {'Start':<8} {'End':<8} {'Duration':<10} {'Frames':<8}")
print("-" * 70)

for i, utt in enumerate(utterances):
    print(f"{i:<6} {utt.speaker_id:<12} {utt.start_time:<8.2f} {utt.end_time:<8.2f} "
          f"{utt.duration():<10.2f} {utt.num_frames:<8}")

# Load Whisper model for verification
print("\n" + "=" * 70)
print("TRANSCRIPTION VERIFICATION")
print("=" * 70)
print("\nLoading Whisper model (base)...")
whisper_model = whisper.load_model("base")

# Transcribe each utterance
print("\nTranscribing utterances...")
for i, utt in enumerate(utterances):
    try:
        result = whisper_model.transcribe(utt.audio.astype('float32'), fp16=False)
        transcription = result['text'].strip()
        print(f"\n[Utterance {i}] Speaker {utt.speaker_id} ({utt.duration():.2f}s)")
        print(f"  \"{transcription}\"")
    except Exception as e:
        print(f"\n[Utterance {i}] Error: {e}")

# Save utterances
output_dir = latest_session / "utterances"
print(f"\n" + "=" * 70)
print(f"Saving utterances to {output_dir}")
print("=" * 70)

builder.save_utterances(utterances, output_dir, save_audio=True)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print(f"\nSession: {latest_session}")
print(f"Utterances: {len(utterances)}")
print(f"Output: {output_dir}")
