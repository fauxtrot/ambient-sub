"""Demonstrate how Whisper enriches utterances for training data"""

import json
from pathlib import Path
import whisper
import sys
import importlib.util

# Import UtteranceBuilder
spec = importlib.util.spec_from_file_location(
    "utterance_builder",
    Path(__file__).parent / "ambient_subconscious" / "training" / "utterance_builder.py"
)
utterance_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utterance_builder)

load_utterances = utterance_builder.load_utterances

print("=" * 80)
print("WHISPER ENRICHMENT PIPELINE DEMO")
print("=" * 80)

# Find latest session
sessions_dir = Path("data/sessions")
latest_session = max([d for d in sessions_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime)

utterances_dir = latest_session / "utterances"

print(f"\nSession: {latest_session.name}")
print(f"Utterances directory: {utterances_dir}")

if not utterances_dir.exists():
    print("\nNo utterances found. Run test_utterance_verification.py first!")
    sys.exit(1)

# Load utterances
print("\nLoading utterances...")
utterances = load_utterances(utterances_dir)
print(f"  Loaded {len(utterances)} utterances")

# Load Whisper model
print("\nLoading Whisper model (base)...")
whisper_model = whisper.load_model("base")

# Create enriched training data
print("\n" + "=" * 80)
print("ENRICHING UTTERANCES WITH WHISPER TRANSCRIPTIONS")
print("=" * 80)

training_samples = []

for i, utt in enumerate(utterances):
    print(f"\n[Utterance {i}]")
    print(f"  Input:")
    print(f"    Audio: utt_{i:04d}.wav")
    print(f"    Speaker: {utt.speaker_id} (from diart)")
    print(f"    Time: {utt.start_time:.2f}s - {utt.end_time:.2f}s")
    print(f"    Duration: {utt.duration():.2f}s")
    print(f"    Frames merged: {utt.num_frames}")

    # Transcribe with Whisper
    result = whisper_model.transcribe(utt.audio.astype('float32'), fp16=False)
    transcription = result['text'].strip()

    # Create training sample
    sample = {
        "utterance_id": f"utt_{i:04d}",
        "session_id": latest_session.name,

        # From UtteranceBuilder
        "speaker_id": utt.speaker_id,
        "start_time": utt.start_time,
        "end_time": utt.end_time,
        "duration": utt.duration(),
        "confidence": utt.confidence,
        "num_frames": utt.num_frames,

        # From Whisper
        "transcription": transcription,
        "language": result.get('language', 'unknown'),

        # Audio features (for Stage 0: has_sound)
        "has_sound": utt.duration() > 0.1,  # Simple threshold
        "audio_path": f"utt_{i:04d}.wav"
    }

    training_samples.append(sample)

    print(f"  Output (enriched):")
    print(f"    Transcription: \"{transcription}\" (from Whisper)")
    print(f"    Language: {sample['language']}")
    print(f"    has_sound: {sample['has_sound']} (for Stage 0 training)")

# Save enriched training data
output_file = utterances_dir / "training_samples.json"
with open(output_file, 'w') as f:
    json.dump(training_samples, f, indent=2)

print("\n" + "=" * 80)
print("TRAINING DATA SUMMARY")
print("=" * 80)

print(f"\nCreated {len(training_samples)} training samples")
print(f"Saved to: {output_file}")

print("\nWhat each sample contains:")
print("  [From Diart]       speaker_id, timing, confidence")
print("  [From Whisper]     transcription, language")
print("  [From Audio]       has_sound, duration, audio_path")

print("\n" + "=" * 80)
print("HOW THIS FEEDS THE AUDIO EXPERT")
print("=" * 80)

print("""
Stage 0 (has_sound detection):
  Input:  Audio features (amplitude, spectral)
  Labels: has_sound (True/False)
  Source: Audio analysis (duration > threshold)

Stage 1 (speaker awareness):
  Input:  Audio features + speaker embeddings
  Labels: speaker_id, speaker_changed
  Source: Diart diarization

Stage 2-3 (speaker identity + temporal):
  Input:  Audio + temporal context
  Labels: speaker_id, timing, duration
  Source: Diart + UtteranceBuilder

Stage 4 (rich recognition - what was said):
  Input:  Audio + full context
  Labels: transcription, language, intent
  Source: Whisper + future NLU models

This is the complete training pipeline:
  Raw session → UtteranceBuilder → Whisper → Training samples → Audio Expert
""")

print("\n" + "=" * 80)
print("EXAMPLE TRAINING SAMPLE")
print("=" * 80)

if training_samples:
    # Show the best example (longest utterance)
    best_sample = max(training_samples, key=lambda s: s['duration'])
    print("\nBest example (longest utterance):")
    print(json.dumps(best_sample, indent=2))

print("\n" + "=" * 80)
print("[SUCCESS] Whisper enrichment complete!")
print("=" * 80)
print("\nNext steps:")
print("  1. Use training_samples.json to train Audio Expert (Stage 0)")
print("  2. Model learns has_sound detection from these labeled examples")
print("  3. Progress to Stage 1 with speaker_id labels")
print("  4. Eventually use transcriptions for Stage 4 (semantic understanding)")
