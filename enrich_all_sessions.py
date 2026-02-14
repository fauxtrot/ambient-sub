"""
Batch enrichment script: Process all sessions with VAD + Whisper + Diart pipeline.

This generates training data for the audio expert.
"""

import json
from pathlib import Path
import whisper
from ambient_subconscious.training.label_pipeline import LabelPipeline

print("=" * 80)
print("BATCH SESSION ENRICHMENT")
print("=" * 80)

# Find all sessions
sessions_dir = Path("data/sessions")
session_paths = sorted([d for d in sessions_dir.iterdir() if d.is_dir()],
                      key=lambda d: d.stat().st_mtime)

print(f"\nFound {len(session_paths)} sessions\n")

# Load Whisper model once (reuse across sessions)
print("Loading Whisper model (base)...")
whisper_model = whisper.load_model("base")

# Create pipeline
pipeline = LabelPipeline(whisper_model=whisper_model)

# Process each session
all_samples = []
session_stats = []

for i, session_path in enumerate(session_paths):
    print(f"\n{'-' * 80}")
    print(f"[{i+1}/{len(session_paths)}] Processing: {session_path.name}")
    print(f"{'-' * 80}")

    try:
        # Generate enriched labels
        ground_truth = pipeline.generate_ground_truth(session_path)

        # Convert to training samples
        samples = ground_truth.to_list()

        print(f"\nEnriched samples: {len(samples)}")

        # Statistics
        with_speaker = sum(1 for s in samples if s['detected_by_diart'])
        gaps = sum(1 for s in samples if s['gap_from_diart'])

        print(f"  With speaker attribution: {with_speaker}")
        print(f"  Gaps (no diart coverage): {gaps}")

        # Show example
        if samples:
            best = max(samples, key=lambda s: len(s.get('transcription', '')))
            print(f"\nBest example:")
            print(f"  Text: \"{best.get('transcription', 'N/A')[:80]}...\"")
            print(f"  Speaker: {best.get('speaker_id', 'unknown')}")
            print(f"  Confidence: {best.get('speaker_confidence', 0):.2f}")

        # Save session enrichment
        output_file = session_path / "enriched_training_data.json"
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)

        print(f"\nSaved to: {output_file}")

        # Accumulate
        all_samples.extend(samples)

        session_stats.append({
            "session_id": session_path.name,
            "samples": len(samples),
            "with_speaker": with_speaker,
            "gaps": gaps,
            "coverage": with_speaker / len(samples) if len(samples) > 0 else 0,
        })

    except Exception as e:
        print(f"\n[ERROR] Failed to process {session_path.name}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "=" * 80)
print("BATCH ENRICHMENT COMPLETE")
print("=" * 80)

print(f"\nTotal sessions processed: {len(session_stats)}")
print(f"Total training samples: {len(all_samples)}")

if all_samples:
    total_with_speaker = sum(s['detected_by_diart'] for s in all_samples)
    total_gaps = sum(s['gap_from_diart'] for s in all_samples)

    print(f"\nOverall statistics:")
    print(f"  Samples with speaker: {total_with_speaker}/{len(all_samples)} ({total_with_speaker/len(all_samples)*100:.1f}%)")
    print(f"  Gaps (no diart): {total_gaps}/{len(all_samples)} ({total_gaps/len(all_samples)*100:.1f}%)")

    # Save combined dataset
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_file = output_dir / "all_sessions_enriched.json"
    with open(combined_file, 'w') as f:
        json.dump(all_samples, f, indent=2)

    stats_file = output_dir / "enrichment_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(session_stats, f, indent=2)

    print(f"\nSaved combined dataset:")
    print(f"  Training data: {combined_file}")
    print(f"  Statistics: {stats_file}")

    print("\n" + "=" * 80)
    print("TRAINING DATA READY")
    print("=" * 80)

    print("""
This enriched dataset provides:
  [OK] Complete speech coverage (VAD + Whisper)
  [OK] Speaker attribution where available (diart)
  [OK] Gap analysis (quality metrics)
  [OK] Ready for Audio Expert training

Next steps:
  1. Use all_sessions_enriched.json for training
  2. Train Audio Expert Stage 0 (has_sound detection)
  3. Progress to Stage 1 (speaker awareness)
  4. Eventually Stage 4 (full transcription understanding)

Sample format:
  {
    "segment_id": "vad_0000",
    "session_id": "20260131_...",
    "start_time": 0.0,
    "end_time": 6.52,
    "duration": 6.52,
    "has_sound": true,
    "has_speaker": true,
    "speaker_id": "speaker0",
    "speaker_confidence": 0.85,
    "transcription": "...",
    "language": "en",
    "detected_by_diart": true,
    "gap_from_diart": false
  }
""")
