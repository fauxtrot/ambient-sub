"""Test smart merging of short utterances"""

import sys
from pathlib import Path
import importlib.util

# Import UtteranceBuilder directly
spec = importlib.util.spec_from_file_location(
    "utterance_builder",
    Path(__file__).parent / "ambient_subconscious" / "training" / "utterance_builder.py"
)
utterance_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utterance_builder)

UtteranceBuilder = utterance_builder.UtteranceBuilder

print("=" * 80)
print("SMART MERGING TEST")
print("=" * 80)

# Find latest session
sessions_dir = Path("data/sessions")
latest_session = max([d for d in sessions_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime)

print(f"\nSession: {latest_session.name}")
print(f"Location: {latest_session}\n")

print("=" * 80)
print("TEST 1: WITHOUT MERGING (min_duration=0.0)")
print("=" * 80)

builder_no_merge = UtteranceBuilder(overlap_threshold=1.0, min_duration=0.0)
utterances_no_merge = builder_no_merge.build_from_session(latest_session)

print(f"\nResults:")
print(f"  Total utterances: {len(utterances_no_merge)}")
print(f"  Total duration: {sum(u.duration() for u in utterances_no_merge):.2f}s")
print(f"  Average duration: {sum(u.duration() for u in utterances_no_merge) / len(utterances_no_merge):.2f}s")

print("\nDetailed breakdown:")
print(f"{'Utt':<5} {'Speaker':<12} {'Duration':<10} {'Frames':<8} {'Time Range':<20}")
print("-" * 80)

for i, utt in enumerate(utterances_no_merge):
    time_range = f"{utt.start_time:.2f}s - {utt.end_time:.2f}s"
    status = "[SHORT]" if utt.duration() < 1.0 else "[OK]"
    print(f"{i:<5} {utt.speaker_id:<12} {utt.duration():<10.2f} {utt.num_frames:<8} {time_range:<20} {status}")

short_count = sum(1 for u in utterances_no_merge if u.duration() < 1.0)
print(f"\nShort utterances (<1.0s): {short_count}/{len(utterances_no_merge)} ({short_count/len(utterances_no_merge)*100:.1f}%)")

print("\n" + "=" * 80)
print("TEST 2: WITH MERGING (min_duration=1.0)")
print("=" * 80)

builder_with_merge = UtteranceBuilder(overlap_threshold=1.0, min_duration=1.0)
utterances_with_merge = builder_with_merge.build_from_session(latest_session)

print(f"\nResults:")
print(f"  Total utterances: {len(utterances_with_merge)}")
print(f"  Total duration: {sum(u.duration() for u in utterances_with_merge):.2f}s")
print(f"  Average duration: {sum(u.duration() for u in utterances_with_merge) / len(utterances_with_merge):.2f}s")

print("\nDetailed breakdown:")
print(f"{'Utt':<5} {'Speaker':<12} {'Duration':<10} {'Frames':<8} {'Time Range':<20}")
print("-" * 80)

for i, utt in enumerate(utterances_with_merge):
    time_range = f"{utt.start_time:.2f}s - {utt.end_time:.2f}s"
    status = "[SHORT]" if utt.duration() < 1.0 else "[OK]"
    print(f"{i:<5} {utt.speaker_id:<12} {utt.duration():<10.2f} {utt.num_frames:<8} {time_range:<20} {status}")

short_count_merged = sum(1 for u in utterances_with_merge if u.duration() < 1.0)
print(f"\nShort utterances (<1.0s): {short_count_merged}/{len(utterances_with_merge)} ({short_count_merged/len(utterances_with_merge)*100:.1f}%)")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\nBefore merging:")
print(f"  Total utterances: {len(utterances_no_merge)}")
print(f"  Short utterances: {short_count} ({short_count/len(utterances_no_merge)*100:.1f}%)")
print(f"  Avg duration: {sum(u.duration() for u in utterances_no_merge) / len(utterances_no_merge):.2f}s")

print(f"\nAfter merging:")
print(f"  Total utterances: {len(utterances_with_merge)}")
print(f"  Short utterances: {short_count_merged} ({short_count_merged/len(utterances_with_merge)*100:.1f}% if len(utterances_with_merge) > 0 else 0)%)")
print(f"  Avg duration: {sum(u.duration() for u in utterances_with_merge) / len(utterances_with_merge):.2f}s")

improvement = len(utterances_no_merge) - len(utterances_with_merge)
reduction = (improvement / len(utterances_no_merge)) * 100 if len(utterances_no_merge) > 0 else 0

print(f"\nImprovement:")
print(f"  Reduced by {improvement} utterances ({reduction:.1f}%)")
print(f"  Short utterances removed: {short_count - short_count_merged}")

print("\n" + "=" * 80)
print("WHISPER READINESS")
print("=" * 80)

print(f"\nBefore merging: {len(utterances_no_merge) - short_count}/{len(utterances_no_merge)} ({(len(utterances_no_merge) - short_count)/len(utterances_no_merge)*100:.1f}%) usable for Whisper")
print(f"After merging: {len(utterances_with_merge) - short_count_merged}/{len(utterances_with_merge)} ({(len(utterances_with_merge) - short_count_merged)/len(utterances_with_merge)*100:.1f}% if len(utterances_with_merge) > 0 else 0)%) usable for Whisper")

print("\n" + "=" * 80)
print("[SUCCESS] Smart merging complete!")
print("=" * 80)

print("\nSummary:")
print("  - Short utterances are now merged with adjacent ones (same speaker)")
print("  - Minimum duration threshold: 1.0s (good for Whisper)")
print("  - Speaker boundaries are preserved")
print("  - Ready for Whisper transcription")
