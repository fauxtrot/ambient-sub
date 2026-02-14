"""Test different overlap thresholds to find optimal utterance grouping"""

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

# Find latest session
sessions_dir = Path("data/sessions")
latest_session = max([d for d in sessions_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime)

print("=" * 70)
print("TESTING DIFFERENT OVERLAP THRESHOLDS")
print("=" * 70)
print(f"\nSession: {latest_session.name}\n")

# Test different thresholds
thresholds = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

for threshold in thresholds:
    builder = UtteranceBuilder(overlap_threshold=threshold)
    utterances = builder.build_from_session(latest_session)

    print(f"\nThreshold: {threshold:.2f}s")
    print(f"  Utterances: {len(utterances)}")
    print(f"  Avg duration: {sum(u.duration() for u in utterances) / len(utterances):.2f}s")
    print(f"  Avg frames/utterance: {sum(u.num_frames for u in utterances) / len(utterances):.1f}")

    # Show first few utterances
    for i, utt in enumerate(utterances[:3]):
        print(f"    [{i}] {utt.start_time:.2f}s - {utt.end_time:.2f}s "
              f"({utt.duration():.2f}s, {utt.num_frames} frames)")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("\nBased on diart's 500ms sliding window:")
print("  - Use threshold >= 0.75s to group overlapping segments")
print("  - Use threshold ~1.0-1.5s for continuous speech")
print("  - Default: 1.0s provides good balance")
