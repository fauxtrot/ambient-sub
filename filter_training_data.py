"""
Filter enriched training data to remove Whisper hallucinations.

This removes common garbage transcriptions and very short segments.
"""

import json
from pathlib import Path
import argparse

# Common Whisper hallucinations on noise
HALLUCINATION_PHRASES = {
    "",
    " ",
    "yeah",
    "yeah.",
    "uh-huh",
    "uh-huh.",
    "mm-hmm",
    "mm-hmm.",
    "thank you",
    "thank you.",
    "thanks",
    "thanks.",
    "okay",
    "okay.",
    "oh",
    "oh.",
    "uh",
    "um",
    "hmm",
    "hm",
    "ah",
    "eh",
    "oh, here",
    "oh, here.",
    "you know",
    "you know.",
}


def is_hallucination(transcription, min_length=5):
    """
    Detect if a transcription is likely a Whisper hallucination.

    Args:
        transcription: Transcribed text
        min_length: Minimum character length for valid speech

    Returns:
        True if likely hallucination
    """
    if not transcription:
        return True

    # Normalize
    text = transcription.strip().lower()

    # Check against known hallucinations
    if text in HALLUCINATION_PHRASES:
        return True

    # Too short (likely just noise)
    if len(text) < min_length:
        return True

    # Just punctuation
    if text.replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip() == "":
        return True

    return False


def filter_training_data(
    input_file,
    output_file=None,
    min_duration=0.5,
    min_text_length=5,
    require_speaker=False
):
    """
    Filter enriched training data.

    Args:
        input_file: Path to enriched_training_data.json
        output_file: Output path (default: input_file with _filtered suffix)
        min_duration: Minimum segment duration in seconds
        min_text_length: Minimum transcription length in characters
        require_speaker: Only keep segments with speaker attribution

    Returns:
        Filtered data
    """
    input_path = Path(input_file)

    if output_file is None:
        output_file = input_path.parent / f"{input_path.stem}_filtered.json"

    # Load data
    with open(input_path) as f:
        data = json.load(f)

    print(f"Input: {len(data)} segments")

    # Filter
    filtered = []
    reasons = {
        "too_short": 0,
        "hallucination": 0,
        "no_speaker": 0,
        "kept": 0
    }

    for segment in data:
        # Check duration
        if segment["duration"] < min_duration:
            reasons["too_short"] += 1
            continue

        # Check hallucination
        if is_hallucination(segment.get("transcription", ""), min_text_length):
            reasons["hallucination"] += 1
            continue

        # Check speaker requirement
        if require_speaker and not segment.get("has_speaker", False):
            reasons["no_speaker"] += 1
            continue

        # Passed all filters
        filtered.append(segment)
        reasons["kept"] += 1

    # Save filtered data
    with open(output_file, 'w') as f:
        json.dump(filtered, f, indent=2)

    # Print statistics
    print(f"\nFiltered Results:")
    print(f"  Kept: {reasons['kept']} ({reasons['kept']/len(data)*100:.1f}%)")
    print(f"  Removed - too short (<{min_duration}s): {reasons['too_short']}")
    print(f"  Removed - hallucination: {reasons['hallucination']}")
    if require_speaker:
        print(f"  Removed - no speaker: {reasons['no_speaker']}")

    print(f"\nOutput: {output_file}")

    # Quality metrics
    if filtered:
        avg_duration = sum(s["duration"] for s in filtered) / len(filtered)
        with_speaker = sum(1 for s in filtered if s.get("has_speaker", False))

        print(f"\nFiltered Data Quality:")
        print(f"  Average duration: {avg_duration:.2f}s")
        print(f"  With speaker attribution: {with_speaker} ({with_speaker/len(filtered)*100:.1f}%)")

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter enriched training data"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input enriched_training_data.json file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: input_filtered.json)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum segment duration in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=5,
        help="Minimum transcription length (default: 5 characters)"
    )
    parser.add_argument(
        "--require-speaker",
        action="store_true",
        help="Only keep segments with speaker attribution"
    )

    args = parser.parse_args()

    filter_training_data(
        args.input_file,
        args.output,
        args.min_duration,
        args.min_text_length,
        args.require_speaker
    )


if __name__ == "__main__":
    main()
