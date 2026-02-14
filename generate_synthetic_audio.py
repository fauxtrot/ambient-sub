"""
Generate synthetic silence samples for diffusion model training.

This creates NEGATIVE examples (has_sound=false) to balance the dataset.
The enrichment data has 25 positive examples (has_sound=true), so we need
negative examples to avoid class imbalance.

For first-time ML builders:
    - Class imbalance = when one label dominates (e.g., 100% positive)
    - Models trained on imbalanced data just predict the majority class
    - Solution: generate synthetic negative examples (silence, low noise)
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import json
import argparse


def generate_silence_samples(
    output_dir: str,
    num_samples: int = 250,
    duration_range: tuple = (1.0, 5.0),
    sample_rate: int = 16000,
    silence_ratio: float = 0.7
):
    """
    Generate silence and low-amplitude noise samples.

    Why we need this:
        - Existing enrichment data (25 samples) are all speech (has_sound=true)
        - Model needs to learn what "no sound" looks like
        - Pure silence teaches clear negative examples
        - Low-amplitude noise teaches "ignore background hum"

    Args:
        output_dir: Where to save WAV files
        num_samples: Number of silence samples to create (default: 250)
        duration_range: Random duration range in seconds (default: 1.0-5.0s)
        sample_rate: Audio sample rate (default: 16000 Hz)
        silence_ratio: Fraction of pure silence vs low noise (default: 0.7 = 70% silence)

    Returns:
        List of sample metadata dictionaries
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_metadata = []

    print(f"Generating {num_samples} silence samples...")
    print(f"  Output: {output_dir}")
    print(f"  Duration range: {duration_range[0]:.1f}s - {duration_range[1]:.1f}s")
    print(f"  {int(silence_ratio * 100)}% pure silence, {int((1-silence_ratio) * 100)}% low noise")
    print()

    for i in range(num_samples):
        # Random duration between 1-5 seconds
        duration = np.random.uniform(*duration_range)
        num_samples_audio = int(duration * sample_rate)

        # 70% pure silence, 30% low-amplitude noise
        if np.random.random() < silence_ratio:
            # Pure silence (all zeros)
            audio = np.zeros(num_samples_audio, dtype=np.float32)
            sample_type = "silence"
        else:
            # Low-amplitude noise (random values, very quiet)
            # Standard normal distribution (mean=0, std=1)
            audio = np.random.randn(num_samples_audio).astype(np.float32)

            # Scale down to be very quiet (below detection threshold)
            # Threshold is typically ~0.01, so we use 0.005 (half the threshold)
            audio = audio * 0.005
            sample_type = "low_noise"

        # Save WAV file
        file_path = output_dir / f"silence_{i:04d}.wav"
        sf.write(file_path, audio, sample_rate)

        # Metadata for this sample
        samples_metadata.append({
            "file_path": str(file_path),
            "has_sound": False,  # Ground truth label
            "duration": float(duration),
            "type": sample_type,
            "sample_rate": sample_rate
        })

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    # Save metadata JSON
    metadata_file = output_dir / "silence_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(samples_metadata, f, indent=2)

    print(f"\n[SUCCESS] Generated {num_samples} silence samples")
    print(f"  Files: {output_dir}/silence_0000.wav - silence_{num_samples-1:04d}.wav")
    print(f"  Metadata: {metadata_file}")
    print(f"\nBreakdown:")
    print(f"  Pure silence: {sum(1 for s in samples_metadata if s['type'] == 'silence')}")
    print(f"  Low noise: {sum(1 for s in samples_metadata if s['type'] == 'low_noise')}")
    print(f"  Total duration: {sum(s['duration'] for s in samples_metadata):.1f}s")

    return samples_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic silence samples for training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/silence",
        help="Output directory for silence samples"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=250,
        help="Number of silence samples to generate"
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=1.0,
        help="Minimum duration in seconds"
    )
    parser.add_argument(
        "--duration-max",
        type=float,
        default=5.0,
        help="Maximum duration in seconds"
    )
    parser.add_argument(
        "--silence-ratio",
        type=float,
        default=0.7,
        help="Ratio of pure silence to low noise (0.0-1.0)"
    )

    args = parser.parse_args()

    # Generate samples
    generate_silence_samples(
        output_dir=args.output,
        num_samples=args.count,
        duration_range=(args.duration_min, args.duration_max),
        silence_ratio=args.silence_ratio
    )


if __name__ == "__main__":
    main()
