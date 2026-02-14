"""
Inspect .pt datasets for debugging.

This tool helps you understand what's inside the hybrid dataset files,
showing tensor shapes, statistics, and sample data.

Usage:
    python inspect_dataset.py data/training/stage0_hybrid.pt
    python inspect_dataset.py data/training/stage0_hybrid.pt --num-samples 20
"""

import torch
import argparse
import random


def inspect_dataset(pt_file, num_samples=10):
    """Load and inspect .pt dataset"""
    print(f"Loading {pt_file}...")
    data = torch.load(pt_file)

    # Overall stats
    stats = data['stats']
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")
    print(f"  Total samples: {stats['num_samples']}")
    print(f"  Positive (has_sound=true): {int(stats['num_positive'])}")
    print(f"  Negative (has_sound=false): {int(stats['num_negative'])}")
    print(f"  Ratio (pos:neg): {stats['num_positive']}/{stats['num_negative']} = {stats['num_positive']/stats['num_negative']:.2f}:1")
    print(f"  Token vocab size: {stats['token_vocab_size']}")
    print(f"  Max sequence length: {stats['max_seq_len']}")
    print(f"  Acoustic feature dim: {stats['acoustic_feature_dim']}")

    # Shapes
    print(f"\n{'='*70}")
    print("TENSOR SHAPES")
    print(f"{'='*70}")
    print(f"  Tokens: {data['tokens'].shape}")
    print(f"  Acoustic: {data['acoustic'].shape}")
    print(f"  Labels: {data['labels'].shape}")
    print(f"  Metadata: {len(data['metadata'])} entries")

    # Acoustic feature statistics
    print(f"\n{'='*70}")
    print("ACOUSTIC FEATURE STATISTICS")
    print(f"{'='*70}")

    feature_names = [
        'rms',
        'max_amplitude',
        'energy',
        'zero_crossing_rate',
        'spectral_centroid',
        'token_entropy',
        'unique_token_ratio',
        'token_concentration'
    ]

    for i, feature_name in enumerate(feature_names):
        feature_vals = data['acoustic'][:, i]

        # Separate by class
        pos_mask = data['labels'].squeeze() > 0.5
        neg_mask = ~pos_mask

        pos_vals = feature_vals[pos_mask]
        neg_vals = feature_vals[neg_mask]

        print(f"\n  {feature_name}:")
        print(f"    Overall: [{feature_vals.min():.4f}, {feature_vals.max():.4f}] (mean: {feature_vals.mean():.4f}, std: {feature_vals.std():.4f})")

        if len(pos_vals) > 0:
            print(f"    Positive: mean={pos_vals.mean():.4f}, std={pos_vals.std():.4f}")

        if len(neg_vals) > 0:
            print(f"    Negative: mean={neg_vals.mean():.4f}, std={neg_vals.std():.4f}")

    # Token statistics
    print(f"\n{'='*70}")
    print("TOKEN STATISTICS")
    print(f"{'='*70}")

    # Count non-padding tokens per sample
    non_padding = (data['tokens'] != 0).sum(dim=1)

    print(f"  Average non-padding tokens: {non_padding.float().mean():.1f}")
    print(f"  Min non-padding tokens: {non_padding.min().item()}")
    print(f"  Max non-padding tokens: {non_padding.max().item()}")
    print(f"  Padding overhead: {(1 - non_padding.float().mean() / stats['max_seq_len']):.1%}")

    # Random samples
    print(f"\n{'='*70}")
    print(f"RANDOM SAMPLES ({num_samples})")
    print(f"{'='*70}")

    indices = random.sample(range(len(data['labels'])), min(num_samples, len(data['labels'])))

    for idx_num, i in enumerate(indices):
        has_sound = bool(data['labels'][i].item())

        print(f"\n[Sample {idx_num+1}/{num_samples}] (Index: {i})")
        print(f"  Label: has_sound={has_sound}")

        # Token info
        tokens = data['tokens'][i]
        non_pad_tokens = tokens[tokens != 0]
        print(f"  Tokens: {non_pad_tokens[:10].tolist()}... (total: {len(non_pad_tokens)})")

        # Acoustic features
        print(f"  Acoustic features:")
        acoustic = data['acoustic'][i]
        for j, feature_name in enumerate(feature_names):
            print(f"    {feature_name:25s}: {acoustic[j]:.4f}")

        # Metadata
        meta = data['metadata'][i]
        print(f"  Metadata:")
        transcription = meta.get('transcription', '')
        if len(transcription) > 60:
            transcription = transcription[:57] + "..."
        print(f"    Transcription: '{transcription}'")
        print(f"    Speaker: {meta.get('speaker_id', 'unknown')}")
        print(f"    Duration: {meta.get('duration', 0):.2f}s")
        print(f"    Source: {meta.get('source', 'unknown')}")


def compare_classes(pt_file):
    """Compare acoustic features between positive and negative classes"""
    print(f"Loading {pt_file}...")
    data = torch.load(pt_file)

    print(f"\n{'='*70}")
    print("CLASS COMPARISON: POSITIVE vs NEGATIVE")
    print(f"{'='*70}")

    feature_names = [
        'rms',
        'max_amplitude',
        'energy',
        'zero_crossing_rate',
        'spectral_centroid',
        'token_entropy',
        'unique_token_ratio',
        'token_concentration'
    ]

    pos_mask = data['labels'].squeeze() > 0.5
    neg_mask = ~pos_mask

    print(f"\nPositive samples: {pos_mask.sum().item()}")
    print(f"Negative samples: {neg_mask.sum().item()}")

    print(f"\n{'Feature':<25} {'Pos Mean':>12} {'Neg Mean':>12} {'Difference':>12} {'Ratio':>10}")
    print("-" * 75)

    for i, feature_name in enumerate(feature_names):
        pos_vals = data['acoustic'][pos_mask, i]
        neg_vals = data['acoustic'][neg_mask, i]

        pos_mean = pos_vals.mean().item()
        neg_mean = neg_vals.mean().item()
        diff = pos_mean - neg_mean

        # Avoid division by zero
        if neg_mean != 0:
            ratio = pos_mean / neg_mean
        else:
            ratio = float('inf') if pos_mean > 0 else 1.0

        print(f"{feature_name:<25} {pos_mean:>12.4f} {neg_mean:>12.4f} {diff:>12.4f} {ratio:>10.2f}x")

    print("\nInterpretation:")
    print("  - Higher ratio (>1.5x) = feature strongly discriminates has_sound=true")
    print("  - Lower ratio (<0.7x) = feature strongly discriminates has_sound=false")
    print("  - Near 1.0 = feature not very discriminative")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect .pt dataset")
    parser.add_argument("pt_file", help="Path to .pt dataset file")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to show")
    parser.add_argument("--compare", action="store_true", help="Compare positive vs negative classes")
    args = parser.parse_args()

    if args.compare:
        compare_classes(args.pt_file)
    else:
        inspect_dataset(args.pt_file, args.num_samples)
