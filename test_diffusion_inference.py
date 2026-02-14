"""
Test the trained diffusion model on new audio samples.

This verifies that:
1. Model can load from checkpoint
2. Inference works correctly
3. Predictions make sense

For first-time ML builders:
    - Inference = using a trained model to make predictions
    - No training happens here - just testing
    - We want to see if the model learned the right patterns
"""

import torch
import soundfile as sf
from pathlib import Path
import argparse
import json

from ambient_subconscious.training.diffusion_model_stage0 import TemporalDiffusionStage0
from tokenize_training_data import AudioTokenizer


def load_model(checkpoint_path):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model.pt file

    Returns:
        Loaded model in eval mode
    """
    print("Loading model...")

    # Create model with same architecture as training
    model = TemporalDiffusionStage0(
        vocab_size=1024,
        hidden_dim=128,
        num_layers=2,
        max_seq_len=100,
        temporal_window=5,
        num_diffusion_steps=50,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Put in eval mode (disables dropout, etc.)
    model.eval()

    print(f"  Loaded from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val accuracy: {checkpoint['val_accuracy']:.3f}")

    return model


def predict_has_sound(model, wav_path, tokenizer, device='cpu'):
    """
    Predict whether audio has sound.

    Args:
        model: Trained diffusion model
        wav_path: Path to WAV file
        tokenizer: AudioTokenizer instance
        device: 'cpu' or 'cuda'

    Returns:
        prediction: Float probability (0-1)
        confidence: How confident (distance from 0.5)
    """
    model = model.to(device)
    model.eval()

    # Tokenize audio
    tokens, metadata = tokenizer.tokenize_wav(wav_path, use_first_codebook_only=True)

    # Pad/truncate to 100 tokens
    if len(tokens) > 100:
        tokens = tokens[:100]
    else:
        tokens = tokens + [0] * (100 - len(tokens))

    # Convert to tensor
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    # Random temporal context (no history for test)
    temporal_context = torch.rand(1, 5).to(device)

    # Run inference (denoising)
    with torch.no_grad():
        prediction = model.denoise_sample(
            tokens_tensor,
            temporal_context,
            num_steps=50  # Full denoising for best accuracy
        )

    probability = prediction.item()

    # Confidence = how far from uncertain (0.5)
    confidence = abs(probability - 0.5)

    return probability, confidence, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Test trained diffusion model on audio samples"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/diffusion_stage0/model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    print("="*80)
    print("TESTING DIFFUSION MODEL INFERENCE")
    print("="*80)
    print()

    # Load model
    model = load_model(args.checkpoint)

    # Create tokenizer
    tokenizer = AudioTokenizer()

    # Test cases
    print("\n" + "="*80)
    print("TEST CASES")
    print("="*80)

    test_files = []

    # 1. Synthetic silence samples (should predict False)
    silence_dir = Path("data/synthetic/silence")
    if silence_dir.exists():
        silence_samples = list(silence_dir.glob("silence_*.wav"))[:5]  # Test 5
        for wav_path in silence_samples:
            test_files.append((wav_path, False, "synthetic_silence"))

    # 2. Enrichment samples (should predict True)
    enrichment_file = Path("data/training/all_sessions_enriched.json")
    if enrichment_file.exists():
        with open(enrichment_file) as f:
            enrichment_data = json.load(f)

        # Test first 5 enrichment samples
        for sample in enrichment_data[:5]:
            session_id = sample['session_id']
            start_time = sample['start_time']
            end_time = sample['end_time']

            # Load session audio
            session_audio_path = Path(f"data/sessions/{session_id}/audio.wav")
            if not session_audio_path.exists():
                continue

            # Extract segment
            audio, sr = sf.read(session_audio_path)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = audio[start_sample:end_sample]

            # Save temp WAV
            temp_wav = Path("data/temp_test_segment.wav")
            temp_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(temp_wav, segment, sr)

            test_files.append((temp_wav, True, "enrichment"))

    # Run tests
    results = []

    for wav_path, expected, source in test_files:
        try:
            probability, confidence, metadata = predict_has_sound(
                model,
                wav_path,
                tokenizer,
                device=args.device
            )

            predicted = probability > 0.5
            correct = predicted == expected

            results.append({
                "file": wav_path.name,
                "source": source,
                "expected": expected,
                "predicted": predicted,
                "probability": probability,
                "confidence": confidence,
                "correct": correct,
                "duration": metadata['duration'],
                "tokens": metadata['seq_len']
            })

            # Print result
            status = "[OK]" if correct else "[WRONG]"
            print(f"\n{status} {wav_path.name}")
            print(f"  Source: {source}")
            print(f"  Expected: {expected}, Predicted: {predicted}")
            print(f"  Probability: {probability:.3f}, Confidence: {confidence:.3f}")
            print(f"  Duration: {metadata['duration']:.2f}s, Tokens: {metadata['seq_len']}")

        except Exception as e:
            print(f"\n[ERROR] {wav_path.name}: {e}")

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    if results:
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total

        print(f"\nTotal tests: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.1%}")

        # By source
        for source in ["synthetic_silence", "enrichment"]:
            source_results = [r for r in results if r['source'] == source]
            if source_results:
                source_correct = sum(1 for r in source_results if r['correct'])
                source_total = len(source_results)
                source_accuracy = source_correct / source_total
                print(f"\n{source}:")
                print(f"  Accuracy: {source_accuracy:.1%} ({source_correct}/{source_total})")

        # Confidence analysis
        avg_confidence = sum(r['confidence'] for r in results) / total
        print(f"\nAverage confidence: {avg_confidence:.3f}")

        # Save results
        output_file = Path("test_results_diffusion_stage0.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

    else:
        print("\nNo test files found. Please:")
        print("  1. Generate synthetic silence: python generate_synthetic_audio.py")
        print("  2. Enrich sessions: python enrich_all_sessions.py")

    print("\n" + "="*80)
    print("What These Results Mean")
    print("="*80)
    print("""
For your first diffusion model:
  - 50-70%: Model learned something, but class imbalance hurt
  - 70-85%: Pretty good! Model learned basic patterns
  - 85-95%: Excellent! Model generalizes well
  - >95%: Very impressive for first model

If accuracy is low:
  1. Check class balance (how many positive vs negative samples?)
  2. Try more training epochs (20 -> 50)
  3. Adjust learning rate (0.001 -> 0.0001)
  4. Get more positive examples (record more speech)

Key insight: Diffusion models work! You just built a temporal denoising
model that predicts has_sound by gradually removing noise from random
predictions. This is the foundation for all diffusion models.
    """)


if __name__ == "__main__":
    main()
