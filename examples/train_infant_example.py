"""Example: Train and test Infant Model Stage 0"""

import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ambient_subconscious.training import (
    SessionLoader,
    EncodecTokenizer,
    TrainingSampleGenerator,
    LabelPipeline,
    AudioFeatureLabelSource,
)
from ambient_subconscious.training.infant_model import InfantModelStage0


def demo_data_pipeline():
    """Demonstrate the data pipeline"""
    print("=" * 60)
    print("DEMO: Data Pipeline")
    print("=" * 60)

    # Load a session
    loader = SessionLoader("data/sessions")
    sessions = loader.list_sessions()

    if not sessions:
        print("No sessions found. Record one first with:")
        print("  python test_with_audio_recording.py")
        return None

    print(f"\nFound {len(sessions)} sessions")
    session_id = sessions[0]
    print(f"Loading session: {session_id}")

    session = loader.load_session(session_id)
    print(f"  Frames: {len(session.frames)}")
    print(f"  Audio length: {len(session.audio) / 16000:.2f}s")

    # Generate labels
    print("\nGenerating labels...")
    pipeline = LabelPipeline([AudioFeatureLabelSource(threshold=0.01)])
    ground_truth = pipeline.generate_ground_truth(session)

    print(f"  Has sound labels: {len(ground_truth.labels.get('has_sound', []))}")
    if ground_truth.labels.get('has_sound'):
        sound_count = sum(ground_truth.labels['has_sound'])
        print(f"  Frames with sound: {sound_count}/{len(ground_truth)}")

    # Generate training samples
    print("\nGenerating training samples...")
    tokenizer = EncodecTokenizer(bandwidth=6.0)
    generator = TrainingSampleGenerator(tokenizer)

    samples = generator.generate_samples(session, ground_truth)
    print(f"  Generated {len(samples)} samples")

    if samples:
        print(f"  Sample 0: {len(samples[0].encodec_tokens)} tokens, has_sound={samples[0].ground_truth.get('has_sound')}")

    return samples


def demo_model_inference():
    """Demonstrate model inference"""
    print("\n" + "=" * 60)
    print("DEMO: Model Inference")
    print("=" * 60)

    model_path = Path("models/stage0_best.pt")

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Train the model first with:")
        print("  python -m ambient_subconscious.training.train_infant --epochs 5")
        return

    # Load model
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InfantModelStage0.load(str(model_path), device=device)
    print(f"Model loaded on {device}")

    # Create dummy input
    print("\nRunning inference on dummy data...")
    dummy_tokens = torch.randint(0, 1024, (1, 100)).to(device)

    prediction = model.predict(dummy_tokens)
    has_sound_prob = prediction[0, 0].item()

    print(f"  Prediction: has_sound={has_sound_prob:.4f}")
    print(f"  Binary: {'HAS SOUND' if has_sound_prob > 0.5 else 'NO SOUND'}")


def main():
    print("=" * 60)
    print("INFANT MODEL STAGE 0 - EXAMPLE")
    print("=" * 60)

    # Demo data pipeline
    samples = demo_data_pipeline()

    # Demo model inference
    demo_model_inference()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Record sessions:")
    print("   python test_with_audio_recording.py")
    print("\n2. Train Stage 0 model:")
    print("   python -m ambient_subconscious.training.train_infant --epochs 10")
    print("\n3. The model will learn to detect has_sound from audio!")


if __name__ == "__main__":
    main()
