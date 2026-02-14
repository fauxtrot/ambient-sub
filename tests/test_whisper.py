"""Test Whisper speech-to-text"""

import whisper
import torch
import numpy as np


def test_whisper_loading():
    """Test loading Whisper model"""
    print("Loading Whisper tiny model (will download if needed)...")
    import os
    os.makedirs('.models', exist_ok=True)
    model = whisper.load_model("tiny", download_root=".models")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model loaded: {model.__class__.__name__}")

    return model


def test_whisper_inference():
    """Test Whisper transcription on dummy audio"""
    model = test_whisper_loading()

    # Create a test audio signal (3 seconds of silence + noise)
    print("\nCreating test audio (3 seconds, 16kHz)...")
    sample_rate = 16000
    duration = 3
    audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.01

    print("Running transcription...")
    result = model.transcribe(audio, fp16=torch.cuda.is_available())

    print(f"\nTranscription result:")
    print(f"  Text: '{result['text']}'")
    print(f"  Language: {result['language']}")
    print(f"  Segments: {len(result['segments'])}")

    if result['text'].strip():
        print("\nNote: Whisper may hallucinate text from noise/silence.")
        print("This is expected behavior on random audio.")

    print("\nWhisper working!")

    return model


if __name__ == "__main__":
    test_whisper_inference()
