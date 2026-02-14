"""Test diart real-time speaker diarization"""

import os
import torch
from diart import SpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.inference import StreamingInference
import rx.operators as ops

# Set HuggingFace cache to .models directory
os.environ['HF_HOME'] = os.path.abspath('.models/huggingface')
os.environ['TORCH_HOME'] = os.path.abspath('.models/torch')


def test_diart_setup():
    """Test diart model loading"""
    print("Setting up diart speaker diarization...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize diarization pipeline
    config = SpeakerDiarization.get_config()
    print(f"Config: {config}")

    pipeline = SpeakerDiarization()
    print(f"Pipeline loaded: {pipeline.__class__.__name__}")

    return pipeline


def test_diart_streaming(duration=10):
    """Test streaming diarization from microphone"""
    print(f"\nStarting {duration}s streaming diarization test...")
    print("Speak into your microphone to test speaker detection!")

    pipeline = test_diart_setup()

    # Create microphone source
    mic = MicrophoneAudioSource(pipeline.sample_rate)

    # Create streaming inference
    inference = StreamingInference(pipeline, do_profile=False)

    # Track events
    event_count = 0

    def on_prediction(prediction):
        nonlocal event_count
        event_count += 1
        if event_count % 10 == 0:  # Print every 10th prediction
            print(f"Prediction {event_count}: {prediction}")

    # Run inference
    print("\nListening...")
    inference(mic).pipe(
        ops.take_while(lambda ann: ann[0].end < duration),
        ops.do_action(on_prediction)
    ).subscribe(
        on_next=lambda _: None,
        on_error=lambda e: print(f"Error: {e}"),
        on_completed=lambda: print(f"\nComplete! Received {event_count} predictions")
    )

    print("\nDiart streaming test complete!")


if __name__ == "__main__":
    try:
        test_diart_streaming()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nIf you see microphone access errors, check Windows permissions.")
