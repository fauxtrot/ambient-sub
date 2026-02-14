"""Test audio capture and device detection"""

import sounddevice as sd
import numpy as np


def list_audio_devices():
    """List all available audio input devices"""
    print("Available audio devices:")
    print(sd.query_devices())

    default_input = sd.query_devices(kind='input')
    print(f"\nDefault input device: {default_input['name']}")
    return default_input


def test_audio_stream(duration=3, sample_rate=16000):
    """Test basic audio capture from default microphone"""
    print(f"\nRecording {duration} seconds of audio at {sample_rate} Hz...")
    print("Speak into your microphone!")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()

    print(f"Recorded {len(recording)} samples")
    print(f"Audio level (RMS): {np.sqrt(np.mean(recording**2)):.4f}")
    print(f"Peak amplitude: {np.max(np.abs(recording)):.4f}")

    if np.max(np.abs(recording)) < 0.01:
        print("\nWARNING: Very low audio level. Check microphone permissions and volume.")
    else:
        print("\nAudio capture working!")

    return recording


if __name__ == "__main__":
    list_audio_devices()
    test_audio_stream()
