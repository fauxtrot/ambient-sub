"""Quick test to verify audio input device is working"""
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path

# Test parameters
DEVICE_ID = 8  # Change this to test different devices
DURATION = 5  # seconds
SAMPLE_RATE = 16000
OUTPUT_FILE = "test_recording.wav"

print("=" * 70)
print("AUDIO INPUT TEST")
print("=" * 70)

# Show device info
device_info = sd.query_devices(DEVICE_ID)
print(f"\nTesting device {DEVICE_ID}:")
print(f"  Name: {device_info['name']}")
print(f"  Max input channels: {device_info['max_input_channels']}")
print(f"  Default sample rate: {device_info['default_samplerate']}")

print(f"\nRecording {DURATION} seconds...")
print("SPEAK NOW!")

# Record
try:
    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        device=DEVICE_ID,
        dtype=np.float32
    )
    sd.wait()  # Wait for recording to finish

    print("\nRecording complete!")

    # Save to file
    sf.write(OUTPUT_FILE, recording, SAMPLE_RATE)
    print(f"Saved to: {OUTPUT_FILE}")

    # Analyze the recording
    max_amplitude = np.max(np.abs(recording))
    rms = np.sqrt(np.mean(recording**2))

    print(f"\nAudio analysis:")
    print(f"  Max amplitude: {max_amplitude:.4f}")
    print(f"  RMS level: {rms:.4f}")

    if max_amplitude < 0.001:
        print("\n  WARNING: Audio level is extremely low!")
        print("  Possible issues:")
        print("    - Microphone is muted")
        print("    - Wrong device selected")
        print("    - Microphone not connected")
        print("    - Microphone permissions not granted")
    elif max_amplitude < 0.01:
        print("\n  WARNING: Audio level is very low")
        print("  You might want to increase microphone volume")
    else:
        print("\n  SUCCESS: Audio was captured!")
        print(f"  Play the file '{OUTPUT_FILE}' to verify it recorded correctly")

except Exception as e:
    print(f"\nERROR: Failed to record audio")
    print(f"  {e}")
    print("\nTry a different device index with:")
    print("  python list_audio_devices.py")
