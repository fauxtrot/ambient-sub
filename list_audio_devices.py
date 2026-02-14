"""
List all available audio input devices.

Helps you find the right device index for your webcam microphone.
"""

import sounddevice as sd

print("=" * 80)
print("AVAILABLE AUDIO INPUT DEVICES")
print("=" * 80)
print()

devices = sd.query_devices()

input_devices = []

for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        input_devices.append((i, device))

        # Mark default device
        default_marker = " (DEFAULT)" if i == sd.default.device[0] else ""

        print(f"[{i}] {device['name']}{default_marker}")
        print(f"    Channels: {device['max_input_channels']}")
        print(f"    Sample Rate: {device['default_samplerate']} Hz")
        print()

print("=" * 80)
print("WEBCAM DETECTION TIPS")
print("=" * 80)
print("""
Webcam microphones often have names like:
  - "USB Camera" or "USB Video Device"
  - "HD Webcam" or "Webcam Audio"
  - Device names with "USB" or "Camera"
  - Device names matching your webcam brand (Logitech, etc.)

Look for a device with:
  - 1-2 input channels (webcams usually have mono or stereo)
  - 44100 or 48000 Hz sample rate

To test a device, record a short sample:
  python test_with_audio_recording.py --device <index> --duration 5
""")

print("=" * 80)
print(f"Found {len(input_devices)} input device(s)")
print("=" * 80)
