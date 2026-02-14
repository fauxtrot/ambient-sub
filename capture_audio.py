"""Simple audio capture to file"""
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime

# Configuration
RECORDINGS_DIR = "recordings"
SAMPLE_RATE = 16000  # Hz
DURATION = 10  # seconds

# Device selection - edit this list
DEVICES_TO_CAPTURE = [
    5,   # Microphone (Realtek Audio)
    # 7,   # Microphone (V8S)
    # 98,  # Stereo Mix
]

def capture_from_device(device_id, duration, sample_rate):
    """Capture audio from a single device"""
    try:
        device_info = sd.query_devices(device_id)
        device_name = device_info['name']
        channels = min(2, device_info['max_input_channels'])  # Mono or stereo

        print(f"\n[Device {device_id}] {device_name}")
        print(f"  Channels: {channels}, Sample rate: {sample_rate} Hz, Duration: {duration}s")
        print(f"  Recording...")

        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            device=device_id,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = device_name.replace(' ', '_').replace('(', '').replace(')', '')[:50]
        filename = f"{timestamp}_{device_id}_{safe_name}.wav"
        filepath = os.path.join(RECORDINGS_DIR, filename)

        # Save to file
        sf.write(filepath, recording, sample_rate)

        # Calculate some stats
        duration_actual = len(recording) / sample_rate
        peak_amplitude = np.max(np.abs(recording))

        print(f"  [OK] Saved: {filename}")
        print(f"  Duration: {duration_actual:.2f}s, Peak: {peak_amplitude:.4f}")

        if peak_amplitude < 0.001:
            print(f"  [WARNING] Very low amplitude - device may be silent or wrong input")

        return filepath

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    """Capture audio from selected devices"""

    # Create recordings directory
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    print("=" * 70)
    print("AUDIO CAPTURE TO FILE")
    print("=" * 70)
    print(f"\nRecordings will be saved to: {RECORDINGS_DIR}/")
    print(f"Duration: {DURATION} seconds")
    print(f"Sample rate: {SAMPLE_RATE} Hz")

    if not DEVICES_TO_CAPTURE:
        print("\nERROR: No devices selected!")
        print("Edit DEVICES_TO_CAPTURE in this script to add device IDs.")
        print("Run list_audio_devices.py to see available devices.")
        return

    print(f"\nDevices to capture: {DEVICES_TO_CAPTURE}")
    print("\n" + "=" * 70)
    print("RECORDING - SPEAK NOW!")
    print("=" * 70)

    # Capture from each device
    files_created = []
    for device_id in DEVICES_TO_CAPTURE:
        filepath = capture_from_device(device_id, DURATION, SAMPLE_RATE)
        if filepath:
            files_created.append(filepath)

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nCaptured {len(files_created)} files:")
    for filepath in files_created:
        print(f"  - {filepath}")

    print("\nListen to these files to verify audio is being captured.")
    print("If silent, try a different device ID from list_audio_devices.py")


if __name__ == "__main__":
    main()
