"""Test script to save raw audio and verify capture"""
import os
import time
import numpy as np
import soundfile as sf
from diart.sources import MicrophoneAudioSource
from diart import SpeakerDiarization
from diart.inference import StreamingInference

# Create recordings directory if it doesn't exist
os.makedirs("recordings", exist_ok=True)

print("=" * 60)
print("AUDIO CAPTURE TEST - Saving to WAV")
print("=" * 60)

sample_rate = 16000
duration_seconds = 10

# Create microphone source
print(f"\nInitializing microphone (sample rate: {sample_rate} Hz)...")
mic = MicrophoneAudioSource(sample_rate)

# Create pipeline and inference
print("Loading speaker diarization model...")
pipeline = SpeakerDiarization()
inference = StreamingInference(pipeline, mic, do_profile=False, do_plot=False)

# Storage for audio chunks
audio_chunks = []
annotation_count = [0]

def on_annotation(annotation_wav):
    """Capture audio chunks as they arrive"""
    annotation, waveform = annotation_wav
    annotation_count[0] += 1

    # waveform shape: (channels, samples) or (samples,)
    print(f"[{annotation_count[0]}] Captured chunk: shape={waveform.shape}, duration={waveform.shape[-1]/sample_rate:.2f}s")

    # Store the waveform
    audio_chunks.append(waveform)

    # Show annotation info
    track_count = len(list(annotation.itertracks()))
    print(f"    Annotation has {track_count} tracks, timestamp={annotation.end:.2f}s")

print(f"\n{'=' * 60}")
print(f"RECORDING {duration_seconds} SECONDS...")
print("SPEAK INTO YOUR MICROPHONE NOW!")
print(f"{'=' * 60}\n")

# Attach hook
inference.attach_hooks(on_annotation)

# Start recording
start_time = time.time()

# Run inference in a thread and timeout after duration
import threading

def run_inference():
    try:
        inference()
    except Exception as e:
        print(f"Inference error: {e}")

thread = threading.Thread(target=run_inference, daemon=True)
thread.start()

# Wait for duration
time.sleep(duration_seconds)

print(f"\n{'=' * 60}")
print("RECORDING COMPLETE")
print(f"{'=' * 60}")
print(f"Annotations received: {annotation_count[0]}")
print(f"Audio chunks captured: {len(audio_chunks)}")

if audio_chunks:
    # Concatenate all audio chunks
    print("\nConcatenating audio chunks...")

    # Handle different shapes
    if len(audio_chunks[0].shape) == 2:
        # (channels, samples) - take first channel
        audio_data = np.concatenate([chunk[0] for chunk in audio_chunks])
    else:
        # (samples,) - use directly
        audio_data = np.concatenate(audio_chunks)

    # Save to WAV file
    output_file = f"recordings/test_capture_{int(time.time())}.wav"
    print(f"Saving to: {output_file}")
    print(f"Audio shape: {audio_data.shape}")
    print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")

    sf.write(output_file, audio_data, sample_rate)

    print(f"\n{'=' * 60}")
    print("SUCCESS - Audio saved!")
    print(f"{'=' * 60}")
    print(f"File: {output_file}")
    print(f"Listen to this file to verify audio is being captured.")
    print(f"If you hear yourself, audio capture works!")
    print(f"If silent, there's a device/input issue.")
else:
    print("\nWARNING: No audio chunks captured!")
    print("This means the on_annotation callback was never triggered.")
    print("Possible causes:")
    print("  - Audio device not accessible")
    print("  - No default microphone selected")
    print("  - Permissions issue")
    print("  - diart not receiving audio from the source")
