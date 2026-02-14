"""Explore and test diart parameters for better utterance detection"""

import inspect
from diart import SpeakerDiarization
from diart.sources import AudioSource
from diart.inference import StreamingInference
import soundfile as sf
from pathlib import Path
import numpy as np

print("=" * 80)
print("DIART PARAMETER EXPLORATION")
print("=" * 80)

# Inspect SpeakerDiarization to see available parameters
print("\nSpeakerDiarization class signature:")
sig = inspect.signature(SpeakerDiarization.__init__)
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
    print(f"  {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'} = {default}")

# Check the config class
try:
    from diart.pipelines import SpeakerDiarizationConfig
    print("\n\nSpeakerDiarizationConfig class:")
    sig = inspect.signature(SpeakerDiarizationConfig.__init__)
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
        print(f"  {param_name}: {default}")
except ImportError as e:
    print(f"\nSpeakerDiarizationConfig not found: {e}")

print("\n" + "=" * 80)
print("KEY PARAMETERS FOR UTTERANCE DETECTION")
print("=" * 80)

print("""
Based on diart documentation, these parameters affect detection:

1. Segmentation parameters:
   - tau_active: Threshold for speech activity detection (0.0-1.0)
     - Lower = more sensitive to quiet speech
     - Higher = only detect clear/loud speech
     - Default: ~0.5

2. Clustering parameters:
   - gamma: Threshold for speaker change detection (0.0-1.0)
     - Lower = more speaker changes (splits utterances more)
     - Higher = fewer speaker changes (merges utterances more)
     - Default: ~0.3

3. Temporal parameters:
   - step: Time between consecutive predictions (seconds)
     - Lower = more frequent updates (more overlapping frames)
     - Higher = fewer updates (less overlap)
     - Default: 0.5s (500ms)

   - duration: Duration of sliding window (seconds)
     - Longer = better context but slower
     - Shorter = faster but less accurate
     - Default: 5.0s

4. Latency parameters:
   - latency: How much buffering to allow (seconds)
     - Lower = faster but less accurate
     - Higher = slower but more stable
     - Default: varies by mode

Let's test different configurations...
""")

print("\n" + "=" * 80)
print("TESTING DIFFERENT CONFIGURATIONS")
print("=" * 80)

# Find latest session
sessions_dir = Path("data/sessions")
latest_session = max([d for d in sessions_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime)

audio_file = latest_session / "audio.wav"
audio, sample_rate = sf.read(audio_file)

print(f"\nTest audio: {audio_file.name}")
print(f"Duration: {len(audio) / sample_rate:.2f}s\n")

# Create audio source from file
class FileAudioSource(AudioSource):
    """Audio source that reads from a file"""
    def __init__(self, audio_data, sample_rate=16000):
        self.audio = audio_data
        self.sample_rate = sample_rate
        self.chunk_size = int(5.0 * sample_rate)  # 5s chunks
        self.position = 0

    def read(self):
        if self.position >= len(self.audio):
            return None
        end = min(self.position + self.chunk_size, len(self.audio))
        chunk = self.audio[self.position:end]
        self.position = end
        return chunk

    def close(self):
        """Required close method"""
        pass

# Import config class
try:
    from diart.pipelines import SpeakerDiarizationConfig
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("Warning: SpeakerDiarizationConfig not available")

# Test configurations
configs = [
    {
        "name": "Default (current)",
        "config": None
    },
]

if HAS_CONFIG:
    configs.extend([
        {
            "name": "High sensitivity (detect quiet speech)",
            "config": SpeakerDiarizationConfig(tau_active=0.3)
        },
        {
            "name": "Low sensitivity (only loud speech)",
            "config": SpeakerDiarizationConfig(tau_active=0.7)
        },
        {
            "name": "More speaker changes (split utterances)",
            "config": SpeakerDiarizationConfig(gamma=0.2)
        },
        {
            "name": "Fewer speaker changes (merge utterances)",
            "config": SpeakerDiarizationConfig(gamma=0.5)
        },
    ])

results = []

for config in configs:
    print(f"\n{'-' * 80}")
    print(f"Testing: {config['name']}")
    print(f"Config: {config['config']}")
    print(f"{'-' * 80}")

    try:
        # Create pipeline with config
        if config['config']:
            pipeline = SpeakerDiarization(config=config['config'])
        else:
            pipeline = SpeakerDiarization()

        # Create source
        source = FileAudioSource(audio, sample_rate)

        # Collect predictions
        predictions = []

        def collect_prediction(prediction):
            if prediction and len(prediction.labels()) > 0:
                predictions.append(prediction)
                print(f"\r    Processed {len(predictions)} predictions", end="", flush=True)

        # Run inference
        inference = StreamingInference(pipeline, source, batch_size=1)
        inference.attach_hooks(collect_prediction)

        try:
            for pred in inference:
                pass
        except StopIteration:
            pass

        print(f"\n    Predictions: {len(predictions)}")

        # Analyze results
        total_speech_time = 0
        num_segments = 0

        for pred in predictions:
            for segment, _, label in pred.itertracks(yield_label=True):
                num_segments += 1
                total_speech_time += segment.end - segment.start

        print(f"    Segments detected: {num_segments}")
        print(f"    Total speech time: {total_speech_time:.2f}s")
        print(f"    Speech ratio: {total_speech_time / (len(audio) / sample_rate) * 100:.1f}%")

        results.append({
            "name": config['name'],
            "predictions": len(predictions),
            "segments": num_segments,
            "speech_time": total_speech_time,
        })

    except Exception as e:
        print(f"    [ERROR] {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

if results:
    print(f"\n{'Configuration':<40} {'Predictions':<15} {'Segments':<15} {'Speech Time':<15}")
    print("-" * 85)
    for r in results:
        print(f"{r['name']:<40} {r['predictions']:<15} {r['segments']:<15} {r['speech_time']:<15.2f}s")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("""
Based on your results above:

1. If default detected too little speech:
   - Lower tau_active (e.g., 0.3) for higher sensitivity
   - This will catch quieter utterances

2. If default detected too much noise:
   - Raise tau_active (e.g., 0.7) for lower sensitivity
   - This will filter out background noise

3. If utterances are too fragmented:
   - Raise gamma (e.g., 0.5) to merge speaker segments
   - This will create longer, continuous utterances

4. If different speakers are being merged:
   - Lower gamma (e.g., 0.2) to split more aggressively
   - This will create clearer speaker boundaries

For your use case (ambient recording with missed speech), try:
   - tau_active = 0.3 (more sensitive)
   - gamma = 0.4 (moderate merging)
""")
else:
    print("\nNo successful tests. Check diart version and parameter compatibility.")
    print("\nAlternative approach: Adjust UtteranceBuilder parameters instead:")
    print("  - Increase overlap_threshold (e.g., 1.5s or 2.0s)")
    print("  - Lower min_duration (e.g., 0.5s)")
    print("  - This works with any diart version")
