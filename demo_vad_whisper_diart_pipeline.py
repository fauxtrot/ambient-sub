"""
Post-processing pipeline: VAD → Whisper → Diart → Gap Analysis

This is the RECOMMENDED approach for enriching sessions:
1. Use VAD to detect all speech (more sensitive than diart)
2. Transcribe VAD segments with Whisper
3. Use diart for speaker attribution (its strength)
4. Analyze gaps to measure quality
"""

import json
from pathlib import Path
import soundfile as sf
import whisper
import numpy as np
import importlib.util

# Import UtteranceBuilder
spec = importlib.util.spec_from_file_location(
    "utterance_builder",
    Path(__file__).parent / "ambient_subconscious" / "training" / "utterance_builder.py"
)
utterance_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utterance_builder)

UtteranceBuilder = utterance_builder.UtteranceBuilder

print("=" * 80)
print("VAD + WHISPER + DIART POST-PROCESSING PIPELINE")
print("=" * 80)

# Find latest session
sessions_dir = Path("data/sessions")
latest_session = max([d for d in sessions_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime)

print(f"\nSession: {latest_session.name}\n")

# Load audio
audio_file = latest_session / "audio.wav"
audio, sample_rate = sf.read(audio_file)

print(f"Audio: {len(audio) / sample_rate:.2f}s @ {sample_rate}Hz\n")

# Load Whisper model
print("Loading Whisper model (base)...")
whisper_model = whisper.load_model("base")

print("\n" + "=" * 80)
print("STEP 1: VAD-BASED SPEECH DETECTION")
print("=" * 80)

# Simple VAD using energy threshold
# (You could use silero-vad or Whisper's VAD for better results)
def simple_vad(audio, sample_rate=16000, threshold=0.01, min_duration=0.3):
    """
    Simple energy-based VAD.

    Args:
        audio: Audio samples
        sample_rate: Sample rate
        threshold: Energy threshold (amplitude)
        min_duration: Minimum segment duration (seconds)

    Returns:
        List of (start_sample, end_sample) tuples
    """
    # Compute frame energy
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)    # 10ms hop

    energy = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        energy.append(np.sqrt(np.mean(frame ** 2)))  # RMS energy

    # Threshold to get speech/non-speech
    speech_frames = np.array(energy) > threshold

    # Find contiguous speech segments
    segments = []
    in_speech = False
    start = None

    for i, is_speech in enumerate(speech_frames):
        if is_speech and not in_speech:
            # Start of speech segment
            start = i * hop_length
            in_speech = True
        elif not is_speech and in_speech:
            # End of speech segment
            end = i * hop_length
            duration = (end - start) / sample_rate

            if duration >= min_duration:
                segments.append((start, end))

            in_speech = False

    # Handle last segment
    if in_speech:
        end = len(audio)
        duration = (end - start) / sample_rate
        if duration >= min_duration:
            segments.append((start, end))

    return segments

print("\nTrying Whisper's VAD first (more accurate)...")
try:
    # Try using Whisper's built-in VAD
    result_full = whisper_model.transcribe(
        audio.astype('float32'),
        fp16=False,
        word_timestamps=True  # Get word-level timestamps
    )

    # Extract speech segments from Whisper's segments
    vad_segments = []
    if 'segments' in result_full:
        for seg in result_full['segments']:
            start = int(seg['start'] * sample_rate)
            end = int(seg['end'] * sample_rate)
            vad_segments.append((start, end))

    print(f"Using Whisper's VAD: {len(vad_segments)} segments")

except Exception as e:
    print(f"Whisper VAD failed: {e}")
    print("\nFalling back to simple energy-based VAD...")
    vad_segments = simple_vad(audio, sample_rate, threshold=0.005, min_duration=0.3)  # Lower threshold

print(f"\nVAD Results:")
print(f"  Segments detected: {len(vad_segments)}")
print(f"  Total speech time: {sum((e - s) / sample_rate for s, e in vad_segments):.2f}s")

print("\nVAD Segments:")
for i, (start, end) in enumerate(vad_segments):
    start_time = start / sample_rate
    end_time = end / sample_rate
    duration = end_time - start_time
    print(f"  [{i}] {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")

print("\n" + "=" * 80)
print("STEP 2: WHISPER TRANSCRIPTION ON VAD SEGMENTS")
print("=" * 80)

vad_transcriptions = []

for i, (start, end) in enumerate(vad_segments):
    start_time = start / sample_rate
    end_time = end / sample_rate

    # Extract segment
    segment_audio = audio[start:end]

    # Transcribe
    result = whisper_model.transcribe(segment_audio.astype('float32'), fp16=False)
    text = result['text'].strip()

    vad_transcriptions.append({
        "segment_id": i,
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "text": text,
        "language": result.get('language', 'unknown')
    })

    print(f"\n[VAD Segment {i}] {start_time:.2f}s - {end_time:.2f}s")
    print(f"  \"{text}\"")

# Combine all transcriptions
full_vad_text = " ".join([t['text'] for t in vad_transcriptions])

print("\n" + "=" * 80)
print("STEP 3: DIART SPEAKER ATTRIBUTION")
print("=" * 80)

print("\nBuilding utterances from diart frames...")
builder = UtteranceBuilder(overlap_threshold=1.0, min_duration=1.0)
utterances = builder.build_from_session(latest_session)

print(f"\nDiart Results:")
print(f"  Utterances: {len(utterances)}")
print(f"  Total speech time: {sum(u.duration() for u in utterances):.2f}s")

print("\nUtterances with transcription:")
utterance_transcriptions = []

for i, utt in enumerate(utterances):
    result = whisper_model.transcribe(utt.audio.astype('float32'), fp16=False)
    text = result['text'].strip()

    utterance_transcriptions.append({
        "utterance_id": i,
        "speaker": utt.speaker_id,
        "start_time": utt.start_time,
        "end_time": utt.end_time,
        "duration": utt.duration(),
        "text": text
    })

    print(f"\n[Utterance {i}] {utt.start_time:.2f}s - {utt.end_time:.2f}s")
    print(f"  Speaker: {utt.speaker_id}")
    print(f"  \"{text}\"")

# Combine all utterance transcriptions
full_utterance_text = " ".join([t['text'] for t in utterance_transcriptions])

print("\n" + "=" * 80)
print("STEP 4: GAP ANALYSIS")
print("=" * 80)

print("\nFull transcriptions:")
print(f"\nVAD-based:       \"{full_vad_text}\"")
print(f"Utterance-based: \"{full_utterance_text}\"")

# Calculate coverage
vad_words = set(full_vad_text.lower().split())
utterance_words = set(full_utterance_text.lower().split())

coverage = len(utterance_words) / len(vad_words) if len(vad_words) > 0 else 0
missing_words = vad_words - utterance_words

print(f"\nCoverage Analysis:")
print(f"  VAD detected words: {len(vad_words)}")
print(f"  Utterance captured words: {len(utterance_words)}")
print(f"  Coverage: {coverage * 100:.1f}%")
print(f"  Missing words: {missing_words}")

# Temporal gap analysis
print(f"\nTemporal Gap Analysis:")

# Find gaps where VAD detected speech but utterances didn't
for vad_seg in vad_transcriptions:
    # Check if this VAD segment overlaps with any utterance
    overlaps = False
    for utt in utterance_transcriptions:
        # Check for temporal overlap
        if (vad_seg['start_time'] < utt['end_time'] and
            vad_seg['end_time'] > utt['start_time']):
            overlaps = True
            break

    if not overlaps:
        print(f"\n  [GAP] {vad_seg['start_time']:.2f}s - {vad_seg['end_time']:.2f}s")
        print(f"    VAD detected: \"{vad_seg['text']}\"")
        print(f"    Diart missed this segment!")

print("\n" + "=" * 80)
print("ENRICHMENT RESULTS")
print("=" * 80)

# Create enriched training samples
enriched_samples = []

# Strategy: Use VAD segments as base, add speaker from nearest utterance
for vad_seg in vad_transcriptions:
    # Find overlapping or nearest utterance
    best_utterance = None
    best_overlap = 0

    for utt in utterance_transcriptions:
        # Calculate overlap
        overlap_start = max(vad_seg['start_time'], utt['start_time'])
        overlap_end = min(vad_seg['end_time'], utt['end_time'])
        overlap = max(0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_utterance = utt

    # Create enriched sample
    sample = {
        "segment_id": vad_seg['segment_id'],
        "start_time": vad_seg['start_time'],
        "end_time": vad_seg['end_time'],
        "duration": vad_seg['duration'],
        "text": vad_seg['text'],
        "language": vad_seg['language'],

        # Speaker attribution (from diart)
        "speaker": best_utterance['speaker'] if best_utterance else "unknown",
        "speaker_confidence": best_overlap / vad_seg['duration'] if best_utterance else 0.0,

        # Quality metrics
        "detected_by_diart": best_overlap > 0,
    }

    enriched_samples.append(sample)

print(f"\nEnriched Samples: {len(enriched_samples)}")
print(f"\nSample breakdown:")
print(f"  With speaker attribution: {sum(1 for s in enriched_samples if s['detected_by_diart'])}")
print(f"  Without speaker (gaps): {sum(1 for s in enriched_samples if not s['detected_by_diart'])}")

# Save enriched data
output_file = latest_session / "enriched_samples.json"
with open(output_file, 'w') as f:
    json.dump(enriched_samples, f, indent=2)

print(f"\nSaved to: {output_file}")

print("\n" + "=" * 80)
print("EXAMPLE ENRICHED SAMPLE")
print("=" * 80)

if enriched_samples:
    best_sample = max(enriched_samples, key=lambda s: len(s['text']))
    print("\nBest example (longest text):")
    print(json.dumps(best_sample, indent=2))

print("\n" + "=" * 80)
print("[SUCCESS] VAD + WHISPER + DIART PIPELINE COMPLETE")
print("=" * 80)

print("""
This pipeline provides:
  [OK] Complete speech coverage (VAD-based)
  [OK] Speaker attribution (from diart)
  [OK] Gap analysis (quantifies quality)
  [OK] Enriched training samples

Advantages over utterance-only approach:
  1. Catches all speech (VAD more sensitive than diart)
  2. Still gets speaker info (diart's strength)
  3. Identifies and quantifies gaps
  4. Creates richer training data

Next steps:
  1. Use enriched_samples.json for training
  2. Test with longer sessions (hours)
  3. Consider better VAD (silero-vad, pyannote.audio)
  4. Add speaker embedding extraction
""")
