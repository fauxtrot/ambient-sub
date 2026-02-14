"""Compare Whisper on full audio vs utterances"""

import whisper
import soundfile as sf
from pathlib import Path

print("=" * 80)
print("WHISPER: FULL AUDIO vs UTTERANCES COMPARISON")
print("=" * 80)

# Find latest session
sessions_dir = Path("data/sessions")
latest_session = max([d for d in sessions_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime)

print(f"\nSession: {latest_session.name}\n")

# Load full audio
audio_file = latest_session / "audio.wav"
audio, sample_rate = sf.read(audio_file)

print(f"Audio file: {audio_file.name}")
print(f"Duration: {len(audio) / sample_rate:.2f}s")
print(f"Samples: {len(audio)} @ {sample_rate}Hz\n")

# Load Whisper model
print("Loading Whisper model (base)...")
whisper_model = whisper.load_model("base")

print("\n" + "=" * 80)
print("OPTION 1: TRANSCRIBE FULL AUDIO (all at once)")
print("=" * 80)

result = whisper_model.transcribe(audio.astype('float32'), fp16=False)
full_transcription = result['text'].strip()
language = result.get('language', 'unknown')

print(f"\nFull transcription:")
print(f'  "{full_transcription}"')
print(f"\nLanguage: {language}")

# Show segments from Whisper
if 'segments' in result:
    print(f"\nWhisper internal segments: {len(result['segments'])}")
    for i, seg in enumerate(result['segments']):
        print(f"  [{i}] {seg['start']:.2f}s - {seg['end']:.2f}s: \"{seg['text'].strip()}\"")

print("\n" + "=" * 80)
print("OPTION 2: TRANSCRIBE UTTERANCES (from UtteranceBuilder)")
print("=" * 80)

# Load merged utterances
import importlib.util
spec = importlib.util.spec_from_file_location(
    "utterance_builder",
    Path(__file__).parent / "ambient_subconscious" / "training" / "utterance_builder.py"
)
utterance_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utterance_builder)

UtteranceBuilder = utterance_builder.UtteranceBuilder

builder = UtteranceBuilder(overlap_threshold=1.0, min_duration=1.0)
utterances = builder.build_from_session(latest_session)

print(f"\nUtterances: {len(utterances)}")

utterance_transcriptions = []
for i, utt in enumerate(utterances):
    result = whisper_model.transcribe(utt.audio.astype('float32'), fp16=False)
    transcription = result['text'].strip()
    utterance_transcriptions.append(transcription)
    print(f"  [{i}] {utt.start_time:.2f}s - {utt.end_time:.2f}s ({utt.duration():.2f}s)")
    print(f'      "{transcription}"')

combined = " ".join(utterance_transcriptions)
print(f"\nCombined utterance transcription:")
print(f'  "{combined}"')

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\nFull audio transcription:")
print(f'  "{full_transcription}"')

print(f"\nUtterance-based transcription:")
print(f'  "{combined}"')

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("""
Key differences:

1. Full audio approach:
   - Transcribes entire 15.50s recording at once
   - Whisper may include silence/noise segments
   - No speaker attribution
   - Single pass through entire audio

2. Utterance-based approach:
   - Only transcribes speech segments (identified by diart)
   - Skips silence/gaps
   - Provides speaker attribution
   - Processes clean audio boundaries

3. Which is better?
   - For training data: Utterance-based (clean segments with speaker labels)
   - For quick transcription: Full audio (simpler, one API call)
   - For quality: Utterance-based (focused on actual speech)
""")
