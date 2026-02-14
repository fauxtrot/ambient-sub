# Diart Parameter Tuning Guide

## Overview

Based on testing, the current version of diart has limited configuration options through the API. However, there are effective ways to improve utterance detection.

## Current Findings

### Diart API Limitations
- `SpeakerDiarization()` accepts a `config` parameter of type `SpeakerDiarizationConfig`
- Config class location varies by diart version
- Direct parameter passing (`tau_active`, `gamma`) is not supported in current version

### What We Know About Diart's Behavior

From our test results:
- **Full audio Whisper:** Detected "to my cat. It is great. He is a cat who talks with his face, but he doesn't make sounds all the time. Who seems happy?"
- **Diart frames:** Only detected speech from 1.12s to 13.01s
- **Missed segments:**
  - "to my cat." (before 1.12s)
  - "He is a cat who talks with his face," (gap around 3-5s)

This suggests diart is missing:
1. Speech at the beginning (first ~1 second)
2. Quieter or less clear speech segments

## Tuning Approaches

### Option 1: Use Both Full Audio + Utterances (RECOMMENDED)

Instead of relying solely on diart detection, use a hybrid approach:

```python
# 1. Get full transcription (captures everything)
full_transcription = whisper_model.transcribe(full_audio)

# 2. Get utterances with speaker attribution (from diart)
utterances = builder.build_utterances(frames, audio)

# 3. Combine for training data
training_sample = {
    "full_text": full_transcription['text'],  # What was said
    "utterances": [                            # Who said what when
        {
            "speaker": utt.speaker_id,
            "start": utt.start_time,
            "end": utt.end_time,
            "text": whisper_model.transcribe(utt.audio)['text']
        }
        for utt in utterances
    ]
}
```

**Benefits:**
- Captures ALL speech (Whisper on full audio)
- Preserves speaker attribution (diart utterances)
- No missed segments

### Option 2: Adjust UtteranceBuilder Parameters

Tune the post-processing instead of diart itself:

```python
builder = UtteranceBuilder(
    overlap_threshold=1.5,  # Increase from 1.0s to merge more frames
    min_duration=0.5,       # Lower from 1.0s to keep shorter utterances
    sample_rate=16000
)
```

**Parameters:**
- `overlap_threshold`: Higher values (1.5-2.0s) merge more frames together
  - Good for: Capturing continuous speech with brief pauses
  - Bad for: Clear speaker turn boundaries

- `min_duration`: Lower values (0.5-0.8s) keep shorter utterances
  - Good for: Capturing brief responses, short words
  - Bad for: Creates more "garbage" fragments for Whisper

### Option 3: Modify Diart Config (Advanced)

If you need to tune diart itself, you'll need to:

1. **Find the config file location:**
   ```bash
   python -c "import diart; print(diart.__file__)"
   # Look for config.yaml or similar in that directory
   ```

2. **Key parameters to tune:**
   - `tau_active`: Speech activity threshold (0.0-1.0)
     - **Lower (0.3)**: More sensitive, catches quiet speech, more false positives
     - **Higher (0.7)**: Less sensitive, only loud/clear speech, fewer false positives
     - **Default (~0.5)**: Balanced

   - `gamma`: Speaker clustering threshold (0.0-1.0)
     - **Lower (0.2)**: More speaker changes, splits more aggressively
     - **Higher (0.5)**: Fewer speaker changes, merges similar speakers
     - **Default (~0.3)**: Balanced

   - `step`: Time between predictions (seconds)
     - **Lower (0.25s)**: More frequent updates, more overlapping frames
     - **Higher (1.0s)**: Fewer updates, less overlap
     - **Default (0.5s)**: Good balance

   - `duration`: Sliding window duration (seconds)
     - **Longer (7-10s)**: Better context, slower processing
     - **Shorter (3-4s)**: Faster processing, less context
     - **Default (5s)**: Good balance

3. **Update AudioListener to use config:**
   ```python
   # In ambient_subconscious/stream/audio_source.py

   from diart import SpeakerDiarization
   from diart.pipelines import SpeakerDiarizationConfig  # Or wherever it is

   class AudioListener:
       def __init__(self, device=None, audio_device=None, diart_config=None):
           # ...

           if diart_config:
               self.pipeline = SpeakerDiarization(config=diart_config)
           else:
               self.pipeline = SpeakerDiarization()
   ```

### Option 4: Pre-process Audio

Sometimes the issue is audio quality, not diart sensitivity:

```python
import numpy as np
from scipy.signal import butter, filtfilt

def preprocess_audio(audio, sample_rate=16000):
    """Enhance audio before diarization"""
    # 1. Normalize volume
    audio = audio / np.max(np.abs(audio))

    # 2. High-pass filter (remove low-frequency noise)
    nyquist = sample_rate / 2
    cutoff = 80  # Hz (remove rumble, hum)
    b, a = butter(4, cutoff / nyquist, btype='high')
    audio = filtfilt(b, a, audio)

    # 3. Amplify quiet sections (optional)
    # This can help diart detect quieter speech
    # But may amplify noise too

    return audio

# Use in recording pipeline
preprocessed_audio = preprocess_audio(raw_audio)
# Then pass to diart
```

## Recommendations for Your Use Case

Based on your goal (ambient all-day recording with minimal missed utterances):

### Immediate Actions:

1. **Use hybrid approach (Option 1)**:
   - Whisper on full audio → complete transcription
   - Diart utterances → speaker attribution + timing
   - Combine both in training data

2. **Adjust UtteranceBuilder**:
   ```python
   builder = UtteranceBuilder(
       overlap_threshold=1.5,  # Merge more frames
       min_duration=0.8,       # Balance between quality and completeness
   )
   ```

### Future Improvements:

3. **Test audio preprocessing**:
   - Normalize volume levels
   - Filter background noise
   - May improve diart's detection of quieter speech

4. **If still missing speech, tune diart config**:
   - Lower `tau_active` to 0.3 (more sensitive)
   - Test on your typical recording environment
   - Monitor false positive rate

## Testing Your Changes

After tuning, verify improvements:

```bash
# 1. Record new test session
python test_with_audio_recording.py

# 2. Compare full audio vs utterances
python test_whisper_full_audio.py

# 3. Check what percentage of speech is captured
# Compare:
#   - Full audio transcription length
#   - Utterance-based transcription length
#   - Missing segments

# 4. Listen to gaps
# If utterances span 1-5s and 8-15s, listen to audio from 5-8s
# Is there actual speech there, or just noise/silence?
```

## Summary

**Problem:** Diart missed "to my cat" and "He is a cat who talks with his face"

**Root Causes:**
1. Speech might be too quiet for default `tau_active`
2. Beginning buffer (first 1s) might be discarded
3. Gaps might contain actual quiet speech vs noise

**Best Solution:**
- **Short term:** Use hybrid approach (full audio + utterances)
- **Medium term:** Adjust UtteranceBuilder thresholds
- **Long term:** Tune diart config if hybrid still insufficient

**Trade-offs:**
- **Higher sensitivity:** More speech captured, more noise/false positives
- **Lower sensitivity:** Cleaner utterances, missed quiet speech
- **Hybrid approach:** Best of both worlds, slightly more complex pipeline
