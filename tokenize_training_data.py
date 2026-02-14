"""
Tokenize all audio (enrichment + synthetic) with Encodec.

This converts raw audio waveforms into discrete tokens that the diffusion
model can process.

For first-time ML builders:
    - Raw audio = continuous waveform (float values)
    - Encodec = neural audio codec that converts to discrete tokens
    - Tokens = integers from 0-1023 (vocab size = 1024)
    - ~75 tokens per second at 6kbps bandwidth
    - First codebook only for Stage 0 (simpler)

Why discrete tokens?
    - Diffusion models work better with discrete representations
    - Easier to learn patterns (1024 values vs infinite floats)
    - Enables self-feeding (use previous tokens as context)
"""

import torch
import soundfile as sf
import json
from pathlib import Path
import argparse
import numpy as np
from scipy import signal


def extract_acoustic_features(audio_segment, sr=16000, encodec_tokens=None):
    """
    Extract acoustic features for sound detection.

    Args:
        audio_segment: Raw audio waveform [samples] (numpy array)
        sr: Sample rate
        encodec_tokens: Optional Encodec tokens for token-based features

    Returns:
        Dict with 8 acoustic features
    """
    # Handle empty or very short audio
    if len(audio_segment) < 100:
        return {
            'rms': 0.0,
            'max_amplitude': 0.0,
            'energy': 0.0,
            'zero_crossing_rate': 0.0,
            'spectral_centroid': 0.0,
            'token_entropy': 0.0,
            'unique_token_ratio': 0.0,
            'token_concentration': 0.0
        }

    # 1. Amplitude features (direct sound indicators)
    rms = np.sqrt(np.mean(audio_segment**2))
    max_amp = np.max(np.abs(audio_segment))
    energy = np.sum(audio_segment**2) / len(audio_segment)

    # 2. Zero-crossing rate (frequency indicator)
    zero_crossings = np.where(np.diff(np.sign(audio_segment)))[0]
    zcr = len(zero_crossings) / len(audio_segment)

    # 3. Spectral centroid (frequency center of mass)
    try:
        f, t, Sxx = signal.spectrogram(audio_segment, sr)
        spectral_centroid = np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)
        spectral_centroid_mean = np.mean(spectral_centroid)
    except:
        spectral_centroid_mean = 0.0

    # 4. Token-based features (if Encodec tokens available)
    if encodec_tokens is not None and len(encodec_tokens) > 0:
        # Shannon entropy: how random are tokens?
        token_counts = np.bincount(encodec_tokens, minlength=1024)
        token_probs = token_counts / (len(encodec_tokens) + 1e-10)
        token_entropy = -np.sum(token_probs * np.log(token_probs + 1e-10))

        # Unique ratio: % of unique tokens
        unique_ratio = len(np.unique(encodec_tokens)) / len(encodec_tokens)

        # Concentration: how much mass in top-10 tokens?
        top10_mass = np.sum(np.sort(token_counts)[-10:]) / (np.sum(token_counts) + 1e-10)
    else:
        token_entropy = 0.0
        unique_ratio = 0.0
        top10_mass = 0.0

    return {
        'rms': float(rms),
        'max_amplitude': float(max_amp),
        'energy': float(energy),
        'zero_crossing_rate': float(zcr),
        'spectral_centroid': float(spectral_centroid_mean),
        'token_entropy': float(token_entropy),
        'unique_token_ratio': float(unique_ratio),
        'token_concentration': float(top10_mass)
    }


class AudioTokenizer:
    """
    Convert audio waveforms to discrete encodec tokens.

    This is our preprocessing step before training.
    """

    def __init__(self, bandwidth=6.0, device=None):
        """
        Initialize encodec model.

        Args:
            bandwidth: Target bandwidth in kbps (default: 6.0)
                      Higher = better quality, more tokens
                      Lower = faster, fewer tokens
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        # Lazy import (encodec is optional)
        try:
            from encodec import EncodecModel
            from encodec.utils import convert_audio
        except ImportError:
            raise ImportError(
                "encodec not installed. Install with: pip install encodec"
            )

        self.convert_audio = convert_audio

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading encodec model (24kHz, {bandwidth}kbps) on {device}...")

        # Load 24kHz model
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model = self.model.to(device)
        self.model.eval()

        self.device = device
        self.bandwidth = bandwidth

        print(f"  Device: {device}")
        print(f"  Bandwidth: {bandwidth}kbps")
        print(f"  Expected tokens: ~{int(75 * bandwidth / 6)} per second")

    def tokenize_wav(self, wav_path, use_first_codebook_only=True):
        """
        Convert WAV file to encodec tokens and extract acoustic features.

        Args:
            wav_path: Path to WAV file
            use_first_codebook_only: Use only first codebook (simpler for Stage 0)

        Returns:
            tokens: List of integers (0-1023) if first codebook only
            acoustic_features: Dict with 8 acoustic features
            metadata: Dict with duration, seq_len, etc.
        """
        # Load audio
        audio, sr = sf.read(wav_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Convert to tensor and resample to 24kHz
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, samples]
        audio_tensor = self.convert_audio(
            audio_tensor, sr, self.model.sample_rate, self.model.channels
        )
        audio_tensor = audio_tensor.unsqueeze(0).to(self.device)  # [1, 1, samples]

        # Encode to tokens
        with torch.no_grad():
            encoded_frames = self.model.encode(audio_tensor)

        # Extract codes: [[1, n_codebooks, seq_len]]
        # encoded_frames is a list of tuples: [(codes, scale), ...]
        # codes shape: [batch=1, n_codebooks=8, seq_len]
        codes = encoded_frames[0][0]  # [batch, n_codebooks, seq_len]

        if use_first_codebook_only:
            # Stage 0: Use only first codebook for simplicity
            # codes[0, 0] = first batch, first codebook -> [seq_len]
            tokens = codes[0, 0].cpu().tolist()  # List of integers
        else:
            # Future stages: Use all codebooks
            # codes[0] = first batch -> [n_codebooks, seq_len]
            tokens = codes[0].cpu().tolist()  # List of lists

        # NEW: Extract acoustic features
        acoustic_features = extract_acoustic_features(
            audio_segment=audio,  # Raw waveform (numpy array)
            sr=sr,
            encodec_tokens=tokens if use_first_codebook_only else None  # Only pass if 1D list
        )

        metadata = {
            "wav_path": str(wav_path),
            "duration": len(audio) / sr,
            "seq_len": codes.shape[1],
            "n_codebooks": codes.shape[0],
            "tokens_per_second": codes.shape[1] / (len(audio) / sr),
        }

        return tokens, acoustic_features, metadata


def tokenize_enrichment_data(enrichment_json, tokenizer):
    """
    Tokenize enrichment data (labeled examples with has_sound field).

    Args:
        enrichment_json: Path to labeled dataset JSON
        tokenizer: AudioTokenizer instance

    Returns:
        List of tokenized samples with labels
    """
    print(f"\nTokenizing labeled data from {enrichment_json}...")

    with open(enrichment_json) as f:
        enrichment_samples = json.load(f)

    training_data = []

    for i, sample in enumerate(enrichment_samples):
        # Load corresponding audio from session
        session_id = sample['session_id']
        start_time = sample['start_time']
        end_time = sample['end_time']

        # Load full session audio
        session_audio_path = Path(f"data/sessions/{session_id}/audio.wav")

        if not session_audio_path.exists():
            print(f"  [SKIP] Session {session_id} audio not found")
            continue

        audio, sr = sf.read(session_audio_path)

        # Extract segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = audio[start_sample:end_sample]

        # Save temp WAV and tokenize
        temp_wav = Path("data/temp_segment.wav")
        temp_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(temp_wav, segment, sr)

        try:
            tokens, acoustic_features, metadata = tokenizer.tokenize_wav(temp_wav)
        except Exception as e:
            print(f"  [ERROR] Failed to tokenize {sample['segment_id']}: {e}")
            continue
        finally:
            if temp_wav.exists():
                temp_wav.unlink()

        # Add to dataset (preserve has_sound label from input)
        training_data.append({
            "tokens": tokens,
            "acoustic": acoustic_features,  # NEW: Include acoustic features
            "has_sound": sample.get('has_sound', True),  # Use label from input
            "speaker_id": sample.get('speaker_id'),
            "transcription": sample.get('transcription'),
            "duration": sample['duration'],
            "source": sample.get('label_source', 'enrichment'),
            "sample_id": sample['segment_id']
        })

        if (i + 1) % 100 == 0:
            print(f"  Tokenized {i + 1}/{len(enrichment_samples)} samples...")

    print(f"  [OK] Tokenized {len(training_data)} labeled samples")
    print(f"    Positive (has_sound=true): {sum(1 for s in training_data if s['has_sound'])}")
    print(f"    Negative (has_sound=false): {sum(1 for s in training_data if not s['has_sound'])}")
    return training_data


def tokenize_synthetic_silence(silence_dir, tokenizer):
    """
    Tokenize synthetic silence (negative examples).

    Args:
        silence_dir: Path to data/synthetic/silence/
        tokenizer: AudioTokenizer instance

    Returns:
        List of tokenized samples with labels
    """
    print("\nTokenizing synthetic silence (negative examples)...")

    silence_metadata_path = Path(silence_dir) / "silence_metadata.json"

    if not silence_metadata_path.exists():
        print(f"  [ERROR] {silence_metadata_path} not found")
        return []

    with open(silence_metadata_path) as f:
        silence_samples = json.load(f)

    training_data = []

    for i, sample in enumerate(silence_samples):
        wav_path = Path(sample['file_path'])

        if not wav_path.exists():
            print(f"  [SKIP] {wav_path} not found")
            continue

        try:
            tokens, acoustic_features, metadata = tokenizer.tokenize_wav(wav_path)
        except Exception as e:
            print(f"  [ERROR] Failed to tokenize {wav_path.name}: {e}")
            continue

        training_data.append({
            "tokens": tokens,
            "acoustic": acoustic_features,  # NEW: Include acoustic features
            "has_sound": False,  # Silence = no sound
            "duration": sample['duration'],
            "source": "synthetic_silence",
            "sample_id": wav_path.stem
        })

        if (i + 1) % 50 == 0:
            print(f"  Tokenized {i + 1}/{len(silence_samples)} samples...")

    print(f"  [OK] Tokenized {len(training_data)} silence samples")
    return training_data


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize training data with encodec"
    )
    parser.add_argument(
        "--enrichment",
        type=str,
        default="data/training/all_sessions_enriched.json",
        help="Path to enrichment JSON"
    )
    parser.add_argument(
        "--silence",
        type=str,
        default="data/synthetic/silence",
        help="Path to synthetic silence directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/stage0_hybrid.pt",
        help="Output path for hybrid dataset (.pt format)"
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=6.0,
        help="Encodec bandwidth in kbps"
    )

    args = parser.parse_args()

    # Create tokenizer
    tokenizer = AudioTokenizer(bandwidth=args.bandwidth)

    # Tokenize both sources
    enrichment_data = tokenize_enrichment_data(args.enrichment, tokenizer)
    silence_data = tokenize_synthetic_silence(args.silence, tokenizer)

    # Combine
    all_training_data = enrichment_data + silence_data

    # Convert to PyTorch tensors for efficient .pt storage
    print(f"\nConverting to PyTorch tensors...")

    all_tokens = []
    all_acoustic = []
    all_labels = []
    all_metadata = []

    for sample in all_training_data:
        # Tokens as tensor
        all_tokens.append(torch.tensor(sample['tokens'], dtype=torch.long))

        # Acoustic features as tensor (8 features)
        acoustic_vec = [
            sample['acoustic']['rms'],
            sample['acoustic']['max_amplitude'],
            sample['acoustic']['energy'],
            sample['acoustic']['zero_crossing_rate'],
            sample['acoustic']['spectral_centroid'],
            sample['acoustic']['token_entropy'],
            sample['acoustic']['unique_token_ratio'],
            sample['acoustic']['token_concentration']
        ]
        all_acoustic.append(torch.tensor(acoustic_vec, dtype=torch.float32))

        # Label
        all_labels.append(torch.tensor([sample['has_sound']], dtype=torch.float32))

        # Metadata (keep for inspection)
        all_metadata.append({
            'transcription': sample.get('transcription', ''),
            'speaker_id': sample.get('speaker_id', 'unknown'),
            'duration': sample['duration'],
            'source': sample['source'],
            'sample_id': sample['sample_id']
        })

    # Pad tokens to same length
    tokens_padded = torch.nn.utils.rnn.pad_sequence(all_tokens, batch_first=True, padding_value=0)

    # Stack acoustic features
    acoustic_stacked = torch.stack(all_acoustic)

    # Stack labels
    labels_stacked = torch.stack(all_labels)

    # Compute statistics
    num_positive = labels_stacked.sum().item()
    num_negative = len(labels_stacked) - num_positive

    # Save as .pt
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}...")

    torch.save({
        'tokens': tokens_padded,  # [N, max_seq_len]
        'acoustic': acoustic_stacked,  # [N, 8]
        'labels': labels_stacked,  # [N, 1]
        'metadata': all_metadata,  # List of dicts
        'stats': {
            'num_samples': len(all_labels),
            'num_positive': int(num_positive),
            'num_negative': int(num_negative),
            'token_vocab_size': 1024,
            'max_seq_len': tokens_padded.shape[1],
            'acoustic_feature_dim': 8
        }
    }, output_path)

    print(f"\n[SUCCESS] Hybrid dataset created")
    print(f"  Format: PyTorch .pt (efficient tensor storage)")
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Positive (has_sound=true): {int(num_positive)}")
    print(f"  Negative (has_sound=false): {int(num_negative)}")
    print(f"  Saved to: {output_path}")

    # Tensor statistics
    print(f"\nTensor shapes:")
    print(f"  Tokens: {tokens_padded.shape} (padded sequences)")
    print(f"  Acoustic: {acoustic_stacked.shape} (8 features)")
    print(f"  Labels: {labels_stacked.shape}")

    # Token statistics
    avg_tokens = np.mean([len(t) for t in all_tokens])
    max_tokens = max([len(t) for t in all_tokens])
    min_tokens = min([len(t) for t in all_tokens])

    print(f"\nToken statistics:")
    print(f"  Average tokens per sample: {avg_tokens:.1f}")
    print(f"  Min tokens: {min_tokens}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Max sequence length (padded): {tokens_padded.shape[1]}")

    # Acoustic feature statistics
    print(f"\nAcoustic feature ranges:")
    for i, feature_name in enumerate(['rms', 'max_amplitude', 'energy', 'zero_crossing_rate',
                                      'spectral_centroid', 'token_entropy', 'unique_token_ratio',
                                      'token_concentration']):
        feature_vals = acoustic_stacked[:, i]
        print(f"  {feature_name:25s}: [{feature_vals.min():.4f}, {feature_vals.max():.4f}] (mean: {feature_vals.mean():.4f})")


if __name__ == "__main__":
    main()
