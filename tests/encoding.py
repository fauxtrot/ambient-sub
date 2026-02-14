"""
Encodec Audio Round-Trip Experiment
===================================
Records audio (or loads existing file), encodes to discrete tokens,
decodes back to audio, and saves both for comparison.

Usage:
    python encodec_experiment.py                    # Record 5 seconds from mic
    python encodec_experiment.py --input test.wav   # Use existing file
    python encodec_experiment.py --bandwidth 1.5    # Test different quality levels
"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio

try:
    import sounddevice as sd
    import numpy as np
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

from encodec import EncodecModel
from encodec.utils import convert_audio


def record_audio(duration: float, sample_rate: int, output_path: str) -> None:
    """Record audio from default microphone."""
    if not HAS_SOUNDDEVICE:
        raise RuntimeError("sounddevice not installed. Run: pip install sounddevice")
    
    print(f"Recording {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("Recording complete.")
    
    # Convert to tensor and save
    audio_tensor = torch.from_numpy(audio.T)  # [channels, samples]
    torchaudio.save(output_path, audio_tensor, sample_rate)
    print(f"Saved to {output_path}")


def playback_audio(audio_path: str) -> None:
    """Play audio file using sounddevice."""
    if not HAS_SOUNDDEVICE:
        raise RuntimeError("sounddevice not installed")

    print(f"Playing {audio_path}...")
    audio, sr = torchaudio.load(audio_path)

    # Convert to numpy for sounddevice
    audio_np = audio.squeeze().numpy()

    # Play
    sd.play(audio_np, sr)
    sd.wait()
    print("Playback complete.")


def encode_decode_roundtrip(
    input_path: str,
    output_path: str,
    tokens_path: str,
    bandwidth: float = 6.0,
    playback: bool = False
) -> dict:
    """
    Encode audio to Encodec tokens, then decode back.
    Returns stats about the encoding.

    Args:
        input_path: Path to input audio file
        output_path: Path to save reconstructed audio
        tokens_path: Path to save token data as JSON
        bandwidth: Encodec bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0)
        playback: If True, play original and reconstructed audio for comparison
    """
    # Load model
    print(f"Loading Encodec model (24kHz, {bandwidth}kbps)...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    
    # Load audio
    print(f"Loading {input_path}...")
    wav, sr = torchaudio.load(input_path)
    duration = wav.shape[-1] / sr
    print(f"  Duration: {duration:.2f}s, Sample rate: {sr}Hz, Channels: {wav.shape[0]}")
    
    # Convert to model's expected format
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)  # Add batch dimension
    
    # Encode
    print("Encoding to tokens...")
    with torch.no_grad():
        encoded = model.encode(wav)
    
    codes = encoded[0][0]  # [batch, n_codebooks, seq_len] -> [n_codebooks, seq_len]
    n_codebooks, seq_len = codes.shape[1], codes.shape[2]
    
    print(f"  Codebooks: {n_codebooks}")
    print(f"  Sequence length: {seq_len} tokens")
    print(f"  Tokens per second: {seq_len / duration:.1f}")
    print(f"  Total tokens: {n_codebooks * seq_len}")
    print(f"  Compression: {wav.numel()} samples -> {n_codebooks * seq_len} tokens")
    print(f"  First 10 tokens (codebook 0): {codes[0, 0, :10].tolist()}")
    
    # Decode
    print("Decoding back to audio...")
    with torch.no_grad():
        reconstructed = model.decode(encoded)
    
    # Save reconstructed audio
    torchaudio.save(output_path, reconstructed.squeeze(0).cpu(), model.sample_rate)
    print(f"Saved reconstructed audio to {output_path}")
    
    # Save tokens
    token_data = {
        "input_file": str(input_path),
        "sample_rate": model.sample_rate,
        "bandwidth_kbps": bandwidth,
        "duration_seconds": duration,
        "n_codebooks": n_codebooks,
        "seq_len": seq_len,
        "tokens_per_second": seq_len / duration,
        "codebooks": codes[0].tolist()  # [n_codebooks, seq_len]
    }
    
    with open(tokens_path, "w") as f:
        json.dump(token_data, f, indent=2)
    print(f"Saved token data to {tokens_path}")

    # Playback comparison if requested
    if playback:
        print("\n" + "=" * 60)
        print("PLAYBACK COMPARISON")
        print("=" * 60)
        input("\nPress Enter to play ORIGINAL audio...")
        playback_audio(input_path)

        input("\nPress Enter to play RECONSTRUCTED audio...")
        playback_audio(output_path)

        print("\nListen for:")
        print("  - Speech clarity and intelligibility")
        print("  - Tone/emotion preservation")
        print("  - Background noise handling")
        print("  - Artifacts or distortion")

    return token_data


def compare_bandwidths(input_path: str, output_dir: str) -> None:
    """Encode at multiple bandwidths for comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    bandwidths = [1.5, 3.0, 6.0, 12.0, 24.0]
    
    for bw in bandwidths:
        print(f"\n{'='*50}")
        print(f"Bandwidth: {bw} kbps")
        print('='*50)
        
        output_path = output_dir / f"output_{bw}kbps.wav"
        tokens_path = output_dir / f"tokens_{bw}kbps.json"
        
        encode_decode_roundtrip(input_path, str(output_path), str(tokens_path), bw)
    
    print(f"\n\nAll outputs saved to {output_dir}/")
    print("Listen to each output_*kbps.wav to hear quality differences.")


def main():
    parser = argparse.ArgumentParser(description="Encodec audio round-trip experiment")
    parser.add_argument("--input", "-i", type=str, help="Input audio file (if not provided, records from mic)")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output audio file")
    parser.add_argument("--tokens", "-t", type=str, default="tokens.json", help="Output tokens file")
    parser.add_argument("--bandwidth", "-b", type=float, default=6.0, 
                        choices=[1.5, 3.0, 6.0, 12.0, 24.0], help="Bandwidth in kbps")
    parser.add_argument("--duration", "-d", type=float, default=10.0, help="Recording duration (seconds)")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare all bandwidth levels")
    parser.add_argument("--playback", "-p", action="store_true", help="Play original and reconstructed audio")
    
    args = parser.parse_args()
    
    # Determine input file
    if args.input:
        input_path = args.input
    else:
        input_path = "input.wav"
        record_audio(args.duration, 24000, input_path)
    
    # Run experiment
    if args.compare:
        compare_bandwidths(input_path, "bandwidth_comparison")
    else:
        encode_decode_roundtrip(input_path, args.output, args.tokens, args.bandwidth, playback=args.playback)
        if not args.playback:
            print(f"\n\nDone! Compare {input_path} vs {args.output}")
            print("Listen for:")
            print("  - Speech clarity")
            print("  - Tone/emotion preservation")
            print("  - Background noise handling")


if __name__ == "__main__":
    main()