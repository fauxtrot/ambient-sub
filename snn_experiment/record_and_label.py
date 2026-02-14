"""Record mic audio, then play back and label each chunk as sound/silence.

Usage:
    python record_and_label.py --device 0 --duration 120

Step 1: Records audio from mic for --duration seconds
Step 2: Plays back each 1-second chunk, shows RMS, auto-labels via threshold
Step 3: You correct any wrong labels with keypress
Step 4: Saves features + labels as a .pt file for training
"""

import argparse
import os
import sys
import time

import numpy as np
import sounddevice as sd
import torch
from scipy import signal as scipy_signal

sys.path.insert(0, os.path.dirname(__file__))

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RMS_THRESHOLD = 0.005  # auto-label threshold — chunks above this are "sound"


def set_threshold(value):
    global RMS_THRESHOLD
    RMS_THRESHOLD = value

READING_PASSAGE = """
The room was quiet except for the hum of the computer fan.

[PAUSE 5 SECONDS - stay silent]

She picked up the phone and said, "Hello? Yes, I can hear you
clearly. The connection sounds fine on my end." She paused to
listen, then continued. "No, I think we should meet on Thursday
instead. Wednesday doesn't work for me."

[PAUSE 5 SECONDS - stay silent]

Outside, a dog barked twice and then stopped. The wind rattled
the window frame. He cleared his throat and began reading aloud
from the manual: "Step one, verify that all connections are
secure. Step two, power on the device and wait for the green
indicator light."

[PAUSE 5 SECONDS - stay silent]

"Can you repeat that?" she asked. "I didn't quite catch the
last part." He sighed and started over, speaking more slowly
this time. "The activation code is alpha, seven, seven, bravo,
nine, delta."

[PAUSE 5 SECONDS - stay silent]

The meeting ended and the room fell silent again. Only the
soft ticking of the clock remained.

[PAUSE 10 SECONDS - stay completely silent until recording ends]
""".strip()


def extract_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract 8 acoustic features from audio chunk."""
    if len(audio) < 100:
        return np.zeros(8, dtype=np.float32)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    max_amp = float(np.max(np.abs(audio)))
    energy = float(np.sum(audio ** 2) / len(audio))

    zero_crossings = np.where(np.diff(np.sign(audio)))[0]
    zcr = float(len(zero_crossings) / len(audio))

    try:
        f, t, Sxx = scipy_signal.spectrogram(audio, sr)
        sc = np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)
        spectral_centroid = float(np.mean(sc))
    except Exception:
        spectral_centroid = 0.0

    return np.array([rms, max_amp, energy, zcr, spectral_centroid,
                     0.0, 0.0, 0.0], dtype=np.float32)


def record_session(device, duration):
    """Record audio from mic."""
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    num_chunks = int(duration / CHUNK_DURATION)

    print(f"\nRecording {num_chunks} chunks ({duration}s) from device {device or 'default'}...")
    print("Read the passage below out loud. Pause for a few seconds between paragraphs.")
    print("Press Ctrl+C to stop early.\n")
    print("=" * 60)
    print(READING_PASSAGE)
    print("=" * 60)
    print()

    chunks = []
    try:
        for i in range(num_chunks):
            audio = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=1,
                           dtype="float32", device=device)
            sd.wait()
            chunk = audio.flatten()
            rms = np.sqrt(np.mean(chunk ** 2))
            bar = "#" * int(min(rms * 200, 40))
            print(f"  [{i+1:3d}/{num_chunks}] RMS: {rms:.4f} {bar}")
            chunks.append(chunk)
    except KeyboardInterrupt:
        print(f"\nStopped early — captured {len(chunks)} chunks.")

    return chunks


def label_session(chunks):
    """Play back chunks and label them. Auto-labels with RMS threshold, user corrects."""
    features_list = []
    labels_list = []

    print(f"\n{'='*60}")
    print("LABELING PHASE")
    print(f"{'='*60}")
    print(f"Auto-labeling with RMS threshold: {RMS_THRESHOLD}")
    print("For each chunk you'll see the auto-label and can:")
    print("  [Enter] = accept auto-label")
    print("  [s]     = mark as SOUND")
    print("  [n]     = mark as SILENCE (no sound)")
    print("  [p]     = play chunk again")
    print("  [q]     = quit labeling (save what we have)")
    print(f"{'='*60}\n")

    for i, chunk in enumerate(chunks):
        features = extract_features(chunk)
        rms = features[0]
        auto_label = 1 if rms > RMS_THRESHOLD else 0
        auto_str = "SOUND" if auto_label else "silence"

        print(f"Chunk {i+1}/{len(chunks)} | RMS: {rms:.5f} | Auto: {auto_str}")

        # Play the chunk
        sd.play(chunk, samplerate=SAMPLE_RATE)
        sd.wait()

        while True:
            resp = input(f"  Label [{auto_str}]: ").strip().lower()
            if resp == "" :
                label = auto_label
                break
            elif resp == "s":
                label = 1
                break
            elif resp == "n":
                label = 0
                break
            elif resp == "p":
                sd.play(chunk, samplerate=SAMPLE_RATE)
                sd.wait()
            elif resp == "q":
                print(f"\nStopped — labeled {len(labels_list)} chunks.")
                return features_list, labels_list
            else:
                print("  Invalid input. Use Enter/s/n/p/q")

        final_str = "SOUND" if label else "silence"
        changed = " (corrected)" if label != auto_label else ""
        print(f"  -> {final_str}{changed}\n")

        features_list.append(features)
        labels_list.append(label)

    return features_list, labels_list


def save_dataset(features_list, labels_list, output_path):
    """Save as .pt file compatible with training pipeline."""
    acoustic = torch.tensor(np.stack(features_list), dtype=torch.float32)
    labels = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(-1)

    num_pos = int((labels == 1).sum().item())
    num_neg = int((labels == 0).sum().item())

    data = {
        "acoustic": acoustic,
        "labels": labels,
        "tokens": torch.zeros(len(labels_list), 1, dtype=torch.long),  # placeholder
        "stats": {
            "num_samples": len(labels_list),
            "num_positive": num_pos,
            "num_negative": num_neg,
            "token_vocab_size": 1024,
            "max_seq_len": 1,
            "acoustic_feature_dim": 8,
        },
        "metadata": [{"source": "live_mic", "chunk_index": i} for i in range(len(labels_list))],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print(f"\nSaved {len(labels_list)} samples to {output_path}")
    print(f"  Sound: {num_pos} | Silence: {num_neg}")


def main():
    parser = argparse.ArgumentParser(description="Record and label mic audio for SNN training")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--duration", type=int, default=120, help="Recording duration in seconds")
    parser.add_argument("--threshold", type=float, default=RMS_THRESHOLD,
                        help=f"RMS auto-label threshold (default: {RMS_THRESHOLD})")
    parser.add_argument("--output", default=None, help="Output .pt file path")
    args = parser.parse_args()

    set_threshold(args.threshold)

    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(DATA_DIR, f"mic_labeled_{timestamp}.pt")

    # Step 1: Record
    chunks = record_session(args.device, args.duration)
    if not chunks:
        print("No audio recorded.")
        return

    # Step 2: Label
    features_list, labels_list = label_session(chunks)
    if not labels_list:
        print("No labels collected.")
        return

    # Step 3: Save
    save_dataset(features_list, labels_list, args.output)


if __name__ == "__main__":
    main()
