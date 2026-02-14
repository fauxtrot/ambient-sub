"""Live SNN inference listener — captures mic audio and classifies speakers."""

import argparse
import json
import signal
import sys
import time
import os
import pickle

import numpy as np
import sounddevice as sd
import torch
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from model import SoundSNN
from features import extract_features_basic, extract_features_full, extract_pyannote_embedding

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0  # seconds
NUM_STEPS = 25
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
API_URL = "http://localhost:5174/api/model-state"

running = True


def signal_handler(sig, frame):
    global running
    running = False


def list_devices():
    print(sd.query_devices())


class AdaptiveNormalizer:
    """EMA-based normalizer that starts from training stats and adapts to live mic."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, alpha: float = 0.01):
        self.mean = mean.copy()
        self.std = std.copy()
        self.alpha = alpha
        self.n = 0

    def normalize(self, features: np.ndarray) -> np.ndarray:
        self.n += 1
        if self.n > 10:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * features
            diff = features - self.mean
            self.std = (1 - self.alpha) * self.std + self.alpha * np.abs(diff)

        z = (features - self.mean) / (self.std + 1e-8)
        return 1.0 / (1.0 + np.exp(-z))  # sigmoid


def rate_encode(probs: np.ndarray, num_steps: int) -> torch.Tensor:
    probs_t = torch.tensor(probs, dtype=torch.float32)
    spikes = torch.rand(num_steps, len(probs)) < probs_t.unsqueeze(0)
    return spikes.float()


def post_state(result: dict, url: str = API_URL):
    """POST classification result to SvelteKit API."""
    try:
        data = json.dumps(result).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass


def load_speaker_mode():
    """Load speaker model, PCA, and metadata. Returns None if not available."""
    meta_path = os.path.join(CHECKPOINT_DIR, "speaker_meta.json")
    model_path = os.path.join(CHECKPOINT_DIR, "best_model_speaker.pt")
    stats_path = os.path.join(CHECKPOINT_DIR, "norm_stats_speaker.pt")
    pca_path = os.path.join(DATA_DIR, "pca_model.pkl")

    if not all(os.path.exists(p) for p in [meta_path, model_path, stats_path, pca_path]):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    model = SoundSNN(
        num_inputs=meta["num_inputs"],
        hidden_size=meta["hidden_size"],
        num_outputs=meta["num_classes"],
        beta=0.95, num_steps=NUM_STEPS
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    norm_stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    with open(pca_path, "rb") as f:
        pca_model = pickle.load(f)

    # Load pyannote embedding model
    from pyannote.audio import Inference as PyannoteInference
    embedding_inference = PyannoteInference("pyannote/embedding", window="whole")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_inference.to(device)

    return {
        "model": model,
        "norm_stats": norm_stats,
        "pca_model": pca_model,
        "embedding_inference": embedding_inference,
        "class_map": meta["class_map"],
    }


def load_binary_mode():
    """Load legacy binary sound/silence model."""
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    stats_path = os.path.join(CHECKPOINT_DIR, "norm_stats.pt")

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        return None

    model = SoundSNN(num_inputs=8, hidden_size=64, num_outputs=2,
                     beta=0.95, num_steps=NUM_STEPS)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    norm_stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    return {
        "model": model,
        "norm_stats": norm_stats,
        "class_map": {0: "silence", 1: "sound"},
    }


def main():
    global running

    parser = argparse.ArgumentParser(description="SNN listener")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument("--url", default=API_URL)
    parser.add_argument("--mode", choices=["auto", "binary", "speaker"], default="auto",
                        help="Model mode (auto tries speaker first, falls back to binary)")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load model
    mode_data = None
    active_mode = args.mode

    if active_mode in ("auto", "speaker"):
        mode_data = load_speaker_mode()
        if mode_data:
            active_mode = "speaker"
            print(f"Speaker model loaded: {mode_data['class_map']}")

    if mode_data is None:
        if active_mode == "speaker":
            print("Error: Speaker model not found. Run extract_embeddings.py then train.py --mode speaker.")
            sys.exit(1)
        mode_data = load_binary_mode()
        active_mode = "binary"

    if mode_data is None:
        print("Error: No model found. Run train.py first.")
        sys.exit(1)

    model = mode_data["model"]
    class_map = mode_data["class_map"]
    normalizer = AdaptiveNormalizer(
        mode_data["norm_stats"]["mean"].numpy(),
        mode_data["norm_stats"]["std"].numpy(),
        alpha=0.01
    )

    if args.verbose:
        print(f"Mode: {active_mode} | Device: {args.device or 'default'}")
        print(f"Chunk: {CHUNK_DURATION}s @ {SAMPLE_RATE}Hz | Steps: {NUM_STEPS}")
        print("-" * 50)

    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

    while running:
        try:
            audio = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=1,
                           dtype="float32", device=args.device)
            sd.wait()

            if not running:
                break

            audio = audio.flatten()

            # Extract features based on mode
            if active_mode == "speaker":
                embedding = extract_pyannote_embedding(
                    audio, SAMPLE_RATE, mode_data["embedding_inference"]
                )
                features = extract_features_full(
                    audio, SAMPLE_RATE, embedding, mode_data["pca_model"]
                )
            else:
                features = extract_features_basic(audio, SAMPLE_RATE)

            probs = normalizer.normalize(features)
            spikes = rate_encode(probs, NUM_STEPS)
            spikes = spikes.unsqueeze(1)  # [num_steps, 1, D]

            with torch.no_grad():
                spk_rec, _ = model(spikes)
                spike_count = spk_rec.sum(dim=0).squeeze(0)
                pred_idx = spike_count.argmax().item()
                pred_label = class_map.get(pred_idx, class_map.get(str(pred_idx), "unknown"))
                confidence = float(spike_count[pred_idx] / (spike_count.sum() + 1e-8))

            is_sound = pred_label != "silence"

            if args.verbose:
                rms = float(np.sqrt(np.mean(audio ** 2)))
                spk_str = ", ".join(f"{v:.0f}" for v in spike_count.tolist())
                print(f"{pred_label:>12} | conf: {confidence:.2f} | RMS: {rms:.4f} | [{spk_str}]")

            if not args.no_post:
                post_state({
                    "is_sound": is_sound,
                    "speaker": pred_label if is_sound else None,
                    "confidence": confidence,
                })

        except sd.PortAudioError as e:
            print(f"Audio error: {e}")
            time.sleep(1)
        except Exception as e:
            if running:
                print(f"Error: {e}")
                time.sleep(1)

    if args.verbose:
        print("\nListener stopped.")


if __name__ == "__main__":
    main()
