"""Extract pyannote embeddings + MFCCs from labeled sessions to build training data.

Usage:
    python extract_embeddings.py
    python extract_embeddings.py --sessions-dir path/to/sessions
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
import pickle
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from features import (
    extract_acoustic, extract_mfccs, extract_pyannote_embedding,
    SAMPLE_RATE
)

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sessions")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
PCA_COMPONENTS = 16


def load_embedding_model():
    """Load pyannote speaker embedding model."""
    from pyannote.audio import Model, Inference
    model = Model.from_pretrained("pyannote/embedding")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inference = Inference(model, window="whole")
    print(f"Embedding model loaded on {device}")
    return inference


def find_labeled_sessions(sessions_dir: str) -> list:
    """Find sessions that have enriched_training_data.json with canonical speaker names."""
    sessions = []
    for entry in os.listdir(sessions_dir):
        session_dir = os.path.join(sessions_dir, entry)
        if not os.path.isdir(session_dir):
            continue

        enriched_path = os.path.join(session_dir, "enriched_training_data.json")
        audio_path = os.path.join(session_dir, "audio.wav")

        if not os.path.exists(enriched_path) or not os.path.exists(audio_path):
            continue

        with open(enriched_path, "r") as f:
            segments = json.load(f)

        if not segments:
            continue

        # Check if segments have canonical speaker names (not just speakerN)
        speaker_ids = set(s["speaker_id"] for s in segments if s.get("has_speaker"))
        has_named = any(not sid.startswith("speaker") and sid != "unknown"
                        for sid in speaker_ids)

        sessions.append({
            "dir": session_dir,
            "segments": segments,
            "audio_path": audio_path,
            "speaker_ids": speaker_ids,
            "has_named_speakers": has_named,
        })

    return sessions


def extract_from_session(session: dict, inference) -> tuple:
    """Extract features and embeddings from a single session.

    Returns:
        features_list: list of [21] arrays (acoustic + MFCC + reserved)
        embeddings_list: list of [192] arrays (raw pyannote embeddings)
        labels_list: list of speaker_id strings
    """
    audio, sr = sf.read(session["audio_path"], dtype="float32")
    if sr != SAMPLE_RATE:
        # Resample if needed
        from scipy.signal import resample
        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, num_samples).astype(np.float32)
        sr = SAMPLE_RATE

    # Handle stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    features_list = []
    embeddings_list = []
    labels_list = []

    segments = [s for s in session["segments"] if s.get("has_sound", False)]

    for i, seg in enumerate(segments):
        speaker_id = seg.get("speaker_id", "unknown")
        if speaker_id == "unknown":
            continue

        start_sample = int(seg["start_time"] * sr)
        end_sample = int(seg["end_time"] * sr)
        chunk = audio[start_sample:end_sample]

        if len(chunk) < 1600:  # < 0.1s
            continue

        # Acoustic + MFCC features
        acoustic = extract_acoustic(chunk, sr)
        mfccs = extract_mfccs(chunk, sr)
        reserved = np.zeros(3, dtype=np.float32)
        feat = np.concatenate([acoustic, mfccs, reserved])

        # Pyannote embedding
        try:
            embedding = extract_pyannote_embedding(chunk, sr, inference)
        except Exception as e:
            print(f"  Warning: embedding failed for {seg['segment_id']}: {e}")
            continue

        features_list.append(feat)
        embeddings_list.append(embedding)
        labels_list.append(speaker_id)

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(segments)} segments...")

    return features_list, embeddings_list, labels_list


def build_silence_samples(sessions: list, num_samples: int = 200) -> tuple:
    """Generate silence samples from gaps between segments.

    Returns:
        features_list, embeddings_list (zeros), labels_list ("silence")
    """
    features_list = []
    embeddings_list = []

    for session in sessions:
        audio, sr = sf.read(session["audio_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample
            num_samples_audio = int(len(audio) * SAMPLE_RATE / sr)
            audio = resample(audio, num_samples_audio).astype(np.float32)
            sr = SAMPLE_RATE

        segments = sorted(session["segments"], key=lambda s: s["start_time"])
        chunk_duration = 1.0
        chunk_samples = int(sr * chunk_duration)

        # Find gaps
        prev_end = 0.0
        for seg in segments:
            gap_start = prev_end
            gap_end = seg["start_time"]
            if gap_end - gap_start >= chunk_duration:
                # Sample from this gap
                start_sample = int(gap_start * sr)
                end_sample = start_sample + chunk_samples
                if end_sample <= len(audio):
                    chunk = audio[start_sample:end_sample]
                    acoustic = extract_acoustic(chunk, sr)
                    mfccs = extract_mfccs(chunk, sr)
                    reserved = np.zeros(3, dtype=np.float32)
                    feat = np.concatenate([acoustic, mfccs, reserved])
                    features_list.append(feat)
                    # Silence has no speaker embedding — use zeros
                    embeddings_list.append(np.zeros(192, dtype=np.float32))

                    if len(features_list) >= num_samples:
                        return features_list, embeddings_list

            prev_end = seg["end_time"]

    return features_list, embeddings_list


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from labeled sessions")
    parser.add_argument("--sessions-dir", default=SESSIONS_DIR)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pca-components", type=int, default=PCA_COMPONENTS)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(OUTPUT_DIR, "speaker_training.pt")

    print("Finding labeled sessions...")
    sessions = find_labeled_sessions(args.sessions_dir)
    if not sessions:
        print("No labeled sessions found. Label sessions in the session editor first.")
        return

    print(f"Found {len(sessions)} session(s):")
    for s in sessions:
        named = "named" if s["has_named_speakers"] else "cluster IDs only"
        print(f"  {os.path.basename(s['dir'])}: {len(s['segments'])} segments, "
              f"speakers: {s['speaker_ids']} ({named})")

    # Load embedding model
    print("\nLoading pyannote embedding model...")
    inference = load_embedding_model()

    # Extract from all sessions
    all_features = []
    all_embeddings = []
    all_labels = []

    for session in sessions:
        print(f"\nProcessing {os.path.basename(session['dir'])}...")
        features, embeddings, labels = extract_from_session(session, inference)
        all_features.extend(features)
        all_embeddings.extend(embeddings)
        all_labels.extend(labels)
        print(f"  Extracted {len(features)} samples")

    # Add silence samples
    print("\nExtracting silence samples...")
    silence_features, silence_embeddings = build_silence_samples(sessions)
    all_features.extend(silence_features)
    all_embeddings.extend(silence_embeddings)
    all_labels.extend(["silence"] * len(silence_features))
    print(f"  Added {len(silence_features)} silence samples")

    if not all_features:
        print("No samples extracted.")
        return

    # Build class map
    unique_speakers = sorted(set(all_labels))
    class_map = {i: name for i, name in enumerate(unique_speakers)}
    label_to_idx = {name: i for i, name in class_map.items()}
    print(f"\nClass map: {class_map}")

    # Convert to arrays
    features_array = np.stack(all_features)  # [N, 21]
    embeddings_array = np.stack(all_embeddings)  # [N, 192]
    labels_array = np.array([label_to_idx[l] for l in all_labels])

    # Fit PCA on embeddings (excluding silence zeros)
    non_silence_mask = labels_array != label_to_idx.get("silence", -1)
    non_silence_embeddings = embeddings_array[non_silence_mask]

    n_components = min(args.pca_components, non_silence_embeddings.shape[0],
                       non_silence_embeddings.shape[1])
    print(f"\nFitting PCA with {n_components} components on {len(non_silence_embeddings)} embeddings...")
    pca = PCA(n_components=n_components)
    pca.fit(non_silence_embeddings)
    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"  PCA explains {explained:.1f}% of variance")

    # Transform all embeddings
    embeddings_pca = pca.transform(embeddings_array).astype(np.float32)  # [N, 16]

    # Concatenate: [21 basic] + [16 PCA embedding] = [37]
    full_features = np.concatenate([features_array, embeddings_pca], axis=1)
    print(f"\nFull feature shape: {full_features.shape}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data = {
        "features": torch.tensor(full_features, dtype=torch.float32),
        "labels": torch.tensor(labels_array, dtype=torch.long),
        "class_map": class_map,
        "num_classes": len(class_map),
        "feature_dim": full_features.shape[1],
        "stats": {
            "num_samples": len(labels_array),
            "per_class": {name: int((labels_array == idx).sum())
                          for idx, name in class_map.items()},
        },
    }
    torch.save(data, args.output)
    print(f"\nSaved {len(labels_array)} samples to {args.output}")
    for idx, name in class_map.items():
        count = int((labels_array == idx).sum())
        print(f"  {name}: {count}")

    # Save PCA model separately
    pca_path = os.path.join(os.path.dirname(args.output), "pca_model.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model to {pca_path}")

    # Save class map separately for easy loading
    class_map_path = os.path.join(os.path.dirname(args.output), "class_map.json")
    with open(class_map_path, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"Saved class map to {class_map_path}")


if __name__ == "__main__":
    main()
