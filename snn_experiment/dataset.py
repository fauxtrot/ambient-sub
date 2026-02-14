"""Dataset loading and rate encoding for the SNN experiment.

Supports both:
- Legacy binary sound/silence (stage0_hybrid.pt)
- Multi-class speaker training (speaker_training.pt)
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training", "stage0_hybrid.pt")
SPEAKER_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "speaker_training.pt")
MIC_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NUM_STEPS = 25


class AcousticSNNDataset(Dataset):
    """Wraps features + labels for SNN training."""

    def __init__(self, features, labels, num_steps=NUM_STEPS):
        self.features = features  # [N, D]
        self.labels = labels      # [N] long
        self.num_steps = num_steps

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.features[idx]  # [D]
        probs = feat.clamp(0.0, 1.0)
        spikes = torch.rand(self.num_steps, feat.shape[0]) < probs.unsqueeze(0)
        return spikes.float(), self.labels[idx]


def load_speaker_data(data_path=SPEAKER_DATA_PATH, num_steps=NUM_STEPS,
                      val_ratio=0.2, seed=42):
    """Load multi-class speaker training data.

    Returns:
        train_loader, val_loader, class_weights, norm_stats, class_map
    """
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    features = data["features"]  # [N, 37]
    labels = data["labels"]      # [N] long
    class_map = data["class_map"]
    num_classes = data["num_classes"]

    # Z-score normalize then sigmoid
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp(min=1e-8)
    features_norm = (features - mean) / std
    features_prob = torch.sigmoid(features_norm)

    # Stratified split
    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio, random_state=seed,
        stratify=labels.numpy()
    )

    train_ds = AcousticSNNDataset(features_prob[train_idx], labels[train_idx], num_steps)
    val_ds = AcousticSNNDataset(features_prob[val_idx], labels[val_idx], num_steps)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Class weights: inverse frequency
    counts = torch.zeros(num_classes)
    for i in range(num_classes):
        counts[i] = (labels == i).sum().float()
    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))

    norm_stats = {"mean": mean, "std": std}

    return train_loader, val_loader, weights, norm_stats, class_map


def load_data(data_path=DATA_PATH, num_steps=NUM_STEPS, val_ratio=0.2, seed=42,
              include_mic_data=True):
    """Load binary sound/silence dataset (legacy).

    Returns:
        train_loader, val_loader, class_weights, norm_stats
    """
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    acoustic = data["acoustic"]  # [N, 8]
    labels = data["labels"].squeeze(-1).long()  # [N]

    if include_mic_data and os.path.isdir(MIC_DATA_DIR):
        mic_files = glob.glob(os.path.join(MIC_DATA_DIR, "mic_labeled_*.pt"))
        for mf in mic_files:
            mic_data = torch.load(mf, map_location="cpu", weights_only=False)
            mic_acoustic = mic_data["acoustic"]
            mic_labels = mic_data["labels"].squeeze(-1).long()
            acoustic = torch.cat([acoustic, mic_acoustic], dim=0)
            labels = torch.cat([labels, mic_labels], dim=0)
            print(f"  Loaded {len(mic_labels)} mic samples from {os.path.basename(mf)}")

    mean = acoustic.mean(dim=0)
    std = acoustic.std(dim=0).clamp(min=1e-8)
    acoustic_norm = (acoustic - mean) / std
    acoustic_prob = torch.sigmoid(acoustic_norm)

    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio, random_state=seed,
        stratify=labels.numpy()
    )

    train_ds = AcousticSNNDataset(acoustic_prob[train_idx], labels[train_idx], num_steps)
    val_ds = AcousticSNNDataset(acoustic_prob[val_idx], labels[val_idx], num_steps)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    num_pos = (labels == 1).sum().float()
    num_neg = (labels == 0).sum().float()
    total = num_pos + num_neg
    weights = torch.tensor([total / (2 * num_neg), total / (2 * num_pos)])

    norm_stats = {"mean": mean, "std": std}

    return train_loader, val_loader, weights, norm_stats
