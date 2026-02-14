"""Train the SNN classifier.

Supports two modes:
- binary: sound/silence (legacy 8-feature, 2-class)
- speaker: multi-class speaker ID (37-feature, N-class)
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from model import SoundSNN
from dataset import load_data, load_speaker_data, NUM_STEPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
LR = 1e-3
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for spikes, labels in loader:
        spikes = spikes.permute(1, 0, 2).to(device)
        labels = labels.to(device)

        spk_rec, _ = model(spikes)
        spike_count = spk_rec.sum(dim=0)
        loss = criterion(spike_count, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (spike_count.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    per_class = {}

    with torch.no_grad():
        for spikes, labels in loader:
            spikes = spikes.permute(1, 0, 2).to(device)
            labels = labels.to(device)

            spk_rec, _ = model(spikes)
            spike_count = spk_rec.sum(dim=0)
            preds = spike_count.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for cls in labels.unique().tolist():
                mask = labels == cls
                cls_correct = (preds[mask] == labels[mask]).sum().item()
                cls_total = mask.sum().item()
                if cls not in per_class:
                    per_class[cls] = {"correct": 0, "total": 0}
                per_class[cls]["correct"] += cls_correct
                per_class[cls]["total"] += cls_total

    return correct / total, per_class


def train():
    parser = argparse.ArgumentParser(description="Train SNN classifier")
    parser.add_argument("--mode", choices=["binary", "speaker"], default="speaker",
                        help="Training mode: binary (sound/silence) or speaker (multi-class)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Mode: {args.mode}")

    if args.mode == "speaker":
        train_loader, val_loader, class_weights, norm_stats, class_map = load_speaker_data(
            num_steps=NUM_STEPS
        )
        num_inputs = 37
        hidden_size = 128
        num_outputs = len(class_map)
        print(f"Classes ({num_outputs}): {class_map}")
    else:
        train_loader, val_loader, class_weights, norm_stats = load_data(num_steps=NUM_STEPS)
        num_inputs = 8
        hidden_size = 64
        num_outputs = 2
        class_map = {0: "silence", 1: "sound"}

    class_weights = class_weights.to(DEVICE)

    model = SoundSNN(num_inputs=num_inputs, hidden_size=hidden_size,
                     num_outputs=num_outputs, beta=0.95, num_steps=NUM_STEPS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        avg_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_acc, per_class = validate(model, val_loader, DEVICE)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | "
                  f"Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

            suffix = f"_{args.mode}" if args.mode == "speaker" else ""
            model_file = f"best_model{suffix}.pt"
            stats_file = f"norm_stats{suffix}.pt"

            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, model_file))
            torch.save(norm_stats,
                       os.path.join(CHECKPOINT_DIR, stats_file))

            if args.mode == "speaker":
                meta = {
                    "class_map": class_map,
                    "num_classes": num_outputs,
                    "num_inputs": num_inputs,
                    "hidden_size": hidden_size,
                }
                with open(os.path.join(CHECKPOINT_DIR, "speaker_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")

    # Print per-class breakdown from last validation
    if per_class:
        print("\nPer-class accuracy:")
        for cls_idx, stats in sorted(per_class.items()):
            name = class_map.get(cls_idx, class_map.get(str(cls_idx), f"class_{cls_idx}"))
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {name}: {acc:.3f} ({stats['correct']}/{stats['total']})")

    print(f"Model saved to {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    train()
