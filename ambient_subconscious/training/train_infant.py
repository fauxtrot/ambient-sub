"""Training script for Infant Model Stage 0"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm
import json

from .infant_model import InfantModelStage0
from .data_pipeline import SessionLoader, EncodecTokenizer, TrainingSampleGenerator
from .label_pipeline import LabelPipeline, AudioFeatureLabelSource


class Stage0Dataset(Dataset):
    """Dataset for Stage 0 training"""

    def __init__(self, samples, max_seq_len=100):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Pad/truncate tokens to max_seq_len
        tokens = sample.encodec_tokens[:self.max_seq_len]
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))

        # Get ground truth
        has_sound = float(sample.ground_truth.get("has_sound", False))

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "has_sound": torch.tensor([has_sound], dtype=torch.float32),
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        tokens = batch["tokens"].to(device)
        labels = batch["has_sound"].to(device)

        # Forward
        optimizer.zero_grad()
        predictions = model(tokens)

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        predicted_labels = (predictions > 0.5).float()
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            tokens = batch["tokens"].to(device)
            labels = batch["has_sound"].to(device)

            # Forward
            predictions = model(tokens)

            # Compute loss
            loss = criterion(predictions, labels)

            # Stats
            total_loss += loss.item()
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Infant Model Stage 0")
    parser.add_argument("--data-dir", type=str, default="data/sessions", help="Session data directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    print("=" * 60)
    print("INFANT MODEL - STAGE 0 TRAINING")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Load sessions
    print("\nLoading sessions...")
    loader = SessionLoader(args.data_dir)
    session_ids = loader.list_sessions()

    if not session_ids:
        print(f"No sessions found in {args.data_dir}")
        print("Please record some sessions first using test_with_audio_recording.py")
        return

    print(f"Found {len(session_ids)} sessions")

    # Create label pipeline
    print("\nCreating label pipeline...")
    pipeline = LabelPipeline([
        AudioFeatureLabelSource(threshold=0.01)
    ])

    # Create tokenizer
    print("Loading Encodec tokenizer...")
    tokenizer = EncodecTokenizer(bandwidth=6.0)

    # Generate training samples
    print("\nGenerating training samples...")
    generator = TrainingSampleGenerator(tokenizer)

    all_samples = []
    for session_id in tqdm(session_ids, desc="Processing sessions"):
        try:
            session = loader.load_session(session_id)
            ground_truth = pipeline.generate_ground_truth(session)
            samples = generator.generate_samples(session, ground_truth)
            all_samples.extend(samples)
        except Exception as e:
            print(f"Error processing {session_id}: {e}")
            continue

    print(f"Generated {len(all_samples)} training samples")

    if len(all_samples) == 0:
        print("No samples generated. Check your sessions have audio data.")
        return

    # Split train/val
    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    # Create datasets
    train_dataset = Stage0Dataset(train_samples)
    val_dataset = Stage0Dataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    print("\nCreating model...")
    model = InfantModelStage0(
        vocab_size=1024,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        max_seq_len=100,
    ).to(args.device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(args.output_dir) / "stage0_best.pt"
            model.save(str(output_path))
            print(f"Saved best model (val_acc={val_acc:.4f}) to {output_path}")

    # Save final model
    final_path = Path(args.output_dir) / "stage0_final.pt"
    model.save(str(final_path))

    # Save training info
    info_path = Path(args.output_dir) / "stage0_info.json"
    with open(info_path, 'w') as f:
        json.dump({
            "epochs": args.epochs,
            "final_val_accuracy": float(val_acc),
            "best_val_accuracy": float(best_val_acc),
            "num_training_samples": len(train_samples),
            "num_validation_samples": len(val_samples),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
