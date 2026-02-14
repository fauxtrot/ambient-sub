"""
Training script for Stage 0 diffusion model.

This trains your first diffusion model to detect has_sound (binary classification).

WHAT HAPPENS DURING TRAINING:
    1. Load tokenized dataset (enrichment + synthetic silence)
    2. Split into train/validation sets (80/20)
    3. For each training sample:
       a. Add random noise to the ground truth label
       b. Model predicts what noise was added
       c. Compute loss (how wrong was the prediction?)
       d. Update model weights to reduce loss
    4. Validate on held-out data
    5. Save the trained model

FOR FIRST-TIME ML BUILDERS:
    - Epoch = one pass through entire dataset
    - Batch = small group of samples processed together
    - Loss = how wrong the model is (lower = better)
    - Optimizer = algorithm that updates model weights
    - Validation = test on data model hasn't seen
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from ambient_subconscious.training.diffusion_model_stage0 import TemporalDiffusionStage0


class DiffusionDataset(Dataset):
    """
    Dataset for diffusion training.

    This handles:
    - Loading tokenized samples
    - Padding/truncating tokens to fixed length
    - Creating temporal context (random for now, will be real history later)
    - Providing labels for training
    """

    def __init__(self, data_path, max_seq_len=100, temporal_window=5):
        """
        Load tokenized dataset.

        Args:
            data_path: Path to tokenized_dataset.json
            max_seq_len: Maximum sequence length (pad/truncate to this)
            temporal_window: Number of previous frames (context size)
        """
        with open(data_path) as f:
            self.samples = json.load(f)

        self.max_seq_len = max_seq_len
        self.temporal_window = temporal_window

        print(f"Loaded {len(self.samples)} samples")
        print(f"  Positive (has_sound=true): {sum(1 for s in self.samples if s['has_sound'])}")
        print(f"  Negative (has_sound=false): {sum(1 for s in self.samples if not s['has_sound'])}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. PREPARE TOKENS (pad or truncate to max_seq_len)
        tokens = sample['tokens'][:self.max_seq_len]

        # Pad with zeros if too short
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))

        # 2. PREPARE TEMPORAL CONTEXT
        # For now, use random values (simulates random previous predictions)
        # In real self-feeding, this would be model's actual previous predictions
        # We'll implement that later - for now, just learn the task
        temporal_context = np.random.rand(self.temporal_window).astype(np.float32)

        # 3. GROUND TRUTH LABEL
        has_sound = float(sample['has_sound'])

        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "temporal_context": torch.tensor(temporal_context, dtype=torch.float32),
            "has_sound": torch.tensor([has_sound], dtype=torch.float32),
        }


def train_epoch(model, dataloader, optimizer, device, noise_schedule, epoch, class_weights=None):
    """
    Train for one epoch.

    An epoch is one complete pass through the training data.

    Args:
        model: The diffusion model
        dataloader: Provides batches of training data
        optimizer: Updates model weights
        device: 'cuda' or 'cpu'
        noise_schedule: Defines how to add noise
        epoch: Current epoch number (for logging)
        class_weights: Tensor [weight_negative, weight_positive] for class balancing

    Returns:
        Average loss for this epoch
    """
    model.train()  # Put model in training mode
    total_loss = 0
    num_batches = 0

    # Progress bar for visual feedback
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch in pbar:
        # Move data to device (GPU or CPU)
        tokens = batch["tokens"].to(device)
        temporal_context = batch["temporal_context"].to(device)
        labels = batch["has_sound"].to(device)

        batch_size = tokens.shape[0]

        # DIFFUSION TRAINING STEP:
        # 1. Pick random timestep for each sample
        #    (Each sample gets different noise level - this helps model learn all levels)
        t = torch.randint(0, noise_schedule.num_steps, (batch_size,), device=device)

        # 2. Add noise to ground truth labels
        #    This is the FORWARD DIFFUSION PROCESS
        #    We corrupt the clean label with noise
        noisy_labels, noise = noise_schedule.add_noise(labels, t)

        # 3. Model predicts the noise
        #    This is what the model learns: "given noisy input, predict what noise was added"
        predicted_noise = model(tokens, temporal_context, noisy_labels, t)

        # 4. Compute loss: how different is predicted noise from actual noise?
        #    MSE = Mean Squared Error (average of squared differences)
        #    Lower loss = model is better at predicting noise

        # CLASS-WEIGHTED LOSS for imbalanced datasets
        if class_weights is not None:
            # Weight each sample by its class
            # labels: [batch, 1] with values 0.0 or 1.0
            # class_weights: [weight_neg, weight_pos]
            sample_weights = torch.where(
                labels > 0.5,
                class_weights[1],  # Positive class weight
                class_weights[0]   # Negative class weight
            )

            # Compute per-sample MSE loss
            mse_loss = (predicted_noise - noise) ** 2

            # Apply class weights and average
            weighted_loss = (mse_loss * sample_weights).mean()
            loss = weighted_loss
        else:
            # Standard unweighted MSE
            loss = nn.MSELoss()(predicted_noise, noise)

        # 5. BACKPROPAGATION: update model weights to reduce loss
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        # Track statistics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, device):
    """
    Validate model on held-out data.

    Validation checks if the model generalizes to new data.
    We measure ACCURACY: how often does the model predict correctly?

    Args:
        model: The diffusion model
        dataloader: Provides batches of validation data
        device: 'cuda' or 'cpu'

    Returns:
        Accuracy (0.0 to 1.0)
    """
    model.eval()  # Put model in evaluation mode (disables dropout, etc.)
    correct = 0
    total = 0

    with torch.no_grad():  # Don't compute gradients (faster, less memory)
        pbar = tqdm(dataloader, desc="Validation")

        for batch in pbar:
            tokens = batch["tokens"].to(device)
            temporal_context = batch["temporal_context"].to(device)
            labels = batch["has_sound"].to(device)

            # Generate prediction via full denoising (inference mode)
            # This is the REVERSE DIFFUSION PROCESS
            # Start with noise, gradually denoise to get clean prediction
            predictions = model.denoise_sample(
                tokens,
                temporal_context,
                num_steps=10  # Use fewer steps for faster validation (50 is slow)
            )

            # Threshold at 0.5: > 0.5 = has sound, < 0.5 = no sound
            predicted_labels = (predictions > 0.5).float()

            # Count correct predictions
            correct += (predicted_labels == labels).sum().item()
            total += labels.shape[0]

            # Update progress bar
            pbar.set_postfix({"accuracy": f"{correct / total:.3f}"})

    accuracy = correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 0 diffusion model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training/tokenized_dataset.json",
        help="Path to tokenized dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/diffusion_stage0",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=50,
        help="Number of diffusion steps"
    )

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print("TRAINING STAGE 0 DIFFUSION MODEL")
    print(f"{'='*80}\n")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = DiffusionDataset(
        args.data,
        max_seq_len=100,
        temporal_window=5
    )

    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    print(f"\nSplit:")
    print(f"  Train: {train_size} samples")
    print(f"  Val: {val_size} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True  # Shuffle for better training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False  # Don't shuffle validation
    )

    # Create model
    print("\nCreating model...")
    model = TemporalDiffusionStage0(
        vocab_size=1024,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_seq_len=100,
        temporal_window=5,
        num_diffusion_steps=args.diffusion_steps,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Optimizer: AdamW (Adam with weight decay)
    # This is a popular optimizer that adapts learning rates automatically
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # COMPUTE CLASS WEIGHTS for imbalanced dataset
    # Count positive and negative samples in full dataset
    num_positive = sum(1 for s in dataset.samples if s['has_sound'])
    num_negative = len(dataset.samples) - num_positive

    print(f"\nClass distribution:")
    print(f"  Positive (has_sound=true): {num_positive}")
    print(f"  Negative (has_sound=false): {num_negative}")
    print(f"  Ratio (neg:pos): {num_negative/num_positive:.1f}:1")

    # Compute balanced class weights
    # Formula: weight = total_samples / (num_classes * samples_per_class)
    total_samples = len(dataset.samples)
    weight_positive = total_samples / (2 * num_positive) if num_positive > 0 else 1.0
    weight_negative = total_samples / (2 * num_negative) if num_negative > 0 else 1.0

    class_weights = torch.tensor([weight_negative, weight_positive], device=device)

    print(f"\nClass weights (to balance training):")
    print(f"  Negative weight: {weight_negative:.2f}")
    print(f"  Positive weight: {weight_positive:.2f}")
    print(f"  (Higher weight = more important during training)")

    # Training loop
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")

    best_accuracy = 0.0

    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            model.noise_schedule,
            epoch,
            class_weights=class_weights  # Pass class weights for balanced training
        )

        # Validate
        val_accuracy = validate(model, val_loader, device)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.3f}")

        # Save if best so far
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = output_dir / "model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'train_loss': train_loss,
            }, checkpoint_path)

            print(f"  [SAVED] New best model (accuracy: {val_accuracy:.3f})")

    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}\n")
    print(f"Best validation accuracy: {best_accuracy:.3f}")
    print(f"Model saved to: {output_dir / 'model.pt'}")

    # Expected results
    print("\nExpected results for first diffusion model:")
    print("  Train loss: 0.05-0.10 (lower is better)")
    print("  Val accuracy: 0.85-0.95 (higher is better)")
    print("\nIf accuracy is low (<0.7), possible issues:")
    print("  - Dataset too small (need more samples)")
    print("  - Class imbalance (too many negatives vs positives)")
    print("  - Model too simple (increase hidden_dim or num_layers)")
    print("  - Learning rate too high/low (try 0.0001 or 0.01)")


if __name__ == "__main__":
    main()
