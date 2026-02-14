"""
Training script for hybrid Stage 0 model.

This is MUCH simpler than the diffusion approach:
- Direct classification (no denoising loops)
- Binary cross-entropy loss (vs MSE on noise prediction)
- Single forward pass per sample (vs 50 diffusion steps)
- Faster training (~2 min vs 10 min for 20 epochs)

Expected results:
- Train loss: 0.1-0.3 (lower is better)
- Val accuracy: 0.85-0.95 (vs 0.50 for diffusion!)
- Training time: ~2 minutes for 20 epochs

Why this works better:
- Acoustic features directly capture sound/silence (3x stronger signal)
- Model learns which features matter via attention
- No need for complex diffusion reverse process
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from ambient_subconscious.training.hybrid_classifier_stage0 import HybridClassifierStage0


class HybridDataset(Dataset):
    """Load .pt dataset for training"""

    def __init__(self, pt_file):
        print(f"Loading dataset from {pt_file}...")
        self.data = torch.load(pt_file)

        stats = self.data['stats']
        print(f"  Loaded {stats['num_samples']} samples")
        print(f"    Positive (has_sound=true): {int(stats['num_positive'])}")
        print(f"    Negative (has_sound=false): {int(stats['num_negative'])}")
        print(f"    Ratio: {stats['num_positive']/stats['num_negative']:.2f}:1")

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):
        return {
            'tokens': self.data['tokens'][idx],
            'acoustic': self.data['acoustic'][idx],
            'label': self.data['labels'][idx]
        }


def train_epoch(model, dataloader, optimizer, device, class_weights=None, epoch=0):
    """
    Train for one epoch.

    This is much simpler than diffusion training:
    - No noise schedule
    - No timesteps
    - Direct classification loss
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.BCELoss(reduction='none')  # Binary cross-entropy

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch in pbar:
        tokens = batch['tokens'].to(device)
        acoustic = batch['acoustic'].to(device)
        labels = batch['label'].to(device)

        # Forward pass (single step!)
        predictions = model(tokens, acoustic)

        # Compute loss
        loss_per_sample = criterion(predictions, labels)

        # Class weighting (balance positive/negative samples)
        if class_weights is not None:
            weights = torch.where(
                labels > 0.5,
                class_weights[1],  # Positive weight
                class_weights[0]   # Negative weight
            )
            loss = (loss_per_sample * weights).mean()
        else:
            loss = loss_per_sample.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track statistics
        total_loss += loss.item()

        # Accuracy
        pred_labels = (predictions > 0.5).float()
        correct += (pred_labels == labels).sum().item()
        total += labels.shape[0]

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.3f}'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, dataloader, device, epoch=0):
    """
    Validation (much faster than diffusion - single forward pass!)
    """
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    # Track attention weights to see which features model uses
    attention_weights_list = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

        for batch in pbar:
            tokens = batch['tokens'].to(device)
            acoustic = batch['acoustic'].to(device)
            labels = batch['label'].to(device)

            # Predict (single forward pass!)
            predictions, attention_weights = model(
                tokens,
                acoustic,
                return_attention=True
            )

            # Threshold at 0.5
            pred_labels = (predictions > 0.5).float()

            # Accuracy
            correct += (pred_labels == labels).sum().item()
            total += labels.shape[0]

            # Store for later analysis
            all_probs.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            attention_weights_list.append(attention_weights.cpu())

            pbar.set_postfix({'accuracy': f'{correct/total:.3f}'})

    accuracy = correct / total

    # Compute confusion matrix metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    pred_labels = (all_probs > 0.5).astype(float)

    true_pos = ((pred_labels == 1) & (all_labels == 1)).sum()
    true_neg = ((pred_labels == 0) & (all_labels == 0)).sum()
    false_pos = ((pred_labels == 1) & (all_labels == 0)).sum()
    false_neg = ((pred_labels == 0) & (all_labels == 1)).sum()

    # Precision, recall, F1
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Average attention on acoustics
    all_attention = torch.cat(attention_weights_list)
    mean_attention = all_attention.mean().item()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_pos': int(true_pos),
        'true_neg': int(true_neg),
        'false_pos': int(false_pos),
        'false_neg': int(false_neg),
        'mean_acoustic_attention': mean_attention
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train hybrid Stage 0 model"
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/training/stage0_hybrid.pt',
        help='Path to .pt dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/hybrid_stage0',
        help='Output directory for trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        help='Hidden dimension'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of transformer layers'
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print("TRAINING HYBRID STAGE 0 MODEL")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}\n")

    # Load dataset
    dataset = HybridDataset(args.data)

    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    print(f"\nSplit:")
    print(f"  Train: {train_size} samples")
    print(f"  Val: {val_size} samples\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )

    # Create model
    stats = dataset.data['stats']
    model = HybridClassifierStage0(
        vocab_size=stats['token_vocab_size'],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        acoustic_dim=stats['acoustic_feature_dim'],
        max_seq_len=stats['max_seq_len']
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")

    # Class weights for balanced training
    num_pos = stats['num_positive']
    num_neg = stats['num_negative']
    total = num_pos + num_neg

    weight_pos = total / (2 * num_pos)
    weight_neg = total / (2 * num_neg)

    class_weights = torch.tensor([weight_neg, weight_pos], device=device)

    print(f"Class weights (to balance training):")
    print(f"  Negative weight: {weight_neg:.2f}")
    print(f"  Positive weight: {weight_pos:.2f}")
    print(f"  (Higher weight = more important during training)\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01  # L2 regularization
    )

    # Learning rate scheduler (reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize accuracy
        factor=0.5,
        patience=3
    )

    # Training loop
    print(f"{'='*70}")
    print("TRAINING")
    print(f"{'='*70}\n")

    best_accuracy = 0.0
    best_f1 = 0.0

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights=class_weights,
            epoch=epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, device, epoch=epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Accuracy: {train_acc:.3f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"  Val Precision: {val_metrics['precision']:.3f}")
        print(f"  Val Recall: {val_metrics['recall']:.3f}")
        print(f"  Val F1: {val_metrics['f1']:.3f}")
        print(f"  Confusion Matrix:")
        print(f"    TP={val_metrics['true_pos']}, TN={val_metrics['true_neg']}")
        print(f"    FP={val_metrics['false_pos']}, FN={val_metrics['false_neg']}")
        print(f"  Mean Acoustic Attention: {val_metrics['mean_acoustic_attention']:.3f}")
        print(f"    (Higher = model trusts acoustic features more)")

        # Step scheduler
        scheduler.step(val_metrics['accuracy'])

        # Save if best so far
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            best_f1 = val_metrics['f1']

            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'class_weights': class_weights,
            }, output_dir / 'model.pt')

            print(f"  [SAVED] New best model (accuracy: {val_metrics['accuracy']:.3f}, F1: {val_metrics['f1']:.3f})")

    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}\n")
    print(f"Best validation accuracy: {best_accuracy:.3f}")
    print(f"Best F1 score: {best_f1:.3f}")
    print(f"Model saved to: {Path(args.output) / 'model.pt'}")

    print(f"\nExpected results:")
    print(f"  Train loss: 0.1-0.3 (lower is better)")
    print(f"  Val accuracy: 0.85-0.95 (MUCH better than diffusion's 0.50!)")
    print(f"  F1 score: 0.85-0.95 (balanced precision/recall)")

    print(f"\nIf accuracy is low (<0.7):")
    print(f"  - Check dataset quality (inspect_dataset.py)")
    print(f"  - Increase hidden_dim or num_layers")
    print(f"  - Try different learning rate")
    print(f"  - Check class balance")


if __name__ == '__main__':
    main()
