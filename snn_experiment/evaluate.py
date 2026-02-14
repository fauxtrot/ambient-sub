"""Evaluate the trained SNN model and print metrics."""

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model import SoundSNN
from dataset import load_data, NUM_STEPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pt")


def evaluate():
    _, val_loader, _, _ = load_data(num_steps=NUM_STEPS)

    model = SoundSNN(num_inputs=8, hidden_size=64, num_outputs=2,
                     beta=0.95, num_steps=NUM_STEPS).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for spikes, labels in val_loader:
            spikes = spikes.permute(1, 0, 2).to(DEVICE)
            labels = labels.to(DEVICE)

            spk_rec, _ = model(spikes)
            spike_count = spk_rec.sum(dim=0)
            preds = spike_count.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("=== SNN Evaluation Results ===")
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.3f}")
    print(f"Precision: {precision_score(all_labels, all_preds, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, zero_division=0):.3f}")
    print(f"F1 Score:  {f1_score(all_labels, all_preds, zero_division=0):.3f}")
    print()

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(f"              Pred Silence  Pred Sound")
    print(f"  Silence     {cm[0][0]:>6}        {cm[0][1]:>6}")
    print(f"  Sound       {cm[1][0]:>6}        {cm[1][1]:>6}")


if __name__ == "__main__":
    evaluate()
