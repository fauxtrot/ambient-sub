# SNN Sound/Silence Classifier

Spiking Neural Network experiment using snnTorch to classify audio segments as **sound** or **silence** based on 8 acoustic features.

## Architecture

- 2-layer feedforward SNN with Leaky Integrate-and-Fire (LIF) neurons
- Input: 8 acoustic features (RMS, energy, ZCR, spectral centroid, token entropy, etc.)
- Rate encoding: features mapped to spike probabilities over 25 timesteps
- Output: spike count over 2 neurons (silence vs sound)
- Training: surrogate gradient descent (fast sigmoid)

## Usage

```bash
pip install -r requirements.txt
python train.py       # Train the model
python evaluate.py    # Evaluate on validation set
```

## Data

Uses `data/training/stage0_hybrid.pt` from the parent project (acoustic features + binary labels).
