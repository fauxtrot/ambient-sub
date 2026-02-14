"""Infant Diffusion Model - Stage 0 simplified implementation"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from pathlib import Path


class InfantModelStage0(nn.Module):
    """
    Simplified Stage 0 model: Binary sound detection.

    For initial implementation, this is a simple classifier.
    Full diffusion architecture will be added in later stages.
    """

    def __init__(
        self,
        vocab_size: int = 1024,  # Encodec codebook size
        hidden_dim: int = 128,
        num_layers: int = 2,
        max_seq_len: int = 100,
    ):
        """
        Args:
            vocab_size: Size of encodec vocabulary
            hidden_dim: Hidden dimension size
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embedding for encodec tokens
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))

        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head for Stage 0: binary classification
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokens: Encodec tokens [batch, seq_len]

        Returns:
            Probability of has_sound [batch, 1]
        """
        batch_size, seq_len = tokens.shape

        # Clamp tokens to vocab range
        tokens = torch.clamp(tokens, 0, self.vocab_size - 1)

        # Embed tokens
        embedded = self.token_embedding(tokens)  # [batch, seq_len, hidden_dim]

        # Add positional encoding
        pos_enc = self.pos_embedding[:, :seq_len, :]
        embedded = embedded + pos_enc

        # Encode
        encoded = self.encoder(embedded)  # [batch, seq_len, hidden_dim]

        # Pool: use mean across sequence
        pooled = encoded.mean(dim=1)  # [batch, hidden_dim]

        # Output
        output = self.output_head(pooled)  # [batch, 1]

        return output

    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Generate prediction (alias for forward in Stage 0).

        Args:
            tokens: Encodec tokens [batch, seq_len]

        Returns:
            Prediction [batch, 1]
        """
        return self.forward(tokens)

    def save(self, path: str):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'max_seq_len': self.max_seq_len,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "InfantModelStage0":
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            vocab_size=checkpoint['vocab_size'],
            hidden_dim=checkpoint['hidden_dim'],
            max_seq_len=checkpoint['max_seq_len'],
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model
