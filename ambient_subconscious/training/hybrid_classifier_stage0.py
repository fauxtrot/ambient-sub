"""
Hybrid classifier for Stage 0: has_sound detection.

This model uses BOTH Encodec tokens and acoustic features with learned attention.
The model will learn which features are important for which tasks:
- Stage 0 (has_sound): Rely mostly on acoustic features (RMS, amplitude, entropy)
- Future stages (speaker ID, transcription): Can use token features more

Architecture:
    1. Token Encoder (Transformer) - learns patterns in Encodec tokens
    2. Acoustic Encoder (MLP) - learns acoustic feature combinations
    3. Cross-Attention - lets model decide which features to use
    4. Output Head - binary classification (has_sound)

Why this works:
    - Acoustic features directly capture sound/silence (RMS, amplitude, energy)
    - Token features capture semantic content (for future stages)
    - Cross-attention learns optimal weighting per task
    - Much simpler and faster than diffusion (single forward pass vs 50 steps)
"""

import torch
import torch.nn as nn


class HybridClassifierStage0(nn.Module):
    """
    Binary classifier with hybrid features (Encodec tokens + acoustic).

    The model learns attention weights to decide:
    - For has_sound: rely mostly on acoustic features (3x stronger signal)
    - For future stages: can use token features for speaker ID, transcription

    This flexibility means we don't need to retrain from scratch later!
    """

    def __init__(
        self,
        vocab_size=1024,          # Encodec codebook size
        hidden_dim=128,           # Size of internal representations
        num_layers=2,             # Number of transformer layers
        acoustic_dim=8,           # Number of acoustic features
        max_seq_len=1300,         # Maximum token sequence length (from padding)
        dropout=0.1,              # Dropout rate for regularization
        num_heads=4               # Number of attention heads
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.acoustic_dim = acoustic_dim
        self.max_seq_len = max_seq_len

        # 1. TOKEN ENCODER (Encodec tokens)
        # Converts discrete tokens (0-1023) into continuous vectors
        # Padding token (0) is ignored
        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=0  # Don't learn embedding for padding
        )

        # Positional encoding: tells model where each token is in sequence
        # This is learnable (vs fixed sinusoidal) for simplicity
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02  # Small init
        )

        # Transformer encoder: learns patterns in token sequences
        # Example patterns: "these tokens always appear together in speech"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm (more stable training)
        )
        self.token_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 2. ACOUSTIC ENCODER (acoustic features)
        # Takes 8 features and projects to hidden_dim
        # Uses BatchNorm for feature normalization
        self.acoustic_encoder = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Normalize features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # 3. CROSS-ATTENTION (let model decide which features to use)
        # Query: token features ("what do tokens say?")
        # Key/Value: acoustic features ("what do acoustics say?")
        # Model learns: "for has_sound, trust acoustics 90%, tokens 10%"
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm after attention
        self.attention_norm = nn.LayerNorm(hidden_dim)

        # 4. OUTPUT HEAD (binary classification)
        # Takes attended features and predicts has_sound probability
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1] range (probability)
        )

    def forward(self, tokens, acoustic_features, return_attention=False):
        """
        Forward pass.

        Args:
            tokens: [batch, seq_len] - Encodec tokens (with padding)
            acoustic_features: [batch, 8] - Acoustic features
            return_attention: If True, also return attention weights

        Returns:
            prediction: [batch, 1] - has_sound probability [0, 1]
            attention_weights: [batch, 1, 1] - optional, how much attention on acoustics
        """
        batch_size, seq_len = tokens.shape

        # 1. ENCODE TOKENS
        # Embedding: [batch, seq_len] -> [batch, seq_len, hidden_dim]
        token_embeds = self.token_embedding(tokens)

        # Add positional encoding (only up to seq_len, not full max_seq_len)
        # This tells model "token 0 is first, token 50 is in middle", etc.
        token_embeds = token_embeds + self.pos_embedding[:, :seq_len, :]

        # Run through transformer to learn patterns
        # Create padding mask (True = ignore this token)
        padding_mask = (tokens == 0)  # [batch, seq_len]

        # Transformer encoding
        token_encoded = self.token_encoder(
            token_embeds,
            src_key_padding_mask=padding_mask
        )  # [batch, seq_len, hidden_dim]

        # Pool tokens: average across sequence (ignoring padding)
        # More robust than just taking first/last token
        token_lengths = (~padding_mask).sum(dim=1, keepdim=True)  # [batch, 1]
        token_lengths = token_lengths.clamp(min=1)  # Avoid division by zero

        # Sum over non-padding tokens, divide by count
        token_pooled = (token_encoded * (~padding_mask).unsqueeze(-1)).sum(dim=1) / token_lengths
        # [batch, hidden_dim]

        token_pooled = token_pooled.unsqueeze(1)  # [batch, 1, hidden_dim]

        # 2. ENCODE ACOUSTIC FEATURES
        # [batch, 8] -> [batch, hidden_dim]
        acoustic_encoded = self.acoustic_encoder(acoustic_features)
        acoustic_encoded = acoustic_encoded.unsqueeze(1)  # [batch, 1, hidden_dim]

        # 3. CROSS-ATTENTION
        # Token features attend to acoustic features
        # Model learns: "how much should I trust acoustics vs tokens?"
        #
        # For Stage 0 (has_sound), we expect HIGH attention on acoustics
        # because RMS/amplitude/entropy are 2-3x stronger signals
        attended, attention_weights = self.cross_attention(
            query=token_pooled,       # What do tokens say?
            key=acoustic_encoded,     # What do acoustics offer?
            value=acoustic_encoded    # Use acoustic information
        )  # attended: [batch, 1, hidden_dim], weights: [batch, 1, 1]

        # Residual connection + layer norm (helps training stability)
        attended = self.attention_norm(attended + token_pooled)

        # 4. PREDICT
        attended = attended.squeeze(1)  # [batch, hidden_dim]
        prediction = self.output(attended)  # [batch, 1]

        if return_attention:
            return prediction, attention_weights
        else:
            return prediction

    def predict(self, tokens, acoustic_features, threshold=0.5):
        """
        Predict with threshold.

        Args:
            tokens: [batch, seq_len]
            acoustic_features: [batch, 8]
            threshold: Classification threshold (default 0.5)

        Returns:
            labels: [batch, 1] - Binary labels (0 or 1)
            probs: [batch, 1] - Probabilities [0, 1]
        """
        probs = self.forward(tokens, acoustic_features)
        labels = (probs > threshold).float()
        return labels, probs

    def get_attention_stats(self, tokens, acoustic_features):
        """
        Analyze attention weights to see which features model uses.

        This is useful for understanding model behavior:
        - High attention on acoustics = model trusts acoustic features
        - Low attention = model trusts token features

        Args:
            tokens: [batch, seq_len]
            acoustic_features: [batch, 8]

        Returns:
            Dict with attention statistics
        """
        _, attention_weights = self.forward(
            tokens,
            acoustic_features,
            return_attention=True
        )  # [batch, 1, 1]

        attention_weights = attention_weights.squeeze()  # [batch]

        return {
            'mean_acoustic_attention': attention_weights.mean().item(),
            'std_acoustic_attention': attention_weights.std().item(),
            'min_acoustic_attention': attention_weights.min().item(),
            'max_acoustic_attention': attention_weights.max().item()
        }


# For testing
if __name__ == "__main__":
    print("Testing HybridClassifierStage0 model...")

    # Create model
    model = HybridClassifierStage0(
        vocab_size=1024,
        hidden_dim=128,
        num_layers=2,
        acoustic_dim=8,
        max_seq_len=1300
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")

    # Create dummy input
    batch_size = 4
    seq_len = 200

    tokens = torch.randint(0, 1024, (batch_size, seq_len))  # Random tokens
    tokens[:, 150:] = 0  # Add padding

    acoustic = torch.rand(batch_size, 8)  # Random acoustic features

    # Test forward pass
    print("\nTesting forward pass...")
    prediction = model(tokens, acoustic)
    print(f"  Input shapes: tokens={tokens.shape}, acoustic={acoustic.shape}")
    print(f"  Output shape: {prediction.shape}")
    print(f"  Output range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    print(f"  Sample predictions: {prediction.squeeze().tolist()}")

    # Test predict method
    print("\nTesting predict method...")
    labels, probs = model.predict(tokens, acoustic, threshold=0.5)
    print(f"  Labels: {labels.squeeze().tolist()}")
    print(f"  Probabilities: {probs.squeeze().tolist()}")

    # Test attention analysis
    print("\nTesting attention analysis...")
    attention_stats = model.get_attention_stats(tokens, acoustic)
    print(f"  Mean acoustic attention: {attention_stats['mean_acoustic_attention']:.3f}")
    print(f"  Std acoustic attention: {attention_stats['std_acoustic_attention']:.3f}")
    print(f"  Range: [{attention_stats['min_acoustic_attention']:.3f}, {attention_stats['max_acoustic_attention']:.3f}]")

    print("\n[SUCCESS] Model works!")
