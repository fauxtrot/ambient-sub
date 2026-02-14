"""
Temporal Diffusion Model for Stage 0: Binary Sound Detection

This is your first diffusion model! Let's learn by building.

WHAT IS DIFFUSION?
    Imagine teaching someone to draw by showing them:
    1. A perfect drawing
    2. That same drawing with a little noise added
    3. That same drawing with MORE noise added
    4. ... until it's just random scribbles

    Then teach them to REVERSE the process:
    - Start with random scribbles
    - Gradually remove noise, step by step
    - End up with the perfect drawing

    That's diffusion! We:
    - TRAIN: Learn to remove noise from noisy predictions
    - INFER: Start with noise, gradually denoise to get clean prediction

FOR THIS MODEL:
    - Input: Encodec tokens (current audio) + previous 5 frames (temporal context)
    - Output: has_sound prediction (0 or 1)
    - Training: Add noise to ground truth, learn to denoise
    - Inference: Start with random noise, denoise over 50 steps
"""

import torch
import torch.nn as nn
import numpy as np


class NoiseSchedule:
    """
    Defines how much noise to add/remove at each diffusion step.

    Think of this like a volume knob that goes from 0% to 100%:
    - Step 0: 0% noise (clean signal)
    - Step 25: 50% noise (half clean, half random)
    - Step 50: 100% noise (completely random)

    The schedule tells us exactly how much noise at each step.
    """

    def __init__(self, num_steps=50, beta_start=0.0001, beta_end=0.02):
        """
        Create a linear noise schedule.

        Args:
            num_steps: How many denoising steps (default: 50)
                      More steps = slower but smoother
                      Fewer steps = faster but rougher
            beta_start: Starting noise level (very small)
            beta_end: Ending noise level (small but noticeable)
        """
        self.num_steps = num_steps

        # Betas: how much noise to add at each step
        # Linear schedule: increases evenly from start to end
        self.betas = torch.linspace(beta_start, beta_end, num_steps)

        # Alphas: how much original signal to keep (1 - beta)
        self.alphas = 1.0 - self.betas

        # Alpha bars: cumulative product (how much signal remains after t steps)
        # Example: after 10 steps, alpha_bar[10] = alpha[0] * alpha[1] * ... * alpha[10]
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        """
        Add noise to clean signal at timestep t.

        This is the FORWARD PROCESS during training:
        - Take clean prediction (x_0)
        - Add noise based on timestep t
        - Return noisy version (x_t)

        Args:
            x_0: Clean signal [batch, dim]
            t: Timestep (0 to num_steps) [batch]

        Returns:
            x_t: Noisy signal at timestep t
            noise: The noise that was added (what model will learn to predict)
        """
        # Generate random noise (same shape as input)
        noise = torch.randn_like(x_0)

        # Get alpha_bar for this timestep
        # Need to move to same device as t (could be CPU or CUDA)
        # Need to reshape for broadcasting: [batch] -> [batch, 1]
        alpha_bar_t = self.alpha_bars.to(t.device)[t].reshape(-1, 1)

        # Diffusion formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # This is a weighted average of signal and noise
        # - At t=0: alpha_bar ≈ 1, so x_t ≈ x_0 (mostly signal)
        # - At t=50: alpha_bar ≈ 0, so x_t ≈ noise (mostly random)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        return x_t, noise


class TemporalDiffusionStage0(nn.Module):
    """
    Diffusion model for has_sound detection with temporal context.

    ARCHITECTURE:
        1. Encode current frame (encodec tokens)
        2. Encode temporal context (last 5 frames' predictions)
        3. Cross-attention between current + context
        4. Add diffusion timestep information
        5. Denoise the has_sound prediction

    FOR BEGINNERS:
        - nn.Module = base class for all PyTorch models
        - forward() = what happens when you call model(input)
        - Layers are like LEGO blocks you stack together
    """

    def __init__(
        self,
        vocab_size=1024,          # Encodec codebook size
        hidden_dim=128,           # Size of internal representations
        num_layers=2,             # How many transformer layers
        max_seq_len=100,          # Max tokens per frame
        temporal_window=5,        # Last 5 frames
        num_diffusion_steps=50,   # How many denoising steps
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window
        self.num_diffusion_steps = num_diffusion_steps
        self.max_seq_len = max_seq_len

        # Noise schedule for diffusion
        self.noise_schedule = NoiseSchedule(num_steps=num_diffusion_steps)

        # 1. ENCODEC TOKEN ENCODER (current frame)
        # Converts discrete tokens (0-1023) into continuous vectors
        # Example: token 42 -> [0.2, -0.5, 0.8, ...] (128 dimensions)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding: "where is this token in the sequence?"
        # Token at position 0 gets different encoding than position 50
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))

        # Transformer encoder: learns patterns in token sequences
        # Example: "These tokens always appear together" or "This pattern means speech"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,                    # 4 attention heads (look at 4 patterns simultaneously)
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,                # Randomly drop 10% of connections (prevents overfitting)
            batch_first=True
        )
        self.token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 2. TEMPORAL CONTEXT ENCODER (previous frames)
        # Each previous frame is just a single number (has_sound prediction: 0 or 1)
        # We have 5 previous frames, so input is [batch, 5]
        # We want to encode this into [batch, hidden_dim] for attention
        self.temporal_embedding = nn.Linear(temporal_window, hidden_dim)

        # 3. CROSS-ATTENTION (current frame attends to temporal context)
        # Lets the model ask: "What happened in previous frames?"
        # Example: If last 5 frames were all silent, current frame probably silent too
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # 4. DIFFUSION TIMESTEP EMBEDDING
        # The model needs to know "how noisy is the input right now?"
        # Different denoising strategies for different noise levels
        self.time_embedding = nn.Embedding(num_diffusion_steps, hidden_dim)

        # 5. DENOISING NETWORK
        # Takes all the context and predicts the NOISE
        # (Not the signal itself - predicting noise is easier to learn!)
        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat current + time embeddings
            nn.ReLU(),                               # Non-linearity (makes model more powerful)
            nn.Dropout(0.1),                         # Regularization
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),           # Output: single value (has_sound)
        )

    def forward(self, tokens, temporal_context, noisy_label, timestep):
        """
        Predict the NOISE in a noisy has_sound label.

        This is the TRAINING forward pass:
        - We have a clean label (0 or 1)
        - We add noise to it
        - Model learns to predict what noise was added
        - Over time, learns to denoise any noisy input

        Args:
            tokens: [batch, seq_len] - Encodec tokens for current frame
            temporal_context: [batch, temporal_window] - Last 5 frames' predictions
            noisy_label: [batch, 1] - Noisy has_sound prediction to denoise
            timestep: [batch] - Which diffusion step (0 to num_steps)

        Returns:
            predicted_noise: [batch, 1] - What noise to remove
        """
        batch_size, seq_len = tokens.shape

        # 1. ENCODE CURRENT FRAME
        # Token embedding: [batch, seq_len] -> [batch, seq_len, hidden_dim]
        token_embeds = self.token_embedding(tokens)

        # Add positional encoding (up to max_seq_len)
        # This tells model "token 0 is first, token 1 is second", etc.
        token_embeds = token_embeds + self.pos_embedding[:, :seq_len, :]

        # Run through transformer to learn patterns
        # Output: [batch, seq_len, hidden_dim]
        current_encoded = self.token_encoder(token_embeds)

        # Pool to single vector: average all tokens
        # [batch, seq_len, hidden_dim] -> [batch, 1, hidden_dim]
        current_pooled = current_encoded.mean(dim=1, keepdim=True)

        # 2. ENCODE TEMPORAL CONTEXT
        # [batch, temporal_window] -> [batch, hidden_dim]
        temporal_encoded = self.temporal_embedding(temporal_context)
        temporal_encoded = temporal_encoded.unsqueeze(1)  # [batch, 1, hidden_dim]

        # 3. CROSS-ATTENTION: current attends to temporal
        # Model learns: "How do previous frames inform current frame?"
        # attended: [batch, 1, hidden_dim]
        attended, _ = self.cross_attention(
            query=current_pooled,       # "What's happening now?"
            key=temporal_encoded,       # "What happened before?"
            value=temporal_encoded      # "Use that context"
        )

        # 4. ADD DIFFUSION TIMESTEP INFO
        # Model needs to know "how much noise is there?"
        # timestep: [batch] -> [batch, hidden_dim]
        time_embed = self.time_embedding(timestep)
        time_embed = time_embed.unsqueeze(1)  # [batch, 1, hidden_dim]

        # 5. CONCATENATE ALL CONTEXT
        # Combine current frame info + timestep info
        # [batch, 1, hidden_dim*2]
        combined = torch.cat([attended, time_embed], dim=2)
        combined = combined.squeeze(1)  # [batch, hidden_dim*2]

        # 6. DENOISE
        # Predict what noise to remove
        # [batch, hidden_dim*2] -> [batch, 1]
        predicted_noise = self.denoiser(combined)

        return predicted_noise

    def denoise_sample(self, tokens, temporal_context, num_steps=None):
        """
        Generate clean prediction via iterative denoising (INFERENCE).

        This is the REVERSE PROCESS:
        - Start with random noise
        - Gradually remove noise over T steps
        - End with clean prediction

        This is the "magic" of diffusion - we can start with garbage
        and slowly refine it into a meaningful prediction!

        Args:
            tokens: [batch, seq_len] - Encodec tokens
            temporal_context: [batch, temporal_window] - Previous predictions
            num_steps: How many denoising steps (default: use all)

        Returns:
            final_prediction: [batch, 1] - Clean has_sound prediction (0-1)
        """
        if num_steps is None:
            num_steps = self.num_diffusion_steps

        batch_size = tokens.shape[0]
        device = tokens.device

        # START WITH RANDOM NOISE
        # This is pure randomness - model will gradually clean it up
        noisy_label = torch.randn(batch_size, 1).to(device)

        # ITERATIVELY DENOISE (reverse order: T -> 0)
        for t in reversed(range(num_steps)):
            # Current timestep for all samples in batch
            timestep = torch.tensor([t] * batch_size, device=device)

            # Predict the noise
            predicted_noise = self.forward(
                tokens,
                temporal_context,
                noisy_label,
                timestep
            )

            # Get schedule parameters for this timestep
            # Move to same device as noisy_label
            alpha_t = self.noise_schedule.alphas.to(device)[t]
            alpha_bar_t = self.noise_schedule.alpha_bars.to(device)[t]
            beta_t = self.noise_schedule.betas.to(device)[t]

            # Add some noise back (except last step)
            # This helps avoid getting stuck in local minima
            if t > 0:
                noise = torch.randn_like(noisy_label)
            else:
                noise = torch.zeros_like(noisy_label)

            # DENOISING FORMULA (DDPM sampling)
            # Remove predicted noise, add small amount of random noise
            # This is the core of the diffusion reverse process!
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

            noisy_label = coef1 * (noisy_label - coef2 * predicted_noise)
            noisy_label = noisy_label + torch.sqrt(beta_t) * noise

        # FINAL PREDICTION
        # Apply sigmoid to get probability (0-1 range)
        final_prediction = torch.sigmoid(noisy_label)

        return final_prediction


# For pedagogical purposes, let's add a simple test
if __name__ == "__main__":
    print("Testing TemporalDiffusionStage0 model...")

    # Create model
    model = TemporalDiffusionStage0(
        vocab_size=1024,
        hidden_dim=128,
        num_layers=2,
        max_seq_len=100,
        temporal_window=5,
        num_diffusion_steps=50,
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create dummy input
    batch_size = 4
    tokens = torch.randint(0, 1024, (batch_size, 100))  # Random tokens
    temporal_context = torch.rand(batch_size, 5)  # Random previous predictions
    noisy_label = torch.randn(batch_size, 1)  # Random noisy label
    timestep = torch.randint(0, 50, (batch_size,))  # Random timesteps

    # Test forward pass
    print("\nTesting forward pass (training)...")
    predicted_noise = model(tokens, temporal_context, noisy_label, timestep)
    print(f"  Input shape: {tokens.shape}")
    print(f"  Output shape: {predicted_noise.shape}")
    print(f"  Output range: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")

    # Test sampling
    print("\nTesting sampling (inference)...")
    prediction = model.denoise_sample(tokens, temporal_context, num_steps=10)
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    print(f"  Sample predictions: {prediction.squeeze().tolist()}")

    print("\n[SUCCESS] Model works!")
