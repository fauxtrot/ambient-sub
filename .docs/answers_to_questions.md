# Ambient Listener Architecture: Q&A Clarifications

This document provides clarifications to questions raised about the Continuous Latent State Machine architecture, with context from the existing hybrid classifier work.

---

## Context: What We've Already Built

The project has a working hybrid encoder/classifier:

```python
# Current hybrid architecture (stage0_hybrid.pt)
{
    'tokens': [1297, 1244],      # Padded Encodec sequences
    'acoustic': [1297, 8],       # 8 acoustic features per sample
    'labels': [1297, 1],         # Binary has_sound labels
    'metadata': [...],           # Transcriptions, speaker IDs, etc.
}

# The hybrid classifier has two logical parts:
# 1. ENCODER: token transformer + acoustic MLP + cross-attention → 128-dim embedding
# 2. HEAD: 128-dim → sigmoid → is_sound prediction
```

The goal is to evolve this toward an M4-style multimodal token format while building the continuous latent state machine.

---

## Q1: Where Does the Hybrid Classifier Fit?

**Answer: Split it into two roles.**

### The Encoder (token transformer + acoustic MLP + cross-attention)
- **Becomes the audio encoder** for the subconscious layer
- Produces the dense embedding that perturbs the latent state
- Project from 128-dim to latent_dim (512 or 1024)

```python
class AudioEncoderHybrid(nn.Module):
    """Reuse trained hybrid encoder, project to latent space"""
    
    def __init__(self, latent_dim=512):
        super().__init__()
        # These weights can be initialized from trained classifier
        self.token_embedding = nn.Embedding(1024, 128, padding_idx=0)
        self.token_encoder = TransformerEncoder(...)
        self.acoustic_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.cross_attention = nn.MultiheadAttention(128, num_heads=4)
        
        # New: project to subconscious latent space
        self.to_latent = nn.Linear(128, latent_dim)
    
    def forward(self, tokens, acoustic_features):
        token_embeds = self.token_embedding(tokens)
        token_encoded = self.token_encoder(token_embeds)
        token_pooled = token_encoded.mean(dim=1)
        
        acoustic_encoded = self.acoustic_encoder(acoustic_features)
        
        attended, _ = self.cross_attention(
            query=token_pooled.unsqueeze(1),
            key=acoustic_encoded.unsqueeze(1),
            value=acoustic_encoded.unsqueeze(1)
        )
        
        return self.to_latent(attended.squeeze(1))  # [batch, latent_dim]
```

### The Classification Head (128-dim → is_sound)
- **Lives in the reflexive layer** as a fast gate
- Runs in parallel with subconscious, provides early annotation
- Can also become a projection head on the subconscious latent later

```python
class ReflexiveIsSound(nn.Module):
    """Fast is_sound detection for reflexive layer"""
    
    def __init__(self):
        # Initialize from trained classifier weights
        self.encoder = AudioEncoderHybrid(latent_dim=128)  # Keep at 128 for speed
        self.head = nn.Linear(128, 1)
    
    def forward(self, tokens, acoustic_features):
        embedding = self.encoder(tokens, acoustic_features)
        return torch.sigmoid(self.head(embedding))
```

### Transition Strategy
1. **Now**: Reflexive classifier runs standalone, provides `is_sound` annotation
2. **Soon**: Subconscious latent also has `is_sound` projection head
3. **Later**: When projection head matches classifier accuracy, deprecate standalone classifier
4. **Or**: Keep both — reflexive for speed, projection head for integrated reasoning

---

## Q2: The Update Rule — Drop Diffusion Framing

**Answer: It's a recurrent latent state model, not diffusion.**

The diffusion intuition was useful for thinking about "input as perturbation" but the DDPM machinery (noise schedules, alpha values, noise prediction) doesn't apply.

### What One Step Actually Looks Like

```python
class ContinuousLatentStateMachine(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Input projection (handles variable input size)
        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)  # temporal_ctx + input
        
        # Update network (transformer block, MLP, GRU-like, whatever works)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Optional: residual connection strength
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def step(self, latent_history: List[Tensor], input_encoding: Tensor) -> Tensor:
        """
        One step of the continuous state machine.
        
        Args:
            latent_history: List of previous latent states (at least 10 frames)
            input_encoding: Encoded audio/annotations from this timestep [latent_dim]
        
        Returns:
            New latent state [latent_dim]
        """
        # Merge temporal context from history
        temporal_ctx = self.compute_temporal_context(latent_history)  # [latent_dim]
        
        # Combine temporal context with new input
        combined = torch.cat([temporal_ctx, input_encoding], dim=-1)  # [latent_dim * 2]
        
        # Project and update
        hidden = self.input_proj(combined)  # [hidden_dim]
        delta = self.update_net(hidden)  # [latent_dim]
        
        # New latent = weighted combination of previous + update
        new_latent = self.residual_weight * latent_history[-1] + (1 - self.residual_weight) * delta
        
        return new_latent
    
    def compute_temporal_context(self, latent_history: List[Tensor]) -> Tensor:
        """Weighted merge of historical frames"""
        # Fixed weights to start, can be learned later
        weights = [0.6, 0.3, 0.1]  # -1, -5, -10 frames
        indices = [-1, -5, -10]
        
        ctx = torch.zeros(self.latent_dim)
        for w, idx in zip(weights, indices):
            if len(latent_history) >= abs(idx):
                ctx += w * latent_history[idx]
            else:
                ctx += w * latent_history[0]  # Use earliest if not enough history
        
        return ctx
```

### No Diffusion Machinery Needed
- No alpha schedules
- No noise prediction
- No fixed number of denoising steps
- Just: `new_state = f(old_state, new_input)`

---

## Q3: Input Encoding and M4 Token Format

**Answer: Preserve structure, don't collapse everything into one embedding.**

### The M4 Token Concept

Instead of just producing a dense embedding, create a structured token that preserves all information:

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch

@dataclass
class M4Token:
    """Multimodal token format - preserves structure for downstream flexibility"""
    
    # Identity
    modality: str                    # "audio", "vision", "text", etc.
    source: str                      # "ambient_mic", "webcam", "keyboard", etc.
    timestamp: float                 # Unix timestamp
    
    # Dense representation (from hybrid encoder)
    embedding: torch.Tensor          # [latent_dim] - main perturbation signal
    
    # Structured features (preserved, not collapsed)
    features: Dict[str, float]       # {"rms": 0.23, "spectral_centroid": 1420.5, ...}
    
    # Reflexive annotations (added by reflexive layer)
    annotations: Dict[str, Any]      # {"is_sound": {"value": True, "confidence": 0.94}}
    
    # Raw data (for models that want it)
    raw_tokens: Optional[List[int]]  # Encodec tokens if audio
    raw_data: Optional[Any]          # Original data reference


class M4AudioTokenizer:
    """Creates M4 tokens from audio input"""
    
    def __init__(self, audio_encoder, reflexive_classifier, latent_dim=512):
        self.audio_encoder = audio_encoder          # Hybrid encoder → embedding
        self.reflexive = reflexive_classifier       # Fast is_sound detection
        self.encodec = EncodecModel(...)            # Tokenization
        self.latent_dim = latent_dim
    
    def tokenize(self, audio_chunk: np.ndarray, sr: int, timestamp: float) -> M4Token:
        # 1. Encodec tokenization
        encodec_tokens = self.encodec.encode(audio_chunk)
        
        # 2. Acoustic feature extraction (your existing 8 features)
        acoustic_features = extract_acoustic_features(audio_chunk, sr, encodec_tokens)
        
        # 3. Dense embedding via hybrid encoder
        tokens_tensor = torch.tensor(encodec_tokens).unsqueeze(0)
        acoustic_tensor = torch.tensor(list(acoustic_features.values())).unsqueeze(0)
        embedding = self.audio_encoder(tokens_tensor, acoustic_tensor).squeeze(0)
        
        # 4. Reflexive annotation (fast is_sound check)
        is_sound_conf = self.reflexive(tokens_tensor, acoustic_tensor).item()
        
        # 5. Package as M4 token
        return M4Token(
            modality="audio",
            source="ambient_mic",
            timestamp=timestamp,
            embedding=embedding,
            features=acoustic_features,  # Preserved, not collapsed!
            annotations={
                "is_sound": {
                    "value": is_sound_conf > 0.5,
                    "confidence": is_sound_conf
                }
            },
            raw_tokens=encodec_tokens,
            raw_data=None  # Don't store raw audio, just reference if needed
        )
```

### How Subconscious Consumes M4 Tokens

```python
class SubconsciousModel(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Different ways to consume M4 token parts
        self.feature_encoder = nn.Linear(8, 64)           # Structured features
        self.annotation_encoder = nn.Linear(16, 64)       # Annotations (varies)
        
        # Fusion options
        self.fusion = nn.Linear(latent_dim + 64 + 64, latent_dim)
        
        # State update
        self.state_machine = ContinuousLatentStateMachine(latent_dim)
    
    def encode_m4_token(self, token: M4Token) -> torch.Tensor:
        """Convert M4 token to input perturbation"""
        
        # Option A: Just use the dense embedding (simplest)
        # return token.embedding
        
        # Option B: Fuse embedding + structured features + annotations
        feature_vec = torch.tensor(list(token.features.values()))
        feature_encoded = self.feature_encoder(feature_vec)
        
        annotation_vec = self.encode_annotations(token.annotations)
        annotation_encoded = self.annotation_encoder(annotation_vec)
        
        combined = torch.cat([token.embedding, feature_encoded, annotation_encoded], dim=-1)
        return self.fusion(combined)
    
    def encode_annotations(self, annotations: Dict) -> torch.Tensor:
        """Flatten annotations to vector (schema-dependent)"""
        # Simple version: just is_sound for now
        is_sound = annotations.get("is_sound", {"value": False, "confidence": 0.0})
        return torch.tensor([
            float(is_sound["value"]),
            is_sound["confidence"]
        ])
    
    def step(self, latent_history: List[torch.Tensor], token: M4Token) -> torch.Tensor:
        input_encoding = self.encode_m4_token(token)
        return self.state_machine.step(latent_history, input_encoding)
```

### Why M4 Format Matters

1. **Flexibility**: Subconscious can use embedding only now, add feature attention later
2. **Debuggability**: You can inspect structured features without decoding embeddings
3. **Multimodal extension**: Vision M4 tokens have same envelope, different features
4. **Reflexive integration**: Annotations flow naturally from reflexive layer
5. **Future-proofing**: Add new features/annotations without breaking existing code

---

## Q4: Memory Resonance Implementation

**Answer: Vector similarity search with safeguards against flooding.**

```python
class MemoryResonanceSystem:
    """Passive memory retrieval via embedding similarity"""
    
    def __init__(
        self,
        latent_dim: int = 512,
        max_memories: int = 100000,
        similarity_threshold: float = 0.7,
        top_k: int = 10,
        recency_halflife: float = 3600.0,  # 1 hour in seconds
        memory_contribution_cap: float = 0.3
    ):
        self.latent_dim = latent_dim
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.recency_halflife = recency_halflife
        self.memory_contribution_cap = memory_contribution_cap
        
        # Storage
        self.embeddings = torch.zeros(max_memories, latent_dim)
        self.timestamps = torch.zeros(max_memories)
        self.count = 0
        self.index = 0  # Circular buffer position
    
    def store(self, latent: torch.Tensor, timestamp: float):
        """Store a frame in memory"""
        self.embeddings[self.index] = latent.detach()
        self.timestamps[self.index] = timestamp
        self.index = (self.index + 1) % len(self.embeddings)
        self.count = min(self.count + 1, len(self.embeddings))
    
    def retrieve(self, query_latent: torch.Tensor, current_time: float) -> torch.Tensor:
        """
        Retrieve memory echo via similarity search.
        
        Returns weighted combination of similar memories,
        or zeros if nothing passes threshold.
        """
        if self.count == 0:
            return torch.zeros(self.latent_dim)
        
        # Get active memories
        active_embeddings = self.embeddings[:self.count]
        active_timestamps = self.timestamps[:self.count]
        
        # Compute similarities (cosine)
        query_norm = query_latent / (query_latent.norm() + 1e-8)
        memory_norms = active_embeddings / (active_embeddings.norm(dim=1, keepdim=True) + 1e-8)
        similarities = torch.matmul(memory_norms, query_norm)  # [count]
        
        # Apply recency weighting
        ages = current_time - active_timestamps
        recency_weights = torch.exp(-ages / self.recency_halflife)
        weighted_similarities = similarities * recency_weights
        
        # Filter by threshold
        mask = similarities > self.similarity_threshold
        if not mask.any():
            return torch.zeros(self.latent_dim)
        
        # Top-k from passing memories
        masked_similarities = weighted_similarities.clone()
        masked_similarities[~mask] = -float('inf')
        
        top_k = min(self.top_k, mask.sum().item())
        top_values, top_indices = torch.topk(masked_similarities, top_k)
        
        # Weighted average
        weights = torch.softmax(top_values, dim=0)
        memory_echo = torch.sum(
            weights.unsqueeze(1) * active_embeddings[top_indices],
            dim=0
        )
        
        return memory_echo
    
    def compute_noise_basis_with_memory(
        self,
        latent_history: List[torch.Tensor],
        current_time: float,
        temporal_weights: List[float] = [0.6, 0.3, 0.1],
        temporal_indices: List[int] = [-1, -5, -10]
    ) -> torch.Tensor:
        """
        Compute noise basis from temporal history + memory resonance.
        """
        # Temporal merge
        temporal_ctx = torch.zeros(self.latent_dim)
        for w, idx in zip(temporal_weights, temporal_indices):
            if len(latent_history) >= abs(idx):
                temporal_ctx += w * latent_history[idx]
            elif len(latent_history) > 0:
                temporal_ctx += w * latent_history[0]
        
        # Memory resonance (query with most recent latent)
        if len(latent_history) > 0:
            memory_echo = self.retrieve(latent_history[-1], current_time)
        else:
            memory_echo = torch.zeros(self.latent_dim)
        
        # Combine with memory contribution capped
        noise_basis = (
            (1 - self.memory_contribution_cap) * temporal_ctx +
            self.memory_contribution_cap * memory_echo
        )
        
        return noise_basis
```

### Safeguards Against Memory Flooding

1. **Similarity threshold** (0.7): Only memories above threshold contribute
2. **Top-k cap** (10): Maximum number of memories in any retrieval
3. **Recency weighting**: Older memories fade exponentially
4. **Contribution cap** (0.3): Memory echo is max 30% of noise basis
5. **Circular buffer**: Old memories automatically evicted

---

## Q5: Temporal Merge Weights

**Answer: Start fixed, make learnable later.**

```python
class TemporalMerger(nn.Module):
    """Handles temporal context merging with optional learning"""
    
    def __init__(
        self,
        latent_dim: int,
        temporal_indices: List[int] = [-1, -5, -10],
        initial_weights: List[float] = [0.6, 0.3, 0.1],
        learnable: bool = False,
        input_dependent: bool = False
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.temporal_indices = temporal_indices
        
        if input_dependent:
            # Dynamic: weights depend on current input
            self.weight_net = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, len(temporal_indices)),
                nn.Softmax(dim=-1)
            )
            self.weights = None
        elif learnable:
            # Static but learnable
            self.weights = nn.Parameter(torch.tensor(initial_weights))
            self.weight_net = None
        else:
            # Fixed
            self.register_buffer('weights', torch.tensor(initial_weights))
            self.weight_net = None
    
    def forward(
        self,
        latent_history: List[torch.Tensor],
        current_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Merge temporal context from history.
        
        Args:
            latent_history: List of previous latents
            current_input: Current input encoding (for input-dependent weights)
        
        Returns:
            Merged temporal context [latent_dim]
        """
        # Compute weights
        if self.weight_net is not None and current_input is not None:
            weights = self.weight_net(current_input)
        else:
            weights = torch.softmax(self.weights, dim=0)  # Normalize
        
        # Weighted merge
        ctx = torch.zeros(self.latent_dim)
        for i, (w, idx) in enumerate(zip(weights, self.temporal_indices)):
            if len(latent_history) >= abs(idx):
                ctx += w * latent_history[idx]
            elif len(latent_history) > 0:
                ctx += w * latent_history[0]
        
        return ctx


# Usage progression:

# Phase 1: Fixed weights (start here)
merger_v1 = TemporalMerger(latent_dim=512, learnable=False)

# Phase 2: Learnable but static (after basic system works)
merger_v2 = TemporalMerger(latent_dim=512, learnable=True)

# Phase 3: Input-dependent (if needed for responsiveness)
merger_v3 = TemporalMerger(latent_dim=512, input_dependent=True)
```

### Sensitivity Analysis

- **Over-weight recent (0.9, 0.08, 0.02)**: System is twitchy, loses context fast
- **Over-weight old (0.2, 0.3, 0.5)**: System is sluggish, slow to respond to changes
- **Balanced (0.6, 0.3, 0.1)**: Reasonable default, recent dominates but context persists

The system will still learn with wrong weights — just slower or with worse temporal reasoning. Not catastrophic.

---

## Q6: Training Data Format

**Answer: Window over existing segments, add temporal eligibility masking.**

### Building Sequences from Segments

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

@dataclass
class TemporalLabel:
    """Label with temporal eligibility"""
    value: Any                    # The actual label value
    eligible_after_ms: float      # Don't penalize until this much audio seen
    
    # Special sentinel for "not eligible yet"
    MASKED = object()


def build_training_sequences(
    segments: List[Dict],         # Your existing enriched segments
    window_size: int = 20,        # Frames per sequence
    stride: int = 5,              # Overlap
    frame_duration_ms: float = 100.0  # How long each frame represents
) -> List[Dict]:
    """
    Convert segment-level labels to temporal sequences with eligibility.
    """
    # Sort by timestamp
    sorted_segments = sorted(segments, key=lambda s: s['timestamp'])
    
    sequences = []
    
    for i in range(0, len(sorted_segments) - window_size, stride):
        window = sorted_segments[i:i + window_size]
        
        sequence = {
            "audio_chunks": [],
            "timestamps": [],
            "labels_per_frame": []
        }
        
        for frame_idx, segment in enumerate(window):
            elapsed_ms = frame_idx * frame_duration_ms
            
            sequence["audio_chunks"].append(segment['audio'])
            sequence["timestamps"].append(segment['timestamp'])
            
            # Build labels with temporal eligibility
            frame_labels = {
                # is_sound: eligible immediately
                "is_sound": segment.get('has_sound', TemporalLabel.MASKED),
                
                # is_speech: eligible after ~150ms (frame 2+)
                "is_speech": (
                    segment.get('is_speech', TemporalLabel.MASKED)
                    if elapsed_ms >= 150 else TemporalLabel.MASKED
                ),
                
                # speaker_id: eligible after ~300ms (frame 3+)
                "speaker_id": (
                    segment.get('speaker_id', TemporalLabel.MASKED)
                    if elapsed_ms >= 300 else TemporalLabel.MASKED
                ),
                
                # affect: eligible after ~500ms (frame 5+)
                "affect": (
                    segment.get('affect', TemporalLabel.MASKED)
                    if elapsed_ms >= 500 else TemporalLabel.MASKED
                ),
            }
            
            sequence["labels_per_frame"].append(frame_labels)
        
        sequences.append(sequence)
    
    return sequences


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset for temporal sequences"""
    
    MASKED = TemporalLabel.MASKED
    
    def __init__(self, sequences: List[Dict], tokenizer: M4AudioTokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Tokenize all audio chunks
        tokens = []
        for audio, ts in zip(seq['audio_chunks'], seq['timestamps']):
            m4_token = self.tokenizer.tokenize(audio, sr=16000, timestamp=ts)
            tokens.append(m4_token)
        
        return {
            "tokens": tokens,
            "labels": seq['labels_per_frame']
        }
```

### Training Loop with Masking

```python
def train_on_sequence(model, sequence, optimizer):
    """Train on one sequence with temporal eligibility masking"""
    
    latent = torch.zeros(model.latent_dim)
    latent_history = [latent.clone() for _ in range(10)]  # Bootstrap history
    
    total_loss = 0
    loss_counts = {"is_sound": 0, "is_speech": 0, "speaker_id": 0, "affect": 0}
    
    for frame_idx, (m4_token, labels) in enumerate(zip(sequence['tokens'], sequence['labels'])):
        # Step the model
        latent = model.step(latent_history, m4_token)
        latent_history.append(latent)
        latent_history = latent_history[-20:]  # Keep last 20
        
        # Read predictions
        predictions = model.read_features(latent)
        
        # Compute loss only for eligible labels
        frame_loss = 0
        for feature_name, target in labels.items():
            if target is not TemporalLabel.MASKED and target is not None:
                pred = predictions[feature_name]
                
                # Choose appropriate loss
                if feature_name in ["is_sound", "is_speech"]:
                    loss = F.binary_cross_entropy_with_logits(pred, torch.tensor([float(target)]))
                elif feature_name == "speaker_id":
                    loss = F.cross_entropy(pred.unsqueeze(0), torch.tensor([target]))
                elif feature_name == "affect":
                    loss = F.mse_loss(pred, torch.tensor(target))
                
                frame_loss += loss
                loss_counts[feature_name] += 1
        
        total_loss += frame_loss
    
    # Backprop
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_counts
```

---

## Q7: Cold Boot Stability

**Answer: Start with zeros, learned prior optional, ~200ms to basic features.**

```python
class ColdBootManager:
    """Handles system initialization and warmup"""
    
    def __init__(
        self,
        latent_dim: int = 512,
        use_learned_prior: bool = False,
        warmup_threshold_ms: float = 200.0,
        frame_duration_ms: float = 50.0
    ):
        self.latent_dim = latent_dim
        self.warmup_threshold_ms = warmup_threshold_ms
        self.frame_duration_ms = frame_duration_ms
        self.frames_for_warmup = int(warmup_threshold_ms / frame_duration_ms)
        
        if use_learned_prior:
            # Learned neutral state (train this to represent "no information")
            self.neutral_prior = nn.Parameter(torch.randn(latent_dim) * 0.01)
        else:
            # Simple zeros
            self.neutral_prior = torch.zeros(latent_dim)
        
        self.frame_count = 0
        self.is_warm = False
    
    def get_initial_latent(self) -> torch.Tensor:
        """Get starting latent for cold boot"""
        self.frame_count = 0
        self.is_warm = False
        return self.neutral_prior.clone()
    
    def get_initial_history(self, length: int = 10) -> List[torch.Tensor]:
        """Bootstrap latent history for cold start"""
        return [self.neutral_prior.clone() for _ in range(length)]
    
    def update(self, new_latent: torch.Tensor) -> Dict[str, Any]:
        """Track warmup progress"""
        self.frame_count += 1
        
        status = {
            "frame_count": self.frame_count,
            "elapsed_ms": self.frame_count * self.frame_duration_ms,
            "warmup_progress": min(1.0, self.frame_count / self.frames_for_warmup),
            "is_warm": self.frame_count >= self.frames_for_warmup
        }
        
        if not self.is_warm and status["is_warm"]:
            self.is_warm = True
            status["just_warmed_up"] = True
        
        return status
    
    def get_feature_confidence_scaling(self) -> Dict[str, float]:
        """
        Scale confidence based on warmup status.
        Early features can be trusted sooner.
        """
        elapsed = self.frame_count * self.frame_duration_ms
        
        return {
            "is_sound": min(1.0, elapsed / 100),      # Full confidence at 100ms
            "is_speech": min(1.0, elapsed / 200),    # Full confidence at 200ms
            "speaker_id": min(1.0, elapsed / 400),   # Full confidence at 400ms
            "affect": min(1.0, elapsed / 800),       # Full confidence at 800ms
        }


# Usage:
boot_manager = ColdBootManager(latent_dim=512, use_learned_prior=False)

# Cold start
latent = boot_manager.get_initial_latent()
latent_history = boot_manager.get_initial_history()

# Processing loop
for m4_token in audio_stream:
    latent = model.step(latent_history, m4_token)
    latent_history.append(latent)
    latent_history = latent_history[-20:]
    
    status = boot_manager.update(latent)
    confidence_scaling = boot_manager.get_feature_confidence_scaling()
    
    # Read features with confidence scaling
    predictions = model.read_features(latent)
    for feature, pred in predictions.items():
        scaled_confidence = pred['confidence'] * confidence_scaling[feature]
        # Use scaled_confidence for downstream decisions
```

### Warmup Timeline

| Time | Frames | Features Available |
|------|--------|-------------------|
| 0ms | 0 | None (cold) |
| 50ms | 1 | is_sound (partial) |
| 100ms | 2 | is_sound (full) |
| 200ms | 4 | is_sound, is_speech |
| 400ms | 8 | is_sound, is_speech, speaker_id |
| 800ms | 16 | All features stable |

---

## Q8: Enforcing Early Convergence

**Answer: Loss weighting is primary, architectural tricks optional.**

```python
class EarlyConvergenceLoss(nn.Module):
    """
    Loss function that rewards early convergence on simple features.
    """
    
    def __init__(
        self,
        feature_configs: Dict[str, Dict] = None,
        convergence_bonus_weight: float = 0.1
    ):
        super().__init__()
        
        # When each feature should converge (frame index)
        self.feature_configs = feature_configs or {
            "is_sound": {
                "target_frame": 2,
                "loss_fn": F.binary_cross_entropy_with_logits,
                "weight_at_target": 1.0,
                "weight_after": 0.3,  # Still penalize regression
            },
            "is_speech": {
                "target_frame": 4,
                "loss_fn": F.binary_cross_entropy_with_logits,
                "weight_at_target": 1.0,
                "weight_after": 0.3,
            },
            "speaker_id": {
                "target_frame": 8,
                "loss_fn": F.cross_entropy,
                "weight_at_target": 1.0,
                "weight_after": 0.3,
            },
            "affect": {
                "target_frame": 12,
                "loss_fn": F.mse_loss,
                "weight_at_target": 1.0,
                "weight_after": 0.3,
            },
        }
        
        self.convergence_bonus_weight = convergence_bonus_weight
    
    def forward(
        self,
        predictions_by_frame: List[Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with early convergence incentives.
        
        Args:
            predictions_by_frame: List of {feature: prediction} for each frame
            targets: {feature: target} ground truth
        
        Returns:
            Dict with total_loss and per-feature breakdowns
        """
        losses = {name: 0.0 for name in self.feature_configs}
        convergence_bonuses = {name: 0.0 for name in self.feature_configs}
        
        for frame_idx, preds in enumerate(predictions_by_frame):
            for feature_name, config in self.feature_configs.items():
                if feature_name not in targets or targets[feature_name] is None:
                    continue
                
                target = targets[feature_name]
                pred = preds.get(feature_name)
                
                if pred is None:
                    continue
                
                target_frame = config["target_frame"]
                loss_fn = config["loss_fn"]
                
                # Compute base loss
                base_loss = loss_fn(pred, target)
                
                # Weight based on frame position
                if frame_idx < target_frame:
                    # Before target: no penalty (not expected to know yet)
                    weight = 0.0
                elif frame_idx == target_frame:
                    # At target: full weight (should know by now)
                    weight = config["weight_at_target"]
                else:
                    # After target: reduced weight (maintain, don't regress)
                    weight = config["weight_after"]
                
                losses[feature_name] += weight * base_loss
                
                # Convergence bonus: reward being correct BEFORE target frame
                if frame_idx < target_frame:
                    # If already correct, give bonus
                    if self._is_correct(pred, target, feature_name):
                        convergence_bonuses[feature_name] += (
                            self.convergence_bonus_weight *
                            (target_frame - frame_idx)  # More bonus for earlier
                        )
        
        total_loss = sum(losses.values()) - sum(convergence_bonuses.values())
        
        return {
            "total_loss": total_loss,
            "per_feature_loss": losses,
            "convergence_bonuses": convergence_bonuses
        }
    
    def _is_correct(self, pred, target, feature_name) -> bool:
        """Check if prediction is correct (for bonus calculation)"""
        config = self.feature_configs[feature_name]
        
        if config["loss_fn"] == F.binary_cross_entropy_with_logits:
            return (torch.sigmoid(pred) > 0.5) == (target > 0.5)
        elif config["loss_fn"] == F.cross_entropy:
            return pred.argmax() == target
        else:
            # MSE: close enough
            return F.mse_loss(pred, target) < 0.1
```

### Optional: Gradient Isolation

If features are bleeding into each other's timing, you can isolate gradients:

```python
class IsolatedFeatureHeads(nn.Module):
    """Feature heads with optional gradient isolation"""
    
    def __init__(self, latent_dim: int, isolate_early_features: bool = False):
        super().__init__()
        self.isolate = isolate_early_features
        
        self.heads = nn.ModuleDict({
            "is_sound": nn.Linear(latent_dim, 1),
            "is_speech": nn.Linear(latent_dim, 1),
            "speaker_id": nn.Linear(latent_dim, 128),  # Embedding or logits
            "affect": nn.Linear(latent_dim, 8),
        })
        
        # Which features read from detached latent (early features)
        self.detach_for = {"is_sound", "is_speech"} if isolate_early_features else set()
    
    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        results = {}
        
        for name, head in self.heads.items():
            if name in self.detach_for:
                # Early features: don't let their loss affect main latent
                results[name] = head(latent.detach())
            else:
                results[name] = head(latent)
        
        return results
```

This prevents late features (affect) from pulling the latent representation in ways that hurt early features (is_sound). Use sparingly — loss weighting should be sufficient in most cases.

---

## Summary: Implementation Priority

1. **M4 Token format**: Define the dataclass, build the tokenizer
2. **Audio encoder**: Adapt hybrid classifier encoder, project to latent_dim
3. **Continuous state machine**: Simple recurrent update, no diffusion machinery
4. **Feature heads**: Linear projections, add as capabilities grow
5. **Training loop**: Sequential, with temporal eligibility masking
6. **Early convergence loss**: Weighted by frame position
7. **Temporal merger**: Fixed weights initially
8. **Memory resonance**: Add after basic system works
9. **Cold boot manager**: Simple zeros, warmup tracking

Start simple. Get the basic loop running. Add complexity only when you see specific failures that need solving.