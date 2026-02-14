# Ambient Listener Architecture v2: Continuous Latent State Machine with Synchronization

## Document Purpose

This document supersedes the previous architecture documents and incorporates insights from Sakana AI's Continuous Thought Machines (CTM) research. It provides a comprehensive specification for the ambient listener's subconscious processing layer, with implementation starting points.

---

## Background: Sakana AI's Continuous Thought Machines

In May 2025, Sakana AI published research on Continuous Thought Machines (CTM) that validates several core intuitions in our architecture while introducing powerful new concepts worth incorporating.

### CTM Core Ideas

1. **Time as computation**: Neurons use timing and synchronization as fundamental to how they compute. Modern AI discards this in favor of efficiency. CTM bridges the gap.

2. **Internal recurrence decoupled from data**: Unlike RNNs that step with data, CTM has an independent "thought dimension" — internal ticks that process regardless of input cadence.

3. **Neuron-level models (NLMs)**: Each neuron has its own private MLP that processes a history of incoming activations. Neurons are heterogeneous, not uniform.

4. **Synchronization as representation**: The pairwise correlation between neurons over time forms a synchronization matrix. This matrix IS the representation used for attention and prediction — not the activation vector itself.

5. **Adaptive compute**: Loss function optimizes for both minimum loss AND maximum certainty across internal ticks. The model learns when it's ready to answer.

### CTM Results

- Solves mazes by building internal world models without positional embeddings
- Superior calibration compared to baselines (and humans on CIFAR-10)
- Exhibits interpretable attention patterns that follow logical reasoning paths
- Generalizes beyond training distribution (larger mazes, longer sequences)

### Relevance to Our Architecture

Our continuous latent state machine shares CTM's core philosophy:
- Time matters
- Dynamics are representation  
- Internal process decoupled from input cadence
- History of activations, not snapshots

Key differences we should address:
- CTM has neuron-level diversity; we have uniform update networks
- CTM uses synchronization matrix; we use latent vector
- CTM has adaptive certainty-based loss; we have fixed convergence targets

This document proposes a hybrid architecture that incorporates CTM insights while maintaining our specific requirements (multimodal input, long-term memory, human-plausible timing).

---

## Architecture Overview

### Three-Layer System

```
┌─────────────────────────────────────────────────────────────────┐
│                       EXECUTIVE LAYER                           │
│  - Attention allocation                                         │
│  - Response planning                                            │
│  - Output decisions (expression, camera, speech)                │
│  - Reads from subconscious latent + synchronization             │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ latent state + sync features
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     SUBCONSCIOUS LAYER                          │
│  - Continuous latent state machine                              │
│  - Neuron-level models (CTM-inspired)                           │
│  - Synchronization tracking                                     │
│  - Memory resonance                                             │
│  - Feature projection heads                                     │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ M4 tokens
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      REFLEXIVE LAYER                            │
│  - Fast DSP-level processing                                    │
│  - Gain adjustment, anomaly detection                           │
│  - One-shot classification (is_sound, object detection)         │
│  - Temporal buffer management                                   │
│  - Produces M4 tokens with annotations                          │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ raw input
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       INPUT SOURCES                             │
│  - Ambient microphone                                           │
│  - Webcam                                                       │
│  - System events                                                │
│  - Text input                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## M4 Token Format

All input flows through a unified multimodal token format.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch

@dataclass
class M4Token:
    """
    Multimodal token format for the ambient listener system.
    Preserves structure while providing dense embeddings.
    """
    
    # === Identity ===
    modality: str                           # "audio", "vision", "text", "system"
    source: str                             # "ambient_mic", "webcam", "keyboard"
    timestamp: float                        # Unix timestamp (seconds)
    duration_ms: float                      # How long this token spans
    
    # === Dense Representation ===
    embedding: torch.Tensor                 # [latent_dim] - from modality encoder
    
    # === Structured Features ===
    # Preserved for interpretability and specialized processing
    features: Dict[str, float] = field(default_factory=dict)
    # Audio: {"rms": 0.23, "spectral_centroid": 1420.5, "zcr": 0.12, ...}
    # Vision: {"motion": 0.1, "faces_detected": 1, "brightness": 0.7, ...}
    
    # === Reflexive Annotations ===
    # Added by reflexive layer, consumed by subconscious
    annotations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # {"is_sound": {"value": True, "confidence": 0.94, "latency_ms": 12}}
    # {"object_detected": {"class": "human", "location": "desk", "confidence": 0.87}}
    
    # === Raw Data References ===
    raw_tokens: Optional[List[int]] = None  # Encodec tokens for audio
    raw_data_ref: Optional[str] = None      # Path/reference to raw data if needed
    
    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to tensor format for model consumption"""
        feature_vec = torch.tensor(list(self.features.values())) if self.features else torch.zeros(8)
        annotation_vec = self._encode_annotations()
        
        return {
            "embedding": self.embedding,
            "features": feature_vec,
            "annotations": annotation_vec,
            "timestamp": torch.tensor([self.timestamp]),
            "duration": torch.tensor([self.duration_ms]),
        }
    
    def _encode_annotations(self) -> torch.Tensor:
        """Flatten annotations to fixed-size vector"""
        # Schema-dependent; expand as capabilities grow
        vec = []
        
        for key in ["is_sound", "is_speech", "speaker_detected", "object_detected"]:
            if key in self.annotations:
                ann = self.annotations[key]
                vec.extend([float(ann.get("value", False)), ann.get("confidence", 0.0)])
            else:
                vec.extend([0.0, 0.0])
        
        return torch.tensor(vec)
```

---

## Subconscious Layer: Core Architecture

### Design Principles (CTM-Informed)

1. **Neuron-level diversity**: Different regions of the latent space have different update dynamics
2. **Synchronization tracking**: Pairwise correlations between regions over time form part of the representation
3. **History-aware processing**: Both pre-activation and post-activation histories inform updates
4. **Adaptive certainty**: Model learns when features are ready, not just what they are

### Architecture Components

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

class SubconsciousLayer(nn.Module):
    """
    Continuous Latent State Machine with CTM-inspired synchronization.
    
    Key features:
    - Regional neuron-level models (heterogeneous processing)
    - Synchronization matrix tracking
    - Multi-temporal context merging
    - Memory resonance integration
    - Adaptive feature projection heads
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        num_regions: int = 8,
        hidden_dim: int = 256,
        pre_activation_history: int = 8,    # M in CTM paper
        sync_pairs: int = 64,               # Subsampled synchronization pairs
        num_feature_heads: int = 8,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_regions = num_regions
        self.region_dim = latent_dim // num_regions
        self.pre_activation_history = pre_activation_history
        self.sync_pairs = sync_pairs
        
        # === Input Processing ===
        self.input_encoder = M4InputEncoder(latent_dim=latent_dim)
        
        # === Regional Neuron-Level Models (CTM-inspired) ===
        # Each region has its own update dynamics
        self.region_models = nn.ModuleList([
            NeuronLevelModel(
                region_dim=self.region_dim,
                history_len=pre_activation_history,
                hidden_dim=hidden_dim // num_regions,
            )
            for _ in range(num_regions)
        ])
        
        # === Synapse Model (cross-region interactions) ===
        self.synapse_model = SynapseModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        
        # === Synchronization Tracking ===
        self.sync_tracker = SynchronizationTracker(
            latent_dim=latent_dim,
            num_regions=num_regions,
            num_pairs=sync_pairs,
        )
        
        # === Temporal Context Merger ===
        self.temporal_merger = TemporalMerger(
            latent_dim=latent_dim,
            indices=[-1, -5, -10, -20],
            learnable=False,  # Start fixed, make learnable later
        )
        
        # === Memory Resonance ===
        self.memory = MemoryResonanceSystem(
            latent_dim=latent_dim,
            max_memories=100000,
        )
        
        # === Feature Projection Heads ===
        self.feature_heads = nn.ModuleDict({
            "is_sound": FeatureHead(latent_dim, sync_pairs, 1, target_step=2),
            "is_speech": FeatureHead(latent_dim, sync_pairs, 1, target_step=4),
            "speaker_id": FeatureHead(latent_dim, sync_pairs, 32, target_step=8),  # 32 speaker embedding
            "affect": FeatureHead(latent_dim, sync_pairs, 8, target_step=12),
        })
        
        # === Certainty Estimator (CTM-inspired) ===
        self.certainty_head = nn.Sequential(
            nn.Linear(latent_dim + sync_pairs, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def step(
        self,
        m4_token: M4Token,
        latent_history: List[torch.Tensor],
        pre_activation_history: List[torch.Tensor],
        post_activation_history: List[torch.Tensor],
        current_time: float,
    ) -> Dict[str, torch.Tensor]:
        """
        One step of the continuous state machine.
        
        Returns dict with:
            - latent: new latent state
            - pre_activation: for history tracking
            - post_activation: for history tracking
            - sync_features: synchronization representation
            - predictions: feature head outputs
            - certainty: model's confidence in current state
        """
        
        # === Encode Input ===
        input_encoding = self.input_encoder(m4_token)
        
        # === Compute Temporal Context ===
        temporal_ctx = self.temporal_merger(latent_history, input_encoding)
        
        # === Memory Resonance ===
        memory_echo = self.memory.retrieve(
            latent_history[-1] if latent_history else torch.zeros(self.latent_dim),
            current_time
        )
        
        # === Combine Context ===
        # Temporal context (70%) + Memory echo (30%)
        context = 0.7 * temporal_ctx + 0.3 * memory_echo
        
        # === Synapse Model: Compute Pre-Activations ===
        if latent_history:
            synapse_input = torch.cat([latent_history[-1], input_encoding], dim=-1)
        else:
            synapse_input = torch.cat([torch.zeros(self.latent_dim), input_encoding], dim=-1)
        
        pre_activation = self.synapse_model(synapse_input)
        
        # === Update Pre-Activation History ===
        pre_activation_history.append(pre_activation)
        if len(pre_activation_history) > self.pre_activation_history:
            pre_activation_history = pre_activation_history[-self.pre_activation_history:]
        
        # Stack history for NLMs: [history_len, latent_dim]
        pre_act_stack = torch.stack(pre_activation_history, dim=0)
        if pre_act_stack.shape[0] < self.pre_activation_history:
            # Pad with zeros if not enough history
            padding = torch.zeros(
                self.pre_activation_history - pre_act_stack.shape[0],
                self.latent_dim
            )
            pre_act_stack = torch.cat([padding, pre_act_stack], dim=0)
        
        # === Regional Neuron-Level Models ===
        # Split into regions, process each with its own NLM
        pre_act_regions = pre_act_stack.chunk(self.num_regions, dim=-1)
        
        post_activation_regions = []
        for region_idx, (region_nlm, region_pre_act) in enumerate(
            zip(self.region_models, pre_act_regions)
        ):
            # region_pre_act: [history_len, region_dim]
            region_post_act = region_nlm(region_pre_act)
            post_activation_regions.append(region_post_act)
        
        # Concatenate regions back to full latent
        post_activation = torch.cat(post_activation_regions, dim=-1)
        
        # === Update Post-Activation History ===
        post_activation_history.append(post_activation)
        
        # === Compute Synchronization Features ===
        sync_features = self.sync_tracker(post_activation_history)
        
        # === Compute New Latent (residual connection) ===
        latent_new = 0.5 * context + 0.5 * post_activation
        
        # === Store in Memory ===
        self.memory.store(latent_new.detach(), current_time)
        
        # === Feature Predictions ===
        predictions = {}
        for name, head in self.feature_heads.items():
            predictions[name] = head(latent_new, sync_features)
        
        # === Certainty Estimation ===
        certainty_input = torch.cat([latent_new, sync_features], dim=-1)
        certainty = self.certainty_head(certainty_input)
        
        return {
            "latent": latent_new,
            "pre_activation": pre_activation,
            "post_activation": post_activation,
            "sync_features": sync_features,
            "predictions": predictions,
            "certainty": certainty,
        }


class NeuronLevelModel(nn.Module):
    """
    CTM-inspired neuron-level model.
    
    Each region has its own private MLP that processes a history
    of incoming pre-activations to produce post-activations.
    """
    
    def __init__(self, region_dim: int, history_len: int, hidden_dim: int):
        super().__init__()
        
        self.region_dim = region_dim
        self.history_len = history_len
        
        # Process temporal history
        self.temporal_net = nn.Sequential(
            nn.Linear(history_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Per-dimension processing
        self.dim_net = nn.Sequential(
            nn.Linear(region_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, region_dim),
        )
    
    def forward(self, pre_activation_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pre_activation_history: [history_len, region_dim]
        
        Returns:
            post_activation: [region_dim]
        """
        # Temporal aggregation: [history_len, region_dim] -> [region_dim]
        temporal_weights = self.temporal_net(pre_activation_history.T)  # [region_dim, 1]
        temporal_weights = F.softmax(temporal_weights, dim=0)
        
        aggregated = (pre_activation_history.T * temporal_weights).sum(dim=-1)  # [region_dim]
        
        # Dimension processing
        post_activation = self.dim_net(aggregated)
        
        return post_activation


class SynapseModel(nn.Module):
    """
    Cross-neuron interaction model (analogous to CTM's synapse model).
    Produces pre-activations from previous state + input.
    """
    
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [latent_dim * 2] - concatenated previous state + input encoding
        
        Returns:
            pre_activation: [latent_dim]
        """
        return self.net(x)


class SynchronizationTracker(nn.Module):
    """
    CTM-inspired synchronization tracking.
    
    Computes pairwise correlations between regions over time,
    subsamples to fixed-size representation.
    """
    
    def __init__(self, latent_dim: int, num_regions: int, num_pairs: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_regions = num_regions
        self.num_pairs = num_pairs
        self.region_dim = latent_dim // num_regions
        
        # Randomly select region pairs to track
        # In full implementation, this could be learned
        all_pairs = []
        for i in range(num_regions):
            for j in range(i, num_regions):
                all_pairs.append((i, j))
        
        # Subsample pairs
        import random
        random.seed(42)  # Reproducible
        self.tracked_pairs = random.sample(all_pairs, min(num_pairs, len(all_pairs)))
        
        # Projection to fixed output dim
        self.output_proj = nn.Linear(len(self.tracked_pairs), num_pairs)
    
    def forward(self, post_activation_history: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute synchronization features from post-activation history.
        
        Args:
            post_activation_history: List of [latent_dim] tensors
        
        Returns:
            sync_features: [num_pairs]
        """
        if len(post_activation_history) < 2:
            return torch.zeros(self.num_pairs)
        
        # Stack history: [T, latent_dim]
        history = torch.stack(post_activation_history, dim=0)
        
        # Split into regions: [T, num_regions, region_dim]
        regions = history.view(history.shape[0], self.num_regions, self.region_dim)
        
        # Compute mean activation per region over time: [num_regions, region_dim]
        region_means = regions.mean(dim=0)
        
        # Compute pairwise correlations for tracked pairs
        sync_values = []
        for i, j in self.tracked_pairs:
            # Correlation between region i and region j over time
            region_i = regions[:, i, :].flatten()  # [T * region_dim]
            region_j = regions[:, j, :].flatten()
            
            # Dot product normalized (cosine similarity)
            corr = F.cosine_similarity(region_i.unsqueeze(0), region_j.unsqueeze(0))
            sync_values.append(corr)
        
        sync_tensor = torch.cat(sync_values)
        
        # Project to fixed output size
        return self.output_proj(sync_tensor)


class FeatureHead(nn.Module):
    """
    Projection head for a specific feature.
    
    Reads from both latent state AND synchronization features.
    Tracks target convergence step for loss weighting.
    """
    
    def __init__(
        self,
        latent_dim: int,
        sync_dim: int,
        output_dim: int,
        target_step: int,
    ):
        super().__init__()
        
        self.target_step = target_step
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + sync_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(
        self,
        latent: torch.Tensor,
        sync_features: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([latent, sync_features], dim=-1)
        return self.net(combined)


class M4InputEncoder(nn.Module):
    """
    Encodes M4 tokens into latent perturbation signal.
    
    Combines:
    - Dense embedding (from modality encoder)
    - Structured features
    - Reflexive annotations
    """
    
    def __init__(self, latent_dim: int, feature_dim: int = 8, annotation_dim: int = 8):
        super().__init__()
        
        self.embedding_proj = nn.Linear(latent_dim, latent_dim)
        self.feature_encoder = nn.Linear(feature_dim, 64)
        self.annotation_encoder = nn.Linear(annotation_dim, 64)
        
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + 64 + 64, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
    
    def forward(self, token: M4Token) -> torch.Tensor:
        tensor_dict = token.to_tensor_dict()
        
        emb = self.embedding_proj(tensor_dict["embedding"])
        feat = self.feature_encoder(tensor_dict["features"])
        ann = self.annotation_encoder(tensor_dict["annotations"])
        
        combined = torch.cat([emb, feat, ann], dim=-1)
        return self.fusion(combined)


class TemporalMerger(nn.Module):
    """
    Merges temporal context from latent history.
    
    Supports fixed, learnable, or input-dependent weights.
    """
    
    def __init__(
        self,
        latent_dim: int,
        indices: List[int] = [-1, -5, -10, -20],
        initial_weights: List[float] = [0.5, 0.25, 0.15, 0.1],
        learnable: bool = False,
        input_dependent: bool = False,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.indices = indices
        
        if input_dependent:
            self.weight_net = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, len(indices)),
            )
            self.weights = None
        elif learnable:
            self.weights = nn.Parameter(torch.tensor(initial_weights))
            self.weight_net = None
        else:
            self.register_buffer("weights", torch.tensor(initial_weights))
            self.weight_net = None
    
    def forward(
        self,
        latent_history: List[torch.Tensor],
        current_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not latent_history:
            return torch.zeros(self.latent_dim)
        
        # Compute weights
        if self.weight_net is not None and current_input is not None:
            weights = F.softmax(self.weight_net(current_input), dim=-1)
        else:
            weights = F.softmax(self.weights, dim=-1)
        
        # Gather and merge
        ctx = torch.zeros(self.latent_dim)
        for w, idx in zip(weights, self.indices):
            if len(latent_history) >= abs(idx):
                ctx = ctx + w * latent_history[idx]
            elif len(latent_history) > 0:
                ctx = ctx + w * latent_history[-1]
        
        return ctx


class MemoryResonanceSystem(nn.Module):
    """
    Passive memory retrieval via embedding similarity.
    
    Current frame embedding continuously queries memory;
    similar frames "echo" back and merge into processing.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        max_memories: int = 100000,
        similarity_threshold: float = 0.7,
        top_k: int = 10,
        recency_halflife: float = 3600.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_memories = max_memories
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.recency_halflife = recency_halflife
        
        # Circular buffer storage
        self.register_buffer("embeddings", torch.zeros(max_memories, latent_dim))
        self.register_buffer("timestamps", torch.zeros(max_memories))
        self.count = 0
        self.index = 0
    
    def store(self, latent: torch.Tensor, timestamp: float):
        self.embeddings[self.index] = latent
        self.timestamps[self.index] = timestamp
        self.index = (self.index + 1) % self.max_memories
        self.count = min(self.count + 1, self.max_memories)
    
    def retrieve(self, query: torch.Tensor, current_time: float) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros(self.latent_dim)
        
        active = self.embeddings[:self.count]
        active_times = self.timestamps[:self.count]
        
        # Cosine similarity
        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        active_norm = F.normalize(active, dim=-1)
        similarities = torch.mm(query_norm, active_norm.T).squeeze(0)
        
        # Recency weighting
        ages = current_time - active_times
        recency = torch.exp(-ages / self.recency_halflife)
        weighted_sim = similarities * recency
        
        # Threshold
        mask = similarities > self.similarity_threshold
        if not mask.any():
            return torch.zeros(self.latent_dim)
        
        # Top-k
        weighted_sim[~mask] = -float("inf")
        k = min(self.top_k, mask.sum().item())
        top_vals, top_idx = torch.topk(weighted_sim, k)
        
        # Weighted average
        weights = F.softmax(top_vals, dim=0)
        echo = (weights.unsqueeze(1) * active[top_idx]).sum(dim=0)
        
        return echo
```

---

## Training System

### Loss Function (CTM-Informed)

```python
class AdaptiveConvergenceLoss(nn.Module):
    """
    Loss function that combines:
    1. Feature-specific convergence targets (our approach)
    2. Adaptive min-loss + max-certainty (CTM approach)
    
    Rewards both being right at the expected time AND knowing when you're ready.
    """
    
    def __init__(
        self,
        feature_configs: Dict[str, Dict] = None,
        use_adaptive: bool = True,
        adaptive_weight: float = 0.3,
    ):
        super().__init__()
        
        self.feature_configs = feature_configs or {
            "is_sound": {"target_step": 2, "loss_fn": "bce", "weight": 1.0},
            "is_speech": {"target_step": 4, "loss_fn": "bce", "weight": 1.0},
            "speaker_id": {"target_step": 8, "loss_fn": "ce", "weight": 1.0},
            "affect": {"target_step": 12, "loss_fn": "mse", "weight": 0.5},
        }
        
        self.use_adaptive = use_adaptive
        self.adaptive_weight = adaptive_weight
    
    def forward(
        self,
        predictions_by_step: List[Dict[str, torch.Tensor]],
        certainties_by_step: List[torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions_by_step: List of {feature: prediction} for each step
            certainties_by_step: List of certainty values for each step
            targets: Ground truth for each feature
        
        Returns:
            Dict with total_loss and diagnostics
        """
        num_steps = len(predictions_by_step)
        
        # === Feature-Specific Convergence Loss ===
        feature_losses = {name: [] for name in self.feature_configs}
        
        for step_idx, preds in enumerate(predictions_by_step):
            for feature_name, config in self.feature_configs.items():
                if feature_name not in targets or targets[feature_name] is None:
                    continue
                
                pred = preds.get(feature_name)
                target = targets[feature_name]
                
                if pred is None:
                    continue
                
                # Compute base loss
                loss_fn = config["loss_fn"]
                if loss_fn == "bce":
                    base_loss = F.binary_cross_entropy_with_logits(pred, target.float())
                elif loss_fn == "ce":
                    base_loss = F.cross_entropy(pred.unsqueeze(0), target.unsqueeze(0))
                else:  # mse
                    base_loss = F.mse_loss(pred, target)
                
                # Weight by step position relative to target
                target_step = config["target_step"]
                if step_idx < target_step:
                    weight = 0.0  # Not expected to know yet
                elif step_idx == target_step:
                    weight = config["weight"]  # Full weight at target
                else:
                    weight = config["weight"] * 0.3  # Maintain, don't regress
                
                feature_losses[feature_name].append(weight * base_loss)
        
        # Sum feature losses
        convergence_loss = sum(
            sum(losses) for losses in feature_losses.values() if losses
        )
        
        # === Adaptive Loss (CTM-inspired) ===
        if self.use_adaptive and certainties_by_step:
            certainties = torch.stack(certainties_by_step)
            
            # Find step with max certainty
            t_max_cert = certainties.argmax().item()
            
            # Compute total loss at each step
            step_losses = []
            for step_idx, preds in enumerate(predictions_by_step):
                step_loss = 0
                for feature_name, config in self.feature_configs.items():
                    if feature_name not in targets or targets[feature_name] is None:
                        continue
                    pred = preds.get(feature_name)
                    target = targets[feature_name]
                    if pred is not None:
                        if config["loss_fn"] == "bce":
                            step_loss += F.binary_cross_entropy_with_logits(pred, target.float())
                        elif config["loss_fn"] == "ce":
                            step_loss += F.cross_entropy(pred.unsqueeze(0), target.unsqueeze(0))
                        else:
                            step_loss += F.mse_loss(pred, target)
                step_losses.append(step_loss)
            
            # Find step with min loss
            step_losses_tensor = torch.stack([torch.tensor(l) for l in step_losses])
            t_min_loss = step_losses_tensor.argmin().item()
            
            # Adaptive loss = average of loss at min-loss step and max-certainty step
            adaptive_loss = (step_losses[t_min_loss] + step_losses[t_max_cert]) / 2
            
            total_loss = (1 - self.adaptive_weight) * convergence_loss + self.adaptive_weight * adaptive_loss
        else:
            total_loss = convergence_loss
            adaptive_loss = torch.tensor(0.0)
        
        return {
            "total_loss": total_loss,
            "convergence_loss": convergence_loss,
            "adaptive_loss": adaptive_loss,
            "feature_losses": {k: sum(v) if v else 0 for k, v in feature_losses.items()},
        }
```

### Training Loop

```python
def train_epoch(
    model: SubconsciousLayer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: AdaptiveConvergenceLoss,
    device: torch.device,
    max_steps_per_sequence: int = 20,
) -> Dict[str, float]:
    """
    Train for one epoch on temporal sequences.
    """
    model.train()
    
    epoch_losses = []
    epoch_certainties = []
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        batch_loss = 0
        
        for sequence in batch:  # Process each sequence independently
            # Initialize histories
            latent_history = []
            pre_activation_history = []
            post_activation_history = []
            
            predictions_by_step = []
            certainties_by_step = []
            
            for step_idx, (m4_token, labels) in enumerate(sequence):
                if step_idx >= max_steps_per_sequence:
                    break
                
                m4_token = m4_token.to(device)
                current_time = m4_token.timestamp
                
                # Step the model
                outputs = model.step(
                    m4_token=m4_token,
                    latent_history=latent_history,
                    pre_activation_history=pre_activation_history,
                    post_activation_history=post_activation_history,
                    current_time=current_time,
                )
                
                # Update histories
                latent_history.append(outputs["latent"])
                pre_activation_history.append(outputs["pre_activation"])
                post_activation_history.append(outputs["post_activation"])
                
                # Keep bounded
                latent_history = latent_history[-50:]
                pre_activation_history = pre_activation_history[-model.pre_activation_history:]
                # post_activation_history grows (needed for sync)
                
                predictions_by_step.append(outputs["predictions"])
                certainties_by_step.append(outputs["certainty"])
            
            # Compute loss for this sequence
            loss_dict = loss_fn(
                predictions_by_step=predictions_by_step,
                certainties_by_step=certainties_by_step,
                targets=sequence.targets,
            )
            
            batch_loss += loss_dict["total_loss"]
            epoch_losses.append(loss_dict["total_loss"].item())
            epoch_certainties.append(
                torch.stack(certainties_by_step).mean().item()
            )
        
        # Backprop
        batch_loss.backward()
        optimizer.step()
    
    return {
        "mean_loss": sum(epoch_losses) / len(epoch_losses),
        "mean_certainty": sum(epoch_certainties) / len(epoch_certainties),
    }
```

---

## Training Data Pipeline

### Building Temporal Sequences

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class TemporalSequence:
    """A sequence of M4 tokens with temporal labels."""
    
    tokens: List[M4Token]
    targets: Dict[str, torch.Tensor]  # Ground truth for the sequence
    
    def __len__(self):
        return len(self.tokens)
    
    def __iter__(self):
        # Yield (token, step_labels) pairs
        for idx, token in enumerate(self.tokens):
            step_labels = self._get_labels_for_step(idx)
            yield token, step_labels
    
    def _get_labels_for_step(self, step_idx: int) -> Dict[str, Optional[torch.Tensor]]:
        """Get labels that are eligible at this step."""
        labels = {}
        
        # is_sound: eligible immediately
        labels["is_sound"] = self.targets.get("is_sound")
        
        # is_speech: eligible after step 2 (150ms at 50ms/step)
        labels["is_speech"] = self.targets.get("is_speech") if step_idx >= 2 else None
        
        # speaker_id: eligible after step 6 (300ms)
        labels["speaker_id"] = self.targets.get("speaker_id") if step_idx >= 6 else None
        
        # affect: eligible after step 10 (500ms)
        labels["affect"] = self.targets.get("affect") if step_idx >= 10 else None
        
        return labels


def build_sequences_from_segments(
    segments: List[Dict],
    audio_encoder,  # Your hybrid encoder
    window_size: int = 20,
    stride: int = 5,
    frame_duration_ms: float = 50.0,
) -> List[TemporalSequence]:
    """
    Convert enriched segments to temporal sequences.
    
    Args:
        segments: List of enriched segment dicts with audio, labels, timestamps
        audio_encoder: Encoder to create embeddings
        window_size: Frames per sequence
        stride: Overlap between sequences
        frame_duration_ms: Duration each frame represents
    
    Returns:
        List of TemporalSequence objects
    """
    # Sort by timestamp
    sorted_segments = sorted(segments, key=lambda s: s["timestamp"])
    
    sequences = []
    
    for i in range(0, len(sorted_segments) - window_size, stride):
        window = sorted_segments[i : i + window_size]
        
        # Build M4 tokens for each frame
        tokens = []
        for seg in window:
            # Encode audio
            with torch.no_grad():
                embedding = audio_encoder(
                    torch.tensor(seg["encodec_tokens"]).unsqueeze(0),
                    torch.tensor(list(seg["acoustic_features"].values())).unsqueeze(0),
                ).squeeze(0)
            
            token = M4Token(
                modality="audio",
                source="ambient_mic",
                timestamp=seg["timestamp"],
                duration_ms=frame_duration_ms,
                embedding=embedding,
                features=seg["acoustic_features"],
                annotations={},  # Reflexive layer would populate these
                raw_tokens=seg["encodec_tokens"],
            )
            tokens.append(token)
        
        # Use last segment's labels as sequence target
        # (Could also aggregate or use majority vote)
        last_seg = window[-1]
        targets = {
            "is_sound": torch.tensor([float(last_seg.get("has_sound", 0))]),
            "is_speech": torch.tensor([float(last_seg.get("is_speech", 0))]),
            "speaker_id": torch.tensor([last_seg.get("speaker_id", 0)]) if last_seg.get("speaker_id") else None,
            "affect": torch.tensor(last_seg.get("affect", [0] * 8)) if last_seg.get("affect") else None,
        }
        
        sequences.append(TemporalSequence(tokens=tokens, targets=targets))
    
    return sequences
```

---

## Cold Boot and Warmup

```python
class SystemBootManager:
    """
    Manages cold boot, warmup, and state persistence.
    """
    
    def __init__(
        self,
        model: SubconsciousLayer,
        warmup_steps: int = 20,
        state_path: Optional[str] = None,
    ):
        self.model = model
        self.warmup_steps = warmup_steps
        self.state_path = state_path
        
        self.step_count = 0
        self.is_warm = False
        
        # Histories
        self.latent_history: List[torch.Tensor] = []
        self.pre_activation_history: List[torch.Tensor] = []
        self.post_activation_history: List[torch.Tensor] = []
    
    def cold_start(self):
        """Initialize from scratch."""
        self.step_count = 0
        self.is_warm = False
        
        # Bootstrap with zeros
        zero_latent = torch.zeros(self.model.latent_dim)
        self.latent_history = [zero_latent.clone() for _ in range(10)]
        self.pre_activation_history = []
        self.post_activation_history = []
    
    def warm_start(self):
        """Attempt to restore from saved state."""
        if self.state_path and os.path.exists(self.state_path):
            state = torch.load(self.state_path)
            self.latent_history = state["latent_history"]
            self.pre_activation_history = state["pre_activation_history"]
            self.post_activation_history = state["post_activation_history"]
            self.step_count = state["step_count"]
            self.is_warm = True
        else:
            self.cold_start()
    
    def step(self, m4_token: M4Token, current_time: float) -> Dict:
        """Process one token and update state."""
        outputs = self.model.step(
            m4_token=m4_token,
            latent_history=self.latent_history,
            pre_activation_history=self.pre_activation_history,
            post_activation_history=self.post_activation_history,
            current_time=current_time,
        )
        
        # Update histories
        self.latent_history.append(outputs["latent"])
        self.pre_activation_history.append(outputs["pre_activation"])
        self.post_activation_history.append(outputs["post_activation"])
        
        # Bound histories
        self.latent_history = self.latent_history[-100:]
        self.pre_activation_history = self.pre_activation_history[-self.model.pre_activation_history:]
        # post_activation_history grows for sync computation
        
        self.step_count += 1
        
        if not self.is_warm and self.step_count >= self.warmup_steps:
            self.is_warm = True
        
        # Add warmup status to outputs
        outputs["warmup_status"] = {
            "step_count": self.step_count,
            "is_warm": self.is_warm,
            "warmup_progress": min(1.0, self.step_count / self.warmup_steps),
        }
        
        return outputs
    
    def save_state(self):
        """Persist current state for warm restart."""
        if self.state_path:
            torch.save({
                "latent_history": self.latent_history,
                "pre_activation_history": self.pre_activation_history,
                "post_activation_history": self.post_activation_history,
                "step_count": self.step_count,
            }, self.state_path)
    
    def get_confidence_scaling(self) -> Dict[str, float]:
        """Scale confidence by warmup progress."""
        elapsed_ms = self.step_count * 50  # Assuming 50ms steps
        
        return {
            "is_sound": min(1.0, elapsed_ms / 100),
            "is_speech": min(1.0, elapsed_ms / 200),
            "speaker_id": min(1.0, elapsed_ms / 400),
            "affect": min(1.0, elapsed_ms / 800),
        }
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

1. **M4Token dataclass** — Implement and test
2. **M4InputEncoder** — Port from existing hybrid encoder
3. **Basic SynapseModel** — Simple MLP, no regional processing
4. **Basic TemporalMerger** — Fixed weights
5. **Simple feature heads** — is_sound only initially

Validation: Can the system learn is_sound from your existing training data?

### Phase 2: CTM Features (Week 3-4)

1. **NeuronLevelModels** — Regional processing with private weights
2. **SynchronizationTracker** — Pairwise correlation tracking
3. **Update feature heads** — Read from latent + sync features
4. **Certainty estimation** — Add certainty head

Validation: Does synchronization improve feature prediction? Does certainty correlate with accuracy?

### Phase 3: Full Pipeline (Week 5-6)

1. **AdaptiveConvergenceLoss** — Combined convergence + adaptive loss
2. **MemoryResonanceSystem** — Passive retrieval
3. **SystemBootManager** — Cold/warm start, state persistence
4. **Temporal sequence pipeline** — Build from enriched segments

Validation: Full training loop on sequences, measure convergence timing.

### Phase 4: Integration (Week 7-8)

1. **Reflexive layer integration** — Fast classifiers populate M4 annotations
2. **A/B hot-swap** — Resume work on model swapping
3. **Executive layer interface** — Define how executive reads from subconscious
4. **Real-time inference** — Benchmark latency, optimize

Validation: End-to-end system running on live audio input.

---

## Key Differences from CTM

| Aspect | CTM | Our Approach |
|--------|-----|--------------|
| Input | Single modality (image, maze) | Multimodal (M4 tokens) |
| Memory | Within-pass only | Long-term resonance system |
| Convergence | Adaptive (model finds timing) | Hybrid (targets + adaptive) |
| Output | Classification, path | Feature projections + certainty |
| Timing goal | Compute efficiency | Human-plausible timing |
| Neuron diversity | Per-neuron MLPs | Regional MLPs (compromise) |

---

## Open Questions for Future Investigation

1. **Optimal region count**: 4? 8? 16? More regions = more diversity but more parameters.

2. **Sync pair selection**: Random vs. learned? Could attention over sync pairs improve?

3. **Memory-sync interaction**: Should memory echoes contribute to synchronization computation?

4. **Drift dynamics**: When input starves, how should sync evolve? Decay? Oscillate?

5. **Cross-modal sync**: When vision and audio both present, should their regions sync?

---

## References

- Sakana AI. (2025). Continuous Thought Machines. https://pub.sakana.ai/ctm/
- Technical Report: https://arxiv.org/abs/2505.05522
- GitHub: https://github.com/SakanaAI/continuous-thought-machines