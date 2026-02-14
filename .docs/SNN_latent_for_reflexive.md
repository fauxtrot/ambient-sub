# Reflexive Template System: SNN-Based Fast Classification Across Modalities

## Overview

This document describes a generalizable architecture pattern for fast, adaptive classification in the reflexive layer. The pattern uses Spiking Neural Networks (SNNs) as efficient feature encoders combined with dynamic template matching for few-shot learning within sessions.

The core insight: separate the **encoder** (frozen, pretrained, expensive to train) from the **templates** (dynamic, few-shot, cheap to update). This allows rapid adaptation to new classes/instances without retraining the underlying network.

---

## The Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     REFLEXIVE TEMPLATE SYSTEM                               │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 1: STATIC CATEGORIES                         │ │
│  │                    (Frozen, Pretrained SNN)                           │ │
│  │                                                                       │ │
│  │  Input → SNN Encoder → Spike Pattern → Category Classifiers          │ │
│  │                                                                       │ │
│  │  Answers broad, stable questions:                                     │ │
│  │  - Audio: is_sound, is_speech, is_music, is_male, is_female          │ │
│  │  - Vision: is_human, is_face, is_motion, is_indoor                   │ │
│  │  - Text: is_question, is_command, is_statement                       │ │
│  │                                                                       │ │
│  │  These never change. Trained once on large datasets.                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    ↓                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 2: KNOWN TEMPLATES                           │ │
│  │                    (Persistent, Slowly Updated)                       │ │
│  │                                                                       │ │
│  │  Spike Pattern → Compare to Stored Templates → Match / No Match      │ │
│  │                                                                       │ │
│  │  Long-term memory of known instances:                                 │ │
│  │  - Audio: Todd's voice, Lars's voice, TV speaker signature           │ │
│  │  - Vision: Todd's face, Lars's face, the cat, the coffee mug         │ │
│  │  - Text: Known command patterns, user's writing style                │ │
│  │                                                                       │ │
│  │  Updated slowly via promotion from session templates.                 │ │
│  │  High threshold for matching (confident recognition only).           │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    ↓                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    LAYER 3: SESSION TEMPLATES                         │ │
│  │                    (Ephemeral, Fast Learning)                         │ │
│  │                                                                       │ │
│  │  Spike Pattern → Compare to Session Templates → Match / Unknown      │ │
│  │                                                                       │ │
│  │  Created on-the-fly when heavy system identifies new instance:        │ │
│  │  - Audio: "Speaker_A" (confirmed by diart), new voice on TV          │ │
│  │  - Vision: "Person_B" (confirmed by face recognition), new object    │ │
│  │  - Text: New named entity, new topic cluster                         │ │
│  │                                                                       │ │
│  │  Reinforced throughout session. Discarded or promoted at end.        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    ↓                                        │
│                           OUTPUT: Annotations                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. SNN Encoder (Frozen, Pretrained)

The encoder converts raw input into spike representations. It is:

- **Modality-specific**: Different encoder architectures for audio, vision, text
- **Trained once**: Large dataset, expensive training, then frozen
- **Temporally-aware**: Captures timing information in spike patterns
- **Reusable**: Same encoder feeds all three layers

```python
class SNNEncoder(nn.Module):
    """
    Base class for modality-specific SNN encoders.
    Converts raw input to spike representation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        spike_dim: int,
        num_steps: int = 20,        # Temporal resolution
        threshold: float = 1.0,      # Spike threshold
        decay: float = 0.9,          # Membrane potential decay
    ):
        super().__init__()
        
        self.num_steps = num_steps
        self.threshold = threshold
        self.decay = decay
        self.spike_dim = spike_dim
        
        # Encoding layers (architecture varies by modality)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # LIF (Leaky Integrate-and-Fire) neuron parameters
        self.lif_weights = nn.Linear(hidden_dim, spike_dim)
    
    def forward(self, x: torch.Tensor) -> SpikePattern:
        """
        Encode input to spike pattern.
        
        Args:
            x: Raw input tensor
        
        Returns:
            SpikePattern with temporal spike information
        """
        # Initial encoding
        encoded = self.encoder(x)
        
        # LIF dynamics
        membrane = torch.zeros(x.shape[0], self.spike_dim)
        spikes = []
        spike_times = []
        
        for t in range(self.num_steps):
            # Integrate input
            membrane = self.decay * membrane + self.lif_weights(encoded)
            
            # Fire if above threshold
            spike = (membrane > self.threshold).float()
            spikes.append(spike)
            
            # Record spike times
            spike_times.append(spike * t)
            
            # Reset neurons that fired
            membrane = membrane * (1 - spike)
        
        spike_train = torch.stack(spikes, dim=1)  # [batch, time, neurons]
        
        return SpikePattern(
            spike_train=spike_train,
            spike_count=spike_train.sum(dim=1),  # [batch, neurons]
            first_spike_time=self._first_spike_time(spike_train),
            spike_rate=spike_train.mean(dim=1),
        )
    
    def _first_spike_time(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Get time of first spike for each neuron (or max_time if none)."""
        # spike_train: [batch, time, neurons]
        first_spike = torch.argmax(spike_train, dim=1).float()
        no_spike_mask = spike_train.sum(dim=1) == 0
        first_spike[no_spike_mask] = self.num_steps
        return first_spike


@dataclass
class SpikePattern:
    """
    Spike representation of an input.
    Contains multiple views of the same spike data for different uses.
    """
    spike_train: torch.Tensor      # [batch, time, neurons] - full temporal pattern
    spike_count: torch.Tensor      # [batch, neurons] - total spikes per neuron
    first_spike_time: torch.Tensor # [batch, neurons] - latency encoding
    spike_rate: torch.Tensor       # [batch, neurons] - firing rate
    
    def to_template_vector(self) -> torch.Tensor:
        """
        Flatten to vector for template storage/comparison.
        Combines multiple spike statistics for richer representation.
        """
        return torch.cat([
            self.spike_count.flatten(),
            self.first_spike_time.flatten(),
            self.spike_rate.flatten(),
        ], dim=-1)
```

### 2. Category Classifiers (Layer 1)

Fast, binary/multi-class classifiers on top of the frozen encoder.

```python
class CategoryClassifier(nn.Module):
    """
    Lightweight classifier head for static categories.
    Trained with encoder, then both frozen.
    """
    
    def __init__(
        self,
        spike_dim: int,
        num_classes: int,
        category_name: str,
    ):
        super().__init__()
        
        self.category_name = category_name
        self.num_classes = num_classes
        
        # Simple linear classifier on spike features
        # Uses spike_count + first_spike_time for temporal info
        self.classifier = nn.Sequential(
            nn.Linear(spike_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, spike_pattern: SpikePattern) -> Dict[str, Any]:
        features = torch.cat([
            spike_pattern.spike_count,
            spike_pattern.first_spike_time,
        ], dim=-1)
        
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        
        return {
            "category": self.category_name,
            "prediction": logits.argmax(dim=-1),
            "confidence": probs.max(dim=-1).values,
            "probabilities": probs,
        }


class CategoryBank:
    """
    Collection of category classifiers for a modality.
    """
    
    def __init__(self, encoder: SNNEncoder):
        self.encoder = encoder
        self.classifiers: Dict[str, CategoryClassifier] = {}
    
    def add_classifier(self, classifier: CategoryClassifier):
        self.classifiers[classifier.category_name] = classifier
    
    def classify(self, raw_input: torch.Tensor) -> Dict[str, Dict]:
        """Run all category classifiers on input."""
        spike_pattern = self.encoder(raw_input)
        
        results = {}
        for name, classifier in self.classifiers.items():
            results[name] = classifier(spike_pattern)
        
        return results


# Example: Audio category bank
def build_audio_category_bank(encoder: SNNEncoder) -> CategoryBank:
    bank = CategoryBank(encoder)
    
    bank.add_classifier(CategoryClassifier(
        spike_dim=encoder.spike_dim,
        num_classes=2,
        category_name="is_sound",
    ))
    
    bank.add_classifier(CategoryClassifier(
        spike_dim=encoder.spike_dim,
        num_classes=2,
        category_name="is_speech",
    ))
    
    bank.add_classifier(CategoryClassifier(
        spike_dim=encoder.spike_dim,
        num_classes=3,  # male, female, child
        category_name="voice_type",
    ))
    
    bank.add_classifier(CategoryClassifier(
        spike_dim=encoder.spike_dim,
        num_classes=2,
        category_name="is_music",
    ))
    
    return bank
```

### 3. Template Store (Layers 2 & 3)

Dynamic template storage and matching.

```python
@dataclass
class SpikeTemplate:
    """
    A stored spike pattern template for instance matching.
    """
    id: str
    pattern: torch.Tensor           # Aggregated spike representation
    variance: torch.Tensor          # Per-dimension variance (for tolerance)
    threshold: float                # Minimum similarity for match
    
    # Metadata
    category_constraints: Dict[str, Any]  # e.g., {"voice_type": "male"}
    sample_count: int               # How many samples built this template
    confidence: float               # How reliable is this template
    
    # Temporal
    created_at: float
    last_matched: float
    match_count: int


class TemplateStore:
    """
    Storage and matching for spike templates.
    Supports both persistent (known) and ephemeral (session) templates.
    """
    
    def __init__(
        self,
        encoder: SNNEncoder,
        similarity_metric: str = "cosine",
    ):
        self.encoder = encoder
        self.similarity_metric = similarity_metric
        
        # Persistent templates (loaded from disk, survive sessions)
        self.known_templates: Dict[str, SpikeTemplate] = {}
        
        # Session templates (created during session, ephemeral)
        self.session_templates: Dict[str, SpikeTemplate] = {}
    
    def match(
        self,
        spike_pattern: SpikePattern,
        category_filter: Optional[Dict[str, Any]] = None,
        include_session: bool = True,
    ) -> Optional[TemplateMatch]:
        """
        Find best matching template for a spike pattern.
        
        Args:
            spike_pattern: The pattern to match
            category_filter: Only consider templates matching these categories
            include_session: Whether to check session templates
        
        Returns:
            TemplateMatch if found, None if no match above threshold
        """
        pattern_vec = spike_pattern.to_template_vector()
        
        best_match = None
        best_similarity = 0.0
        
        # Check known templates first (higher priority)
        for template in self.known_templates.values():
            if not self._category_match(template, category_filter):
                continue
            
            similarity = self._compute_similarity(pattern_vec, template)
            
            if similarity > template.threshold and similarity > best_similarity:
                best_match = template
                best_similarity = similarity
        
        # Check session templates if no known match (or if session has better match)
        if include_session:
            for template in self.session_templates.values():
                if not self._category_match(template, category_filter):
                    continue
                
                similarity = self._compute_similarity(pattern_vec, template)
                
                if similarity > template.threshold and similarity > best_similarity:
                    best_match = template
                    best_similarity = similarity
        
        if best_match is None:
            return None
        
        return TemplateMatch(
            template_id=best_match.id,
            similarity=best_similarity,
            source="known" if best_match.id in self.known_templates else "session",
            template=best_match,
        )
    
    def create_template(
        self,
        template_id: str,
        samples: List[torch.Tensor],
        category_constraints: Dict[str, Any],
        persistent: bool = False,
        initial_threshold: float = 0.6,
    ) -> SpikeTemplate:
        """
        Create a new template from samples.
        
        Args:
            template_id: Unique identifier
            samples: List of raw input samples
            category_constraints: Category metadata for filtering
            persistent: If True, add to known_templates; else session_templates
            initial_threshold: Starting similarity threshold
        """
        # Encode all samples
        spike_patterns = [self.encoder(s) for s in samples]
        pattern_vecs = [sp.to_template_vector() for sp in spike_patterns]
        
        # Aggregate (mean)
        stacked = torch.stack(pattern_vecs)
        mean_pattern = stacked.mean(dim=0)
        variance = stacked.var(dim=0)
        
        template = SpikeTemplate(
            id=template_id,
            pattern=mean_pattern,
            variance=variance,
            threshold=initial_threshold,
            category_constraints=category_constraints,
            sample_count=len(samples),
            confidence=min(0.5 + len(samples) * 0.05, 0.95),
            created_at=time.time(),
            last_matched=time.time(),
            match_count=0,
        )
        
        if persistent:
            self.known_templates[template_id] = template
        else:
            self.session_templates[template_id] = template
        
        return template
    
    def reinforce_template(
        self,
        template_id: str,
        new_sample: torch.Tensor,
        learning_rate: float = 0.1,
    ):
        """
        Update template with new confirmed sample.
        Tightens threshold as confidence grows.
        """
        # Find template
        template = self.session_templates.get(template_id) or self.known_templates.get(template_id)
        if template is None:
            return
        
        # Encode new sample
        spike_pattern = self.encoder(new_sample)
        new_vec = spike_pattern.to_template_vector()
        
        # Exponential moving average update
        template.pattern = (1 - learning_rate) * template.pattern + learning_rate * new_vec
        
        # Update statistics
        template.sample_count += 1
        template.last_matched = time.time()
        template.match_count += 1
        template.confidence = min(0.5 + template.sample_count * 0.05, 0.95)
        
        # Tighten threshold as confidence grows
        template.threshold = 0.6 + template.confidence * 0.25  # 0.6 → 0.85
    
    def _compute_similarity(
        self,
        pattern_vec: torch.Tensor,
        template: SpikeTemplate,
    ) -> float:
        """Compute similarity between pattern and template."""
        if self.similarity_metric == "cosine":
            return F.cosine_similarity(
                pattern_vec.unsqueeze(0),
                template.pattern.unsqueeze(0)
            ).item()
        elif self.similarity_metric == "euclidean":
            dist = torch.dist(pattern_vec, template.pattern)
            return 1.0 / (1.0 + dist.item())
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def _category_match(
        self,
        template: SpikeTemplate,
        filter: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if template matches category filter."""
        if filter is None:
            return True
        
        for key, value in filter.items():
            if key in template.category_constraints:
                if template.category_constraints[key] != value:
                    return False
        
        return True
    
    # === Persistence ===
    
    def save_known_templates(self, path: str):
        """Save persistent templates to disk."""
        torch.save({
            name: {
                "pattern": t.pattern,
                "variance": t.variance,
                "threshold": t.threshold,
                "category_constraints": t.category_constraints,
                "sample_count": t.sample_count,
                "confidence": t.confidence,
            }
            for name, t in self.known_templates.items()
        }, path)
    
    def load_known_templates(self, path: str):
        """Load persistent templates from disk."""
        data = torch.load(path)
        for name, d in data.items():
            self.known_templates[name] = SpikeTemplate(
                id=name,
                pattern=d["pattern"],
                variance=d["variance"],
                threshold=d["threshold"],
                category_constraints=d["category_constraints"],
                sample_count=d["sample_count"],
                confidence=d["confidence"],
                created_at=time.time(),
                last_matched=time.time(),
                match_count=0,
            )
    
    # === Session Management ===
    
    def clear_session(self):
        """Clear all session templates."""
        self.session_templates.clear()
    
    def promote_to_known(self, template_id: str) -> bool:
        """
        Promote a session template to known (persistent).
        Returns True if successful.
        """
        if template_id not in self.session_templates:
            return False
        
        template = self.session_templates.pop(template_id)
        self.known_templates[template_id] = template
        return True
    
    def get_promotable_templates(
        self,
        min_samples: int = 20,
        min_confidence: float = 0.85,
    ) -> List[str]:
        """Get session templates eligible for promotion."""
        return [
            t.id for t in self.session_templates.values()
            if t.sample_count >= min_samples and t.confidence >= min_confidence
        ]


@dataclass
class TemplateMatch:
    """Result of a template matching operation."""
    template_id: str
    similarity: float
    source: str  # "known" or "session"
    template: SpikeTemplate
```

### 4. Memory Correlation System

Links template activations to the continuous latent state machine.

```python
class ReflexiveMemoryBridge:
    """
    Bridges reflexive template system with subconscious memory.
    
    Tracks correlations between:
    - Template activations (who's speaking, what's visible)
    - Latent state dynamics (what was the subconscious doing)
    
    Enables queries like:
    - "What was happening when Todd last spoke?"
    - "What latent states correlate with music listening?"
    """
    
    def __init__(
        self,
        latent_dim: int,
        max_correlations: int = 10000,
    ):
        self.latent_dim = latent_dim
        self.max_correlations = max_correlations
        
        # Circular buffer of (template_activation, latent_state, timestamp)
        self.correlation_buffer: List[CorrelationEntry] = []
        self.buffer_idx = 0
        
        # Aggregated correlations per template
        self.template_latent_associations: Dict[str, LatentAssociation] = {}
    
    def record(
        self,
        template_match: Optional[TemplateMatch],
        category_results: Dict[str, Dict],
        latent_state: torch.Tensor,
        timestamp: float,
    ):
        """
        Record a correlation between reflexive output and latent state.
        """
        entry = CorrelationEntry(
            template_id=template_match.template_id if template_match else None,
            template_similarity=template_match.similarity if template_match else 0.0,
            categories={k: v["prediction"] for k, v in category_results.items()},
            latent_state=latent_state.detach().clone(),
            timestamp=timestamp,
        )
        
        # Store in buffer
        if len(self.correlation_buffer) < self.max_correlations:
            self.correlation_buffer.append(entry)
        else:
            self.correlation_buffer[self.buffer_idx] = entry
            self.buffer_idx = (self.buffer_idx + 1) % self.max_correlations
        
        # Update aggregated associations
        if template_match:
            self._update_association(template_match.template_id, latent_state)
    
    def _update_association(self, template_id: str, latent_state: torch.Tensor):
        """Update running association between template and latent states."""
        if template_id not in self.template_latent_associations:
            self.template_latent_associations[template_id] = LatentAssociation(
                template_id=template_id,
                mean_latent=latent_state.clone(),
                count=1,
            )
        else:
            assoc = self.template_latent_associations[template_id]
            # Running mean
            assoc.count += 1
            alpha = 1.0 / assoc.count
            assoc.mean_latent = (1 - alpha) * assoc.mean_latent + alpha * latent_state
    
    def get_template_context(self, template_id: str) -> Optional[torch.Tensor]:
        """
        Get the typical latent context when this template is active.
        Useful for memory resonance queries.
        """
        assoc = self.template_latent_associations.get(template_id)
        if assoc is None:
            return None
        return assoc.mean_latent
    
    def query_by_latent(
        self,
        latent_state: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find templates whose typical context is similar to given latent.
        "What templates are associated with this kind of mental state?"
        """
        results = []
        
        for template_id, assoc in self.template_latent_associations.items():
            similarity = F.cosine_similarity(
                latent_state.unsqueeze(0),
                assoc.mean_latent.unsqueeze(0)
            ).item()
            results.append((template_id, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


@dataclass
class CorrelationEntry:
    """Single correlation record."""
    template_id: Optional[str]
    template_similarity: float
    categories: Dict[str, int]
    latent_state: torch.Tensor
    timestamp: float


@dataclass
class LatentAssociation:
    """Aggregated association between a template and latent states."""
    template_id: str
    mean_latent: torch.Tensor
    count: int
```

---

## Session Context

Manages the temporal/conversational context for template matching.

```python
@dataclass
class SessionContext:
    """
    Session-level context for adaptive template matching.
    """
    session_id: str
    start_time: float
    
    # Active modalities
    active_modalities: Set[str] = field(default_factory=lambda: {"audio"})
    
    # Speaker context (audio-specific)
    expected_speakers: List[str] = field(default_factory=list)
    identified_speakers: Dict[str, SpeakerSession] = field(default_factory=dict)
    
    # Visual context
    expected_faces: List[str] = field(default_factory=list)
    identified_objects: Dict[str, ObjectSession] = field(default_factory=dict)
    
    # Environment
    environment: str = "unknown"
    ambient_profile: Optional[str] = None  # "quiet_home", "office", "noisy"
    
    # Global tolerance modifiers
    tolerance_modifier: float = 0.0  # Adjust all thresholds
    
    # Listening mode (affects which categories are prioritized)
    listening_mode: str = "ambient"  # "ambient", "active_conversation", "music"


@dataclass
class SpeakerSession:
    """Per-speaker tracking within a session."""
    speaker_id: str
    voice_type: str  # "male", "female", "child"
    template_id: Optional[str]  # If template exists
    
    first_seen: float
    last_seen: float
    speaking_time: float
    utterance_count: int
    
    confirmed_by: str  # "known_template", "diart", "user"
    confidence: float


@dataclass 
class ObjectSession:
    """Per-object tracking within a session (vision)."""
    object_id: str
    object_class: str
    template_id: Optional[str]
    
    first_seen: float
    last_seen: float
    detection_count: int
    
    bounding_box: Optional[Tuple[int, int, int, int]]  # Last known location


class SessionManager:
    """
    Manages session context and coordinates template systems across modalities.
    """
    
    def __init__(self):
        self.current_session: Optional[SessionContext] = None
        self.template_stores: Dict[str, TemplateStore] = {}  # modality → store
    
    def start_session(
        self,
        session_id: Optional[str] = None,
        expected_speakers: List[str] = None,
        environment: str = "unknown",
        listening_mode: str = "ambient",
    ) -> SessionContext:
        """Start a new session."""
        self.current_session = SessionContext(
            session_id=session_id or str(uuid.uuid4()),
            start_time=time.time(),
            expected_speakers=expected_speakers or [],
            environment=environment,
            listening_mode=listening_mode,
        )
        
        # Clear session templates from all modalities
        for store in self.template_stores.values():
            store.clear_session()
        
        return self.current_session
    
    def end_session(self) -> Dict[str, Any]:
        """
        End current session.
        Returns summary and list of templates eligible for promotion.
        """
        if self.current_session is None:
            return {}
        
        summary = {
            "session_id": self.current_session.session_id,
            "duration": time.time() - self.current_session.start_time,
            "speakers_identified": list(self.current_session.identified_speakers.keys()),
            "promotable_templates": {},
        }
        
        # Check each modality for promotable templates
        for modality, store in self.template_stores.items():
            promotable = store.get_promotable_templates()
            if promotable:
                summary["promotable_templates"][modality] = promotable
        
        self.current_session = None
        return summary
    
    def register_speaker(
        self,
        speaker_id: str,
        voice_type: str,
        samples: List[torch.Tensor],
        confirmed_by: str = "diart",
    ):
        """
        Register a newly identified speaker in the session.
        Creates session template for future fast matching.
        """
        if self.current_session is None:
            return
        
        audio_store = self.template_stores.get("audio")
        if audio_store is None:
            return
        
        # Create template
        template = audio_store.create_template(
            template_id=f"session_{speaker_id}",
            samples=samples,
            category_constraints={"voice_type": voice_type},
            persistent=False,
            initial_threshold=0.6,
        )
        
        # Track in session
        self.current_session.identified_speakers[speaker_id] = SpeakerSession(
            speaker_id=speaker_id,
            voice_type=voice_type,
            template_id=template.id,
            first_seen=time.time(),
            last_seen=time.time(),
            speaking_time=0.0,
            utterance_count=1,
            confirmed_by=confirmed_by,
            confidence=0.5,
        )
    
    def update_speaker(
        self,
        speaker_id: str,
        new_sample: torch.Tensor,
        speaking_duration: float = 0.0,
    ):
        """Update speaker tracking with new confirmed sample."""
        if self.current_session is None:
            return
        
        speaker = self.current_session.identified_speakers.get(speaker_id)
        if speaker is None:
            return
        
        # Reinforce template
        audio_store = self.template_stores.get("audio")
        if audio_store and speaker.template_id:
            audio_store.reinforce_template(speaker.template_id, new_sample)
        
        # Update session tracking
        speaker.last_seen = time.time()
        speaker.speaking_time += speaking_duration
        speaker.utterance_count += 1
        speaker.confidence = min(0.95, speaker.confidence + 0.02)
    
    def set_listening_mode(self, mode: str):
        """Change the current listening mode."""
        if self.current_session:
            self.current_session.listening_mode = mode
```

---

## Unified Reflexive Processor

Brings everything together for a single modality.

```python
class ReflexiveProcessor:
    """
    Complete reflexive processing pipeline for a modality.
    
    Combines:
    - SNN encoder (frozen)
    - Category classifiers (frozen)
    - Template matching (dynamic)
    - Session management
    - Memory correlation
    """
    
    def __init__(
        self,
        modality: str,
        encoder: SNNEncoder,
        category_bank: CategoryBank,
        template_store: TemplateStore,
        memory_bridge: ReflexiveMemoryBridge,
        heavy_identifier: Optional[Callable] = None,  # e.g., diart for audio
    ):
        self.modality = modality
        self.encoder = encoder
        self.category_bank = category_bank
        self.template_store = template_store
        self.memory_bridge = memory_bridge
        self.heavy_identifier = heavy_identifier
        
        self.session_manager: Optional[SessionManager] = None
    
    def process(
        self,
        raw_input: torch.Tensor,
        timestamp: float,
        latent_state: Optional[torch.Tensor] = None,
    ) -> ReflexiveOutput:
        """
        Process raw input through the reflexive pipeline.
        
        Returns annotations for M4 token construction.
        """
        # Step 1: Encode to spikes
        spike_pattern = self.encoder(raw_input)
        
        # Step 2: Category classification
        category_results = self.category_bank.classify(raw_input)
        
        # Step 3: Build category filter from results
        # (Only match templates consistent with detected categories)
        category_filter = self._build_category_filter(category_results)
        
        # Step 4: Template matching
        template_match = self.template_store.match(
            spike_pattern,
            category_filter=category_filter,
            include_session=True,
        )
        
        # Step 5: If no match and heavy identifier available, use it
        needs_heavy_identification = False
        if template_match is None and self.heavy_identifier:
            # Check if this is worth heavy processing
            # (e.g., only if is_speech and no speaker match)
            if self._should_use_heavy_identifier(category_results):
                needs_heavy_identification = True
        
        # Step 6: Record correlation with latent state
        if latent_state is not None:
            self.memory_bridge.record(
                template_match=template_match,
                category_results=category_results,
                latent_state=latent_state,
                timestamp=timestamp,
            )
        
        # Step 7: Update template if matched
        if template_match:
            self.template_store.reinforce_template(
                template_match.template_id,
                raw_input,
            )
        
        return ReflexiveOutput(
            modality=self.modality,
            timestamp=timestamp,
            spike_pattern=spike_pattern,
            categories=category_results,
            template_match=template_match,
            needs_heavy_identification=needs_heavy_identification,
        )
    
    def _build_category_filter(
        self,
        category_results: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Build template filter from category results."""
        filter = {}
        
        # Example: if voice_type detected, only match templates of that type
        if "voice_type" in category_results:
            result = category_results["voice_type"]
            if result["confidence"] > 0.8:
                voice_types = ["male", "female", "child"]
                filter["voice_type"] = voice_types[result["prediction"]]
        
        return filter
    
    def _should_use_heavy_identifier(
        self,
        category_results: Dict[str, Dict],
    ) -> bool:
        """Decide if we should invoke heavy identification."""
        # For audio: only use heavy identifier if speech detected
        if self.modality == "audio":
            is_speech = category_results.get("is_speech", {})
            return is_speech.get("prediction", 0) == 1 and is_speech.get("confidence", 0) > 0.7
        
        return False


@dataclass
class ReflexiveOutput:
    """Output from reflexive processing."""
    modality: str
    timestamp: float
    spike_pattern: SpikePattern
    categories: Dict[str, Dict]
    template_match: Optional[TemplateMatch]
    needs_heavy_identification: bool
    
    def to_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Convert to M4 token annotations format."""
        annotations = {}
        
        # Add category results
        for cat_name, result in self.categories.items():
            annotations[cat_name] = {
                "value": bool(result["prediction"]) if result["num_classes"] == 2 else result["prediction"],
                "confidence": float(result["confidence"]),
                "source": "reflexive_snn",
            }
        
        # Add template match
        if self.template_match:
            annotations["instance_id"] = {
                "value": self.template_match.template_id,
                "confidence": self.template_match.similarity,
                "source": self.template_match.source,
            }
        
        return annotations
```

---

## Modality-Specific Implementations

### Audio Reflexive System

```python
def build_audio_reflexive_system(
    pretrained_encoder_path: str,
    known_templates_path: Optional[str] = None,
    latent_dim: int = 512,
) -> ReflexiveProcessor:
    """
    Build complete audio reflexive system.
    """
    # Load pretrained SNN encoder
    encoder = AudioSNNEncoder(
        input_dim=128,  # e.g., mel spectrogram bins
        hidden_dim=256,
        spike_dim=128,
        num_steps=20,
    )
    encoder.load_state_dict(torch.load(pretrained_encoder_path))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Build category bank
    category_bank = build_audio_category_bank(encoder)
    
    # Initialize template store
    template_store = TemplateStore(encoder)
    if known_templates_path:
        template_store.load_known_templates(known_templates_path)
    
    # Memory bridge
    memory_bridge = ReflexiveMemoryBridge(latent_dim=latent_dim)
    
    # Heavy identifier (diart wrapper)
    def diart_identifier(audio_chunk):
        # Wrapper around diart for speaker diarization
        # Returns speaker labels
        pass
    
    return ReflexiveProcessor(
        modality="audio",
        encoder=encoder,
        category_bank=category_bank,
        template_store=template_store,
        memory_bridge=memory_bridge,
        heavy_identifier=diart_identifier,
    )


class AudioSNNEncoder(SNNEncoder):
    """Audio-specific SNN encoder with mel spectrogram preprocessing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Audio-specific: process temporal frames
        self.temporal_conv = nn.Conv1d(
            in_channels=kwargs.get("input_dim", 128),
            out_channels=kwargs.get("hidden_dim", 256),
            kernel_size=5,
            padding=2,
        )
    
    def preprocess(self, audio: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """Convert raw audio to mel spectrogram."""
        # Use torchaudio or librosa
        mel_spec = compute_mel_spectrogram(audio, sr)
        return mel_spec
```

### Vision Reflexive System

```python
def build_vision_reflexive_system(
    pretrained_encoder_path: str,
    known_templates_path: Optional[str] = None,
    latent_dim: int = 512,
) -> ReflexiveProcessor:
    """
    Build complete vision reflexive system.
    """
    encoder = VisionSNNEncoder(
        input_dim=256,  # e.g., patch embedding dim
        hidden_dim=384,
        spike_dim=128,
        num_steps=15,
    )
    encoder.load_state_dict(torch.load(pretrained_encoder_path))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Build category bank
    category_bank = CategoryBank(encoder)
    category_bank.add_classifier(CategoryClassifier(
        spike_dim=128, num_classes=2, category_name="is_human"
    ))
    category_bank.add_classifier(CategoryClassifier(
        spike_dim=128, num_classes=2, category_name="is_face"
    ))
    category_bank.add_classifier(CategoryClassifier(
        spike_dim=128, num_classes=2, category_name="is_motion"
    ))
    
    template_store = TemplateStore(encoder)
    if known_templates_path:
        template_store.load_known_templates(known_templates_path)
    
    memory_bridge = ReflexiveMemoryBridge(latent_dim=latent_dim)
    
    return ReflexiveProcessor(
        modality="vision",
        encoder=encoder,
        category_bank=category_bank,
        template_store=template_store,
        memory_bridge=memory_bridge,
        heavy_identifier=None,  # Could add face recognition here
    )
```

---

## Integration with Subconscious Layer

```python
class IntegratedSystem:
    """
    Full system integrating reflexive processors with subconscious layer.
    """
    
    def __init__(
        self,
        subconscious: SubconsciousLayer,
        reflexive_processors: Dict[str, ReflexiveProcessor],
        session_manager: SessionManager,
    ):
        self.subconscious = subconscious
        self.reflexive_processors = reflexive_processors
        self.session_manager = session_manager
        
        # Connect stores to session manager
        for modality, processor in reflexive_processors.items():
            session_manager.template_stores[modality] = processor.template_store
    
    def process_input(
        self,
        modality: str,
        raw_input: torch.Tensor,
        timestamp: float,
    ) -> Dict[str, Any]:
        """
        Process input through full pipeline.
        """
        # Get current latent state (for memory correlation)
        current_latent = self.subconscious.get_current_latent()
        
        # Reflexive processing
        processor = self.reflexive_processors.get(modality)
        if processor is None:
            raise ValueError(f"No processor for modality: {modality}")
        
        reflexive_output = processor.process(
            raw_input=raw_input,
            timestamp=timestamp,
            latent_state=current_latent,
        )
        
        # Build M4 token
        m4_token = self._build_m4_token(
            modality=modality,
            raw_input=raw_input,
            reflexive_output=reflexive_output,
            timestamp=timestamp,
        )
        
        # Step subconscious
        subconscious_output = self.subconscious.step(
            m4_token=m4_token,
            latent_history=self.subconscious.latent_history,
            pre_activation_history=self.subconscious.pre_activation_history,
            post_activation_history=self.subconscious.post_activation_history,
            current_time=timestamp,
        )
        
        # Handle heavy identification if needed
        if reflexive_output.needs_heavy_identification:
            self._trigger_heavy_identification(
                modality=modality,
                raw_input=raw_input,
                reflexive_output=reflexive_output,
            )
        
        return {
            "reflexive": reflexive_output,
            "subconscious": subconscious_output,
            "m4_token": m4_token,
        }
    
    def _build_m4_token(
        self,
        modality: str,
        raw_input: torch.Tensor,
        reflexive_output: ReflexiveOutput,
        timestamp: float,
    ) -> M4Token:
        """Build M4 token from reflexive output."""
        # Get embedding from hybrid encoder (existing system)
        embedding = self.subconscious.input_encoder.encode_raw(modality, raw_input)
        
        return M4Token(
            modality=modality,
            source=f"reflexive_{modality}",
            timestamp=timestamp,
            duration_ms=50.0,  # Configurable
            embedding=embedding,
            features={},  # Could extract from spike pattern
            annotations=reflexive_output.to_annotations(),
            raw_tokens=None,
        )
    
    def _trigger_heavy_identification(
        self,
        modality: str,
        raw_input: torch.Tensor,
        reflexive_output: ReflexiveOutput,
    ):
        """
        Trigger heavy identification in background.
        Results will register new session templates.
        """
        processor = self.reflexive_processors[modality]
        
        if processor.heavy_identifier is None:
            return
        
        # This would typically be async/background
        # For now, synchronous
        result = processor.heavy_identifier(raw_input)
        
        if result and modality == "audio":
            # Register speaker
            self.session_manager.register_speaker(
                speaker_id=result["speaker_id"],
                voice_type=reflexive_output.categories.get("voice_type", {}).get("prediction", "unknown"),
                samples=[raw_input],
                confirmed_by="diart",
            )
```

---

## Summary: The Pattern

1. **SNN Encoder** (frozen, pretrained)
   - Converts raw modality input to spike representation
   - Expensive to train, cheap to run
   - Captures temporal structure inherently

2. **Category Classifiers** (frozen, pretrained)
   - Fast binary/multiclass decisions
   - Answer stable, broad questions
   - Gate more expensive processing

3. **Template Store** (dynamic)
   - Known templates: persistent, high threshold
   - Session templates: ephemeral, adaptive
   - Few-shot creation from confirmed samples
   - Online reinforcement during session

4. **Session Manager** (contextual)
   - Tracks temporal/conversational context
   - Manages speaker/object identification
   - Coordinates promotion from session to known

5. **Memory Bridge** (correlation)
   - Links template activations to latent states
   - Enables context-aware queries
   - Supports memory resonance in subconscious

The pattern is **modality-agnostic**. Same architecture applies to audio, vision, text, or any other input type. Only the encoder architecture and category definitions change.

---

---

## Concept-Primed Recognition

### The Insight

The reflexive layer doesn't just passively recognize — it can be **primed** by context. When "car" comes up in conversation, the system retrieves learned weights that bias recognition toward car-like features. This is how expertise and context-sensitivity emerge.

The priming flows from the subconscious back down to the reflexive layer:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXECUTIVE                                      │
│                                                                             │
│   "There's a car outside. Todd mentioned he was expecting a delivery.      │
│    Should I alert him? He's in a meeting. I'll note it and wait."          │
│                                                                             │
│   Actions: update context, suppress alert, queue for later                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ latent state + sync + certainty
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SUBCONSCIOUS                                     │
│                                                                             │
│   Continuous latent state integrates:                                       │
│   - "car detected" (reflexive annotation)                                   │
│   - "outside" (spatial context from audio/visual)                           │
│   - "Todd expecting delivery" (memory resonance from earlier conversation)  │
│   - "Todd in meeting" (recent context)                                      │
│                                                                             │
│   Builds: situational gestalt, relevance weighting, uncertainty estimate    │
│                                                                             │
│   ──► Primes reflexive: "be sensitive to delivery-related stimuli"         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ M4 tokens with annotations
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                             REFLEXIVE                                       │
│                                                                             │
│   Fast, primed recognition:                                                 │
│   - SNN: "is_sound" ✓ "is_mechanical" ✓ "is_speech" ✗                      │
│   - Template match: "car" (primed from earlier conversation)                │
│   - Confidence: 0.87 (boosted because contextually expected)                │
│                                                                             │
│   Outputs: {"object": "car", "confidence": 0.87, "location": "outside"}    │
│                                                                             │
│   This layer doesn't know about deliveries or meetings.                     │
│   It just says "car" fast and moves on.                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ raw sensory input
```

**The reflexive layer's job is shortcuts.** It enables the subconscious to think about *meaning* rather than spending cycles on *identification*. The priming makes those shortcuts context-sensitive.

### Concept Correlation Weights

When the user labels something ("that's a car"), the system stores the spike pattern weights that characterize that concept. Later, those weights can prime recognition.

```python
@dataclass
class ConceptCorrelationWeights:
    """
    Stored weights that bias recognition toward a concept.
    Built from accumulated experience with that concept.
    """
    concept_id: str                    # "car", "todd", "music_jazz"
    
    # Per-modality attention biases
    modality_weights: Dict[str, torch.Tensor]  
    # {
    #   "vision": tensor[spike_dim],    # which visual features matter for cars
    #   "audio": tensor[spike_dim],     # what cars sound like
    # }
    
    # Category priors (what categories correlate with this concept)
    category_priors: Dict[str, Dict[str, float]]
    # {
    #   "vision": {"is_motion": 0.7, "is_outdoor": 0.8},
    #   "audio": {"is_mechanical": 0.9, "is_speech": 0.1},
    # }
    
    # Template associations (known instances of this concept)
    associated_templates: List[str]    # ["car_001", "car_002", "todds_car"]
    
    # Learning metadata
    example_count: int
    confidence: float
    last_updated: float


class ConceptWeightStore:
    """
    Stores and retrieves concept-specific correlation weights.
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.concepts: Dict[str, ConceptCorrelationWeights] = {}
        
        # Concept embeddings for semantic retrieval
        # (find related concepts, not just exact match)
        self.concept_embeddings: Dict[str, torch.Tensor] = {}
    
    def store_concept(
        self,
        concept_id: str,
        modality: str,
        spike_pattern: SpikePattern,
        category_results: Dict[str, float],
        semantic_embedding: Optional[torch.Tensor] = None,
    ):
        """
        Learn/update weights for a concept from an example.
        """
        if concept_id not in self.concepts:
            self.concepts[concept_id] = ConceptCorrelationWeights(
                concept_id=concept_id,
                modality_weights={},
                category_priors={},
                associated_templates=[],
                example_count=0,
                confidence=0.0,
                last_updated=time.time(),
            )
        
        concept = self.concepts[concept_id]
        concept.example_count += 1
        concept.last_updated = time.time()
        
        # Update modality weights (running average of spike patterns)
        pattern_vec = spike_pattern.to_template_vector()
        if modality not in concept.modality_weights:
            concept.modality_weights[modality] = pattern_vec.clone()
        else:
            alpha = 1.0 / concept.example_count
            concept.modality_weights[modality] = (
                (1 - alpha) * concept.modality_weights[modality] + 
                alpha * pattern_vec
            )
        
        # Update category priors
        if modality not in concept.category_priors:
            concept.category_priors[modality] = {}
        for cat_name, cat_value in category_results.items():
            if cat_name not in concept.category_priors[modality]:
                concept.category_priors[modality][cat_name] = cat_value
            else:
                alpha = 1.0 / concept.example_count
                concept.category_priors[modality][cat_name] = (
                    (1 - alpha) * concept.category_priors[modality][cat_name] +
                    alpha * cat_value
                )
        
        # Update confidence
        concept.confidence = min(0.95, 0.3 + concept.example_count * 0.05)
        
        # Store semantic embedding if provided
        if semantic_embedding is not None:
            self.concept_embeddings[concept_id] = semantic_embedding
    
    def get_priming_weights(
        self,
        concept_id: str,
        modality: str,
    ) -> Optional[PrimingWeights]:
        """
        Get weights to prime recognition for a concept.
        """
        concept = self.concepts.get(concept_id)
        if concept is None:
            return None
        
        modality_weight = concept.modality_weights.get(modality)
        category_prior = concept.category_priors.get(modality, {})
        
        if modality_weight is None:
            return None
        
        return PrimingWeights(
            concept_id=concept_id,
            attention_bias=modality_weight,
            category_priors=category_prior,
            confidence=concept.confidence,
        )
    
    def find_related_concepts(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find concepts semantically related to a query.
        Enables priming from context even without exact concept mention.
        """
        results = []
        for concept_id, embedding in self.concept_embeddings.items():
            similarity = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                embedding.unsqueeze(0)
            ).item()
            results.append((concept_id, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


@dataclass
class PrimingWeights:
    """Weights to prime recognition system for a concept."""
    concept_id: str
    attention_bias: torch.Tensor      # Bias toward these spike features
    category_priors: Dict[str, float] # Expected category values
    confidence: float
```

### Primed Reflexive Processor

```python
class PrimedReflexiveProcessor(ReflexiveProcessor):
    """
    Reflexive processor that can be primed by context from the subconscious.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concept_store = ConceptWeightStore()
        self.active_priming: List[PrimingWeights] = []
    
    def prime(self, concept_ids: List[str]):
        """
        Prime recognition for specific concepts.
        Called when subconscious/context suggests these concepts are relevant.
        """
        self.active_priming = []
        for concept_id in concept_ids:
            weights = self.concept_store.get_priming_weights(
                concept_id, 
                self.modality
            )
            if weights:
                self.active_priming.append(weights)
    
    def prime_from_latent(self, latent_state: torch.Tensor, top_k: int = 3):
        """
        Prime based on semantic similarity to current subconscious state.
        This is how context automatically influences perception.
        """
        related = self.concept_store.find_related_concepts(
            latent_state, 
            top_k=top_k
        )
        self.prime([concept_id for concept_id, _ in related])
    
    def clear_priming(self):
        """Clear all active priming."""
        self.active_priming = []
    
    def process(
        self,
        raw_input: torch.Tensor,
        timestamp: float,
        latent_state: Optional[torch.Tensor] = None,
    ) -> ReflexiveOutput:
        """
        Process with priming applied.
        """
        # Auto-prime from latent state if provided
        if latent_state is not None and not self.active_priming:
            self.prime_from_latent(latent_state, top_k=3)
        
        # Encode to spikes
        spike_pattern = self.encoder(raw_input)
        
        # Apply priming bias to spike pattern for matching
        primed_pattern = self._apply_priming(spike_pattern)
        
        # Category classification (with primed thresholds)
        category_results = self._primed_classify(raw_input, spike_pattern)
        
        # Template matching (with primed similarity boost)
        template_match = self._primed_match(primed_pattern, category_results)
        
        # Rest of processing...
        return ReflexiveOutput(
            modality=self.modality,
            timestamp=timestamp,
            spike_pattern=spike_pattern,
            categories=category_results,
            template_match=template_match,
            needs_heavy_identification=template_match is None,
        )
    
    def _apply_priming(self, spike_pattern: SpikePattern) -> SpikePattern:
        """
        Boost spike features that match primed concepts.
        """
        if not self.active_priming:
            return spike_pattern
        
        pattern_vec = spike_pattern.to_template_vector()
        
        # Weighted combination of priming biases
        total_bias = torch.zeros_like(pattern_vec)
        total_weight = 0.0
        
        for priming in self.active_priming:
            total_bias += priming.confidence * priming.attention_bias
            total_weight += priming.confidence
        
        if total_weight > 0:
            total_bias /= total_weight
            
            # Soft boost where pattern aligns with priming
            alignment = F.cosine_similarity(
                pattern_vec.unsqueeze(0),
                total_bias.unsqueeze(0)
            )
            
            # Apply boost to aligned dimensions (up to 30% boost)
            boosted_pattern = pattern_vec * (1 + 0.3 * F.normalize(total_bias, dim=0))
            
            return SpikePattern(
                spike_train=spike_pattern.spike_train,
                spike_count=boosted_pattern[:spike_pattern.spike_count.shape[0]],
                first_spike_time=spike_pattern.first_spike_time,
                spike_rate=spike_pattern.spike_rate,
            )
        
        return spike_pattern
    
    def _primed_classify(
        self,
        raw_input: torch.Tensor,
        spike_pattern: SpikePattern,
    ) -> Dict[str, Dict]:
        """
        Classify with primed category expectations.
        If priming expects "is_outdoor" and we detect outdoor, boost confidence.
        """
        base_results = self.category_bank.classify(raw_input)
        
        if not self.active_priming:
            return base_results
        
        for cat_name, result in base_results.items():
            for priming in self.active_priming:
                if cat_name in priming.category_priors:
                    expected = priming.category_priors[cat_name]
                    actual = result["probabilities"][1].item()
                    
                    # If prediction aligns with prior, boost confidence
                    if (expected > 0.5 and actual > 0.5) or (expected < 0.5 and actual < 0.5):
                        result["confidence"] *= (1 + 0.2 * priming.confidence)
        
        return base_results
    
    def _primed_match(
        self,
        spike_pattern: SpikePattern,
        category_results: Dict[str, Dict],
    ) -> Optional[TemplateMatch]:
        """
        Template matching with priming boost for associated templates.
        """
        # Get associated templates from active priming
        primed_templates = set()
        for priming in self.active_priming:
            concept = self.concept_store.concepts.get(priming.concept_id)
            if concept:
                primed_templates.update(concept.associated_templates)
        
        # Standard matching
        match = self.template_store.match(
            spike_pattern,
            category_filter=self._build_category_filter(category_results),
        )
        
        # Boost similarity for primed templates
        if match and match.template_id in primed_templates:
            match.similarity *= 1.15  # 15% boost for contextually expected
        
        return match
```

### Learning from User Feedback

```python
def learn_from_user_label(
    system: IntegratedSystem,
    concept_label: str,
    modality: str,
    raw_input: torch.Tensor,
    semantic_embedding: torch.Tensor,  # From language model
):
    """
    User provides a label for current input: "that's a car"
    System learns the association for future priming.
    """
    processor = system.reflexive_processors[modality]
    
    # Get current spike pattern
    spike_pattern = processor.encoder(raw_input)
    
    # Get current category results
    category_results = {
        k: v["probabilities"][1].item() 
        for k, v in processor.category_bank.classify(raw_input).items()
    }
    
    # Store concept weights
    processor.concept_store.store_concept(
        concept_id=concept_label,
        modality=modality,
        spike_pattern=spike_pattern,
        category_results=category_results,
        semantic_embedding=semantic_embedding,
    )
    
    # Optionally create a template for this specific instance
    template_id = f"{concept_label}_{int(time.time())}"
    processor.template_store.create_template(
        template_id=template_id,
        samples=[raw_input],
        category_constraints=category_results,
        persistent=False,
    )
    
    # Associate template with concept
    processor.concept_store.concepts[concept_label].associated_templates.append(template_id)
```

### The Learning Loop

```
User: "That looks like a car"
                ↓
System learns: car spike pattern, car category priors
                ↓
Later conversation: "I heard something outside"
                ↓
Subconscious: latent state shifts toward "outdoor", "vehicle" concepts
                ↓
Auto-prime: "car" weights retrieved, visual system primed
                ↓
Next visual frame: car-like features boosted, faster recognition
                ↓
System: "I see a car outside"
                ↓
User: [doesn't correct] → implicit reinforcement
                ↓
Car concept strengthens, associated templates reinforced
```

This is how expertise develops:
- A mechanic has heavily-weighted "car part" concepts
- A musician has heavily-weighted "chord voicing" patterns  
- You have heavily-weighted "Todd's voice" and "Lars's voice" patterns

The weights accumulate through experience and prime perception automatically based on context. The reflexive layer becomes increasingly efficient at recognizing things that matter to the user.

---

## Role in the Larger Architecture

The reflexive layer is **the shortcut system**. Its job is to answer "what is this?" as fast as possible so higher layers can focus on "what does this mean?" and "what should I do?"

**What the reflexive layer does:**
- Fast identification (SNN-based, sub-100ms)
- Context-primed recognition (sensitive to what's expected)
- Template matching (known voices, faces, objects)
- Session learning (new speakers/objects identified on the fly)
- Gating (should we even run heavy processing?)

**What the reflexive layer does NOT do:**
- Understand meaning or context
- Make decisions about actions
- Integrate information across modalities
- Maintain long-term state
- Reason about relationships

Those are subconscious and executive functions. The reflexive layer is the foundation that makes them efficient.

**The priming feedback loop:**
```
Subconscious latent state
        │
        ├──► encodes current context/focus
        │
        └──► primes reflexive layer
                    │
                    └──► reflexive outputs annotations
                                │
                                └──► subconscious integrates
                                            │
                                            └──► latent state updates
                                                        │
                                                        └──► priming adjusts...
```

This loop means perception is never purely bottom-up. What you're thinking about (subconscious) influences what you notice (reflexive), which influences what you think about, and so on.

---

## Next Steps

1. **Audio SNN encoder**: Train on your existing is_sound/is_speech data
2. **Speaker templates**: POC with Todd vs not-Todd
3. **Session management**: Integrate with diart for heavy identification
4. **Memory correlation**: Connect to subconscious latent states
5. **Concept priming**: Implement weight store and primed processor
6. **Vision extension**: Apply same pattern to webcam input
7. **Cross-modal priming**: "car" concept primes both audio (engine) and vision (shape)