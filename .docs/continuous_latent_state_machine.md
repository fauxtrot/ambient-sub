# Ambient Listener Architecture: Continuous Latent State Machine

## Overview

This document captures the architectural direction for the ambient listener's "subconscious" processing layer. After extensive discussion, we've moved away from traditional diffusion models toward a **continuous latent state machine** that better models how human perception and cognition actually work.

## Core Concept

The system is a continuously running state machine that:

1. **Never stops stepping** — inference isn't discrete runs, it's a perpetual process
2. **Accepts input as noise perturbation** — new sensory data perturbs the current latent state rather than starting fresh
3. **Converges on features at different rates** — simple features (is_sound) stabilize early, complex features (speaker_id, affect) take longer
4. **Drifts when input is sparse** — without grounding input, the system free-associates/hallucinates, similar to human mind-wandering or dreaming

## Why Not Traditional Diffusion

Traditional diffusion models:
- Run discrete inference passes (start from noise → denoise → output)
- Have fixed step counts
- Treat each generation as independent

Our requirements:
- Continuous processing with temporal continuity
- Variable convergence rates per feature
- Input as perturbation to ongoing state, not trigger for new inference
- Learned shortcuts for simple determinations

The **continuous latent state machine** uses diffusion-like denoising as its update rule, but the framing is fundamentally different. We're not generating samples — we're maintaining and updating a persistent representation of "now."

## Three-Layer Architecture

### Reflexive Layer
- **Fast, DSP-level processing**
- Handles: gain adjustment, clipping detection, silence detection, anomaly flagging
- Also serves as **temporal buffer manager** — tracks what frames the subconscious has processed, packages "what you missed" for handoff
- Can be pretrained/frozen models (YOLO, CLIP) for perception tasks
- One-shot identification lives here (object detection, face recognition)
- Outputs structured annotations that condition the subconscious layer

### Subconscious Layer (This Document's Focus)
- **Continuous latent state machine**
- Receives: encoded audio (encodec tokens), reflexive annotations, previous frame(s)
- Outputs: latent frame embedding that feature heads read from
- Maintains layered temporal context:
  - Perceptual now (high fidelity, milliseconds)
  - Working memory (seconds to minutes)
  - Background/ambient (persistent, low update rate)
- Memory retrieval is **passive resonance** — current frame embedding radiates into memory space, similar frames echo back and merge into noise basis

### Executive Layer
- **Decision making, attention, output**
- Reads from subconscious latent
- Makes slower, deliberate choices (expression changes, response planning, camera selection)
- Decides what to **attend to** from what surfaces, not what surfaces

## The Frame

The frame is a latent vector (suggest 512-1024 dimensions) that serves as shared representation. It is **not** a fixed schema — it's a rich embedding that supports arbitrary projection heads.

```python
frame_latent = tensor[latent_dim]  # the core representation

# Feature heads project from latent — add new ones as capabilities grow
heads = {
    "is_sound": nn.Linear(latent_dim, 1),        # converges early (~50-100ms)
    "is_speech": nn.Linear(latent_dim, 1),       # converges mid (~150-300ms)
    "speaker_id": nn.Linear(latent_dim, n),      # converges late (~300-500ms)
    "affect": nn.Linear(latent_dim, affect_dim), # converges late (~500ms-2s)
    # Future heads added without retraining core:
    # "whisper_trigger": nn.Linear(latent_dim, 1),
    # "tts_decision": nn.Linear(latent_dim, 1),
}
```

The latent IS the multimodal token (M4-style). All downstream systems read from this shared representation.

## Multi-Temporal Frame Merging

Previous frames contribute to the noise basis for the current step, creating implicit temporal context:

```python
def compute_noise_basis(latent_history):
    return (
        0.6 * latent_history[-1] +   # immediate
        0.3 * latent_history[-5] +   # recent context  
        0.1 * latent_history[-10]    # background context
    )
```

Weights may be fixed initially, learned later, or dynamic based on attention/input characteristics.

This creates natural forgetting curves and temporal binding without explicit memory management.

## Convergence as Signal

The rate at which features converge is itself meaningful:

- **Fast convergence** → high confidence, clear input
- **Slow convergence** → ambiguous, edge case
- **Oscillation** → genuinely uncertain, may need more context

Training should reward early convergence on simple features:

```python
loss = (
    early_weight * is_sound_loss(step_2_output) +
    mid_weight * is_speech_loss(step_4_output) +
    late_weight * speaker_id_loss(step_8_output)
)
```

Target convergence rates should approximate human timing:
- Sound detection: ~50-100ms
- Speech vs noise: ~150-300ms
- Speaker ID: ~300-500ms
- Emotional read: ~500ms-2s

## Memory System

Memory is not a query service. It's a **resonance system**.

Current frame embedding continuously radiates into embedding space. Similar stored frames echo back and merge into the noise basis. This is automatic and passive — relevant memories surface because they're geometrically close, not because they were requested.

This explains:
- **Morning recall**: sensory input triggers pattern matches to yesterday
- **Intrusive thoughts**: accidental embedding proximity surfaces unwanted memories
- **Context building**: recent similar frames naturally contribute more

### Cold Boot Sequence

```
1. Generate from pure noise → Frame 0 (neutral prior)
2. Go live with sensory input
3. Reflexive layer populates "now"
4. Subconscious frames start forming
5. Frame embeddings trigger memory retrieval organically
6. Context builds over seconds/minutes
```

No forced memory retrieval at boot. Let the system warm up naturally.

## Input Starvation / Drift

When input is sparse or absent:
- The denoising process continues on its own residue
- The system **drifts** — amplifying patterns, filling gaps with learned priors
- This is equivalent to mind-wandering, daydreaming, or sleep

Rich input = grounded perception  
No input = dreaming

Same process, different noise source.

## Training Approach

### Data Requirements

Labeled audio sequences with temporal annotations:

```python
training_sequence = {
    "audio_chunks": [chunk_0, chunk_1, ..., chunk_N],
    "timestamps": [t_0, t_1, ..., t_N],
    "labels_per_chunk": [
        # Early chunks: only early-converging features labeled
        {"is_sound": 1, "is_speech": 0, "speaker": None, "affect": None},
        # Later chunks: more features become valid targets
        {"is_sound": 1, "is_speech": 1, "speaker": "todd", "affect": [...]},
    ]
}
```

Include sequences with **gaps** (silence, no input) to train drift behavior.

### Training Loop (Conceptual)

```python
def train_on_sequence(model, sequence):
    latent = torch.zeros(latent_dim)  # cold start
    latent_history = [latent] * history_len
    total_loss = 0
    
    for i, (audio_chunk, labels) in enumerate(sequence):
        # Encode input as noise perturbation
        if audio_chunk is not None:
            input_noise = audio_encoder(audio_chunk)
        else:
            input_noise = torch.zeros(...)  # or learned idle noise
        
        # Compute noise basis from history
        noise_basis = compute_noise_basis(latent_history)
        
        # Step the model
        latent = model.step(noise_basis, input_noise, timestep=i)
        latent_history.append(latent)
        
        # Read features and compute loss for non-null labels
        predictions = model.read_features(latent)
        for feature, target in labels.items():
            if target is not None:
                total_loss += loss_fn(predictions[feature], target)
    
    total_loss.backward()
```

### Phased Capability Training

```
Phase 1: Train denoiser + is_sound head (establish latent structure)
Phase 2: Add is_speech head, fine-tune
Phase 3: Add speaker_id head, fine-tune
Phase 4: Add affect head, fine-tune
Phase N: Add new capability heads as needed
```

New heads can often be trained with frozen denoiser if latent is rich enough. Fine-tune denoiser only when latent needs to learn fundamentally new features.

## Integration with Reflexive Layer

Reflexive outputs become structured conditioning tokens:

```python
reflexive_annotation = {
    "source": "reflexive/vision/webcam",
    "type": "object_detected", 
    "payload": {"class": "human", "location": "at_computer"},
    "timestamp": ...
}
```

Subconscious learns correlations:
- `human_at_computer` + `voice_detected` → probably Todd
- `human_at_computer` + `silence` → Todd is reading
- `no_human` + `voice_detected` → anomaly

No explicit rules. Patterns emerge from data.

## A/B Hot-Swap Training

The system uses A/B model architecture:
- Model A runs inference
- Model B trains on recent data
- Swap when B surpasses A

Current phase: offline baseline training for both runners before live A/B cycling begins.

New capabilities require:
1. Offline enrichment of training data
2. Baseline training
3. Then live A/B refinement

## Next Steps

1. **Define latent dimensionality** — start with 512, evaluate
2. **Implement basic denoiser** — simple transformer to start
3. **Build audio encoder** — encodec or learned
4. **Create training data pipeline** — labeled sequences with temporal annotations
5. **Train Phase 1** — is_sound head, establish that the architecture works
6. **Iterate** — add heads, evaluate convergence rates, tune

## Open Questions

- Optimal latent dimension?
- Fixed vs learned vs dynamic temporal merge weights?
- How to handle very long-term context (hours, days) vs working memory (seconds, minutes)?
- Consistency loss during drift, or let it free-associate?
- How do reflexive annotations encode into the noise perturbation?

---

*This architecture aims for human-plausible timing and behavior — not just correct answers, but correct answers at the right time, with appropriate uncertainty, and natural drift when ungrounded.*