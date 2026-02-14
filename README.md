# Ambient Subconscious

**Agent Swarm System with SpacetimeDB Backbone**

An interactive multimodal streaming system that:
- Captures and processes audio, video, and screen context in real-time
- Coordinates multiple AI agents through SpacetimeDB
- Maintains continuous latent context (CLCT heartbeat)
- Drives interactive 3D character behavior
- Learns from user corrections through feedback loops

## Architecture

Built around **agent swarm coordination** with SpacetimeDB as the central nervous system:

```
Input Agents → SpacetimeDB → Enrichment → Subconscious → Executive → Outputs
    ↓            ↓              ↓             ↓             ↓          ↓
 Whisper    Real-time       Sentiment    CLCT Heartbeat  Decisions  3D Char
 Diart      Pub/Sub         Intent       Working Memory  Learning   Web UI
 YOLO       Storage         Context      SSN Models      Behavior   Discord
```

### Core Concept

Audio flows continuously through the listener, providing the timebase for all events. Visual captures, transcription, and predictions all anchor to audio events. The subconscious model maintains working memory and learns online from user corrections.

**Key Innovation:** Agent swarm with meta-awareness - the system knows when you're editing its predictions ("why are you editing my brain?") and uses that interaction to improve.

See [.docs/project-direction.md](.docs/project-direction.md) for original design and [.claude/plans/snuggly-herding-frost.md](.claude/plans/snuggly-herding-frost.md) for agent swarm architecture.

## Setup

Python 3.13.5 with venv:

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (already done)
pip install -r requirements.txt
```

## Quick Start

### Agent Swarm System (New!)

Start the agent coordinator (foundation ready, awaiting agent implementations):

```powershell
# Start agent swarm
python -m ambient_subconscious start

# Start specific agents (when implemented)
python -m ambient_subconscious start --agents audio,webcam

# Check status
python -m ambient_subconscious status

# Use custom config
python -m ambient_subconscious start --config my_config.yaml
```

Configure agents in `ambient_subconscious/config.yaml`:
```yaml
agents:
  audio:
    enabled: true
    providers: [whisper, diart]
  webcam:
    enabled: false  # Enable when implemented
```

### Original Listener System

Record a 30-second stream with speaker diarization:

```powershell
python examples\stream_recording.py
```

This will create a session in `data/sessions/` with:
- `frames.jsonl` - Frame-by-frame audio stream data
- `metadata.json` - Session information

## Testing Components

Run individual component tests:

```powershell
# Test the core listener (10s interactive)
python tests\test_listener.py

# Test CUDA/GPU
python tests\test_cuda.py

# Test audio capture
python tests\test_audio_capture.py

# Test screenshot capture
python tests\test_screenshot.py

# Test YOLOv8
python tests\test_yolo.py

# Test Whisper
python tests\test_whisper.py

# Test diart directly (interactive, 10s)
python tests\test_diart.py
```

Or run all automated tests:

```powershell
python tests\run_all_tests.py
```

## Project Structure

```
ambient_subconscious/
├── stream/           # Audio source and visual anchors (core listener)
├── encoders/         # Audio, visual, text encoding
└── subconscious/     # Prediction models and online training

tests/                # Component tests
.models/              # Downloaded ML models (gitignored)
data/                 # Runtime data, frames, embeddings (gitignored)
recordings/           # Audio recordings (gitignored)
```

## Directory Conventions

- `.models/` - All downloaded models (YOLO, Whisper, pyannote, etc.)
- `data/` - Runtime data, captured frames, embeddings
- `recordings/` - Audio recordings for testing/replay

## Current Status

### ✅ Phase 1 Complete: Agent Infrastructure Foundation

**Agent System:**
- [x] Base agent classes (Agent, ProviderAgent, EnrichmentAgent, TrainingAgent)
- [x] Event system with typed events (EntryCreated, FrameCreated, Correction, etc.)
- [x] Agent coordinator for lifecycle management
- [x] Health monitoring and graceful shutdown

**SpacetimeDB Integration:**
- [x] Python client wrapper for HTTP API
- [x] Reducer invocation helpers
- [x] Query interface
- [x] Convenience functions

**Configuration & CLI:**
- [x] YAML-based configuration system
- [x] Command-line interface (start/stop/status)
- [x] Agent enable/disable flags
- [x] Logging configuration

**Original Components:**
- [x] Environment setup (Python 3.13.5, RTX 3060)
- [x] Core dependencies installed
- [x] Test scripts created
- [x] **Core listener implemented** ([audio_source.py](ambient_subconscious/stream/audio_source.py))
- [x] **Frame structure and storage** ([frame.py](ambient_subconscious/stream/frame.py), [stream_state.py](ambient_subconscious/stream/stream_state.py))
  - JSONL storage format
  - Session management with metadata
  - Enrichment hooks for visual/text
  - Configuration via .env
- [x] **Provider system** (M4 tokens, registry, router, Diart adapter)
- [x] **Training infrastructure** (SSN models, hybrid classifier, data pipeline)

### 🚧 Next: Phase 2-6

**Phase 2: Vision Pipeline (Week 2-3)**
- [ ] YOLO Provider Adapter
- [ ] Frame table in SpacetimeDB
- [ ] Webcam Agent
- [ ] Screen Capture Agent
- [ ] Vision Correction UI

**Phase 3: CLCT Heartbeat (Week 3-4)**
- [ ] Subconscious Layer Service
- [ ] Working Memory Buffer
- [ ] Latent Representation
- [ ] CLCT Heartbeat Loop

**Phase 4: Interactive Interface (Week 4-5)**
- [ ] Executive Layer Service
- [ ] Character Behavior Protocol
- [ ] Godot Integration Bridge
- [ ] User Interaction Awareness
- [ ] Enhanced Web UI

**Phase 5: Training Loop (Week 5-6)**
- [ ] Correction Collector Agent
- [ ] Training Data Export
- [ ] Model Retraining Trigger
- [ ] A/B Testing Infrastructure

**Phase 6: Integration & Polish (Week 6-7)**
- [ ] Monitoring Dashboard
- [ ] Documentation
- [ ] End-to-end testing

**Note:** There is a known compatibility issue with `torchaudio.AudioMetaData` and `pyannote.audio` that needs to be resolved.

## Hardware

- RTX 3060 (local)
- Dell R630 with MI25 (pending hardware)
