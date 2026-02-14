Project: Ambient Subconscious - Multimodal Streaming Identity & Context System
Core Concept
A streaming audio-first architecture where continuous audio with real-time diarization forms the primary timebase. Visual captures and transcription anchor to audio events. A "subconscious" model generates fast predictions about who's speaking and what's happening, trained online via A/B model hot-swapping when corrections arrive.
Key Architecture Decisions
Streaming Foundation

Use diart for real-time speaker diarization (wraps pyannote, RxPY-based, 500ms-5s latency)
Audio is the continuous river; diarization markers and visual anchors float on it
Screenshots trigger on audio events (speaker change, speech start, etc.)

Encoding Stack

Audio: diart's speaker embeddings or wav2vec2
Visual: YOLOv8 feature extraction (pre-detection-head embeddings)
Text: Whisper STT (arrives async, becomes training signal)

Subconscious Model

Generates frame predictions continuously
Ephemeral prediction buffer with decay—hypotheses influence attention before validation
A/B hot-swap training: when user corrects a frame, active model goes to training, standby takes over
Standby model's "wrong" outputs become contrastive noise for training
Weight sync on quiescence or forced "sleep" consolidation when overwhelmed

Frame Structure
python@dataclass
class Frame:
    timestamp: int  # sample position in audio stream
    speaker_prediction: Optional[str]
    visual_context: Optional[dict]  # YOLO detections, embeddings
    text_hypothesis: Optional[str]
    confidence: float
    weight: float  # decays over time
```

### Proposed Directory Structure
```
ambient_subconscious/
├── stream/
│   ├── audio_source.py      # Wraps diart, emits diarization events
│   ├── visual_anchor.py     # Captures screenshots on triggers
│   └── stream_state.py      # Continuous audio + markers + anchors
├── encoders/
│   ├── audio_encoder.py
│   ├── visual_encoder.py    # YOLOv8
│   └── text_encoder.py      # Whisper
├── subconscious/
│   ├── frame.py             # Frame + prediction buffer
│   ├── model.py             # A/B swappable predictor
│   └── trainer.py           # Online correction + weight sync
└── main.py
Today's Goal
Set up python venv.
Get the streaming pipeline running: diart capturing and diarizing, visual anchors triggering, encoders producing embeddings. This creates the substrate for subconscious training experiments.
Future Integration
This will eventually connect to the existing SpacetimeDB ambient listener codebase for coordination and persistence.

Future: Client-Server Split
Design with eventual separation in mind. The stream/audio_source.py should abstract whether audio comes from local mic or network socket. Consider a simple protocol early (WebSocket with raw PCM chunks + timestamps). Server receives stream, runs diarization and encoding, returns frame events. This enables distributed recorders and centralized processing on PowerEdge with MI25.