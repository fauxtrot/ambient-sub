# Agent Prompts — Capture Streaming (Phase 3)

Two parallel tasks: the Redot client sends raw audio + video frames, the MI25 host receives and routes them into the inference pipeline.

---

## Transport Decision: ZMQ PUSH/PULL

**Why ZMQ over WebSocket:**
- No HTTP overhead, no framing, no masking — raw bytes on the wire
- PUSH/PULL pattern gives automatic load balancing and backpressure
- Reconnects automatically if server restarts
- Multipart messages let us send a header + payload in one atomic operation
- pyzmq on the server side, godot_zeromq GDExtension (or raw TCP fallback) on the Redot side

**Socket topology:**
```
Redot Client (Windows Desktop)              MI25 Host (Linux Server)
┌─────────────────────────┐                ┌──────────────────────────────────┐
│                         │                │                                  │
│  Audio Capture          │   ZMQ PUSH     │   Audio Receiver (PULL :5555)    │
│  (mic @ 16kHz mono)     │ ──────────────>│     → routes to Diart / Whisper  │
│                         │                │                                  │
│  Video Capture          │   ZMQ PUSH     │   Video Receiver (PULL :5556)    │
│  (webcam @ 640x480 RGB) │ ──────────────>│     → routes to YOLO / Florence2 │
│                         │                │                                  │
│  Control Channel        │   ZMQ REQ/REP  │   Control (REP :5557)            │
│  (config, heartbeat)    │ <─────────────>│     → config negotiation         │
│                         │                │                                  │
└─────────────────────────┘                └──────────────────────────────────┘
```

**Message format (multipart ZMQ):**
```
Frame 0 (header):  JSON bytes — {"type": "audio"|"video", "ts": unix_ms, "seq": int, ...}
Frame 1 (payload): raw bytes — PCM int16 for audio, JPEG/raw RGB for video
```

This keeps the transport dumb and the parsing simple on both ends.

---

## AGENT PROMPT 1: Redot Client (Gaming Desktop)

### Context

You are working on a Redot Engine 26 (Godot 4.x fork) project called "Selestia" — an avatar/copilot client. The project already exists with expression systems, WebSocket control, and TTS. You are adding capture and streaming capabilities so the client can send raw microphone audio and webcam video to a remote server for AI processing.

The remote server is a Dell R630 running Ubuntu 22.04 with dual AMD MI25 GPUs. Its IP on the LAN will be configured via an export variable. The server will run ZMQ PULL sockets to receive data.

### Your Task

Add audio and video capture nodes to the Selestia project that stream raw data to the server via ZMQ PUSH sockets.

### Architecture Requirements

**Option A (preferred): godot_zeromq GDExtension**
- Repository: https://github.com/funatsufumiya/godot_zeromq
- Provides `ZMQSender` and `ZMQReceiver` nodes with `sendString()` and `sendBytes()` via `PackedByteArray`
- Supports PUSH/PULL, PUB/SUB, REQ/REP patterns
- Must be compiled as a GDExtension for Windows x86_64 (Redot 26 is Godot 4.x compatible)
- If compilation is problematic, fall back to Option B

**Option B (fallback): Raw TCP via StreamPeerTCP**
- Godot/Redot has built-in `StreamPeerTCP`
- Implement a simple length-prefixed protocol: [4-byte header len][JSON header][payload bytes]
- Less elegant but zero external dependencies
- Server would use asyncio TCP instead of ZMQ

### Audio Capture Implementation

Create an `AudioCaptureStreamer` node:

```
Requirements:
- Use AudioServer with an AudioEffectCapture on the mic bus
- Capture at 16kHz, mono, int16 PCM (this is what Whisper and Diart expect)
- If the mic bus provides a different sample rate, resample to 16kHz
- Buffer audio into chunks (configurable, default 100ms = 1600 samples)
- Each chunk is sent as a ZMQ multipart message:
    Frame 0: {"type": "audio", "ts": <unix_ms>, "seq": <counter>, "sr": 16000, "ch": 1, "fmt": "int16"}
    Frame 1: raw PCM bytes (3200 bytes per 100ms chunk at 16kHz mono int16)
- Include a configurable send rate / chunk size
- Expose an export variable for the server address: "tcp://<server_ip>:5555"
- Handle connection failures gracefully — log and retry, don't crash
- Include a visual indicator (e.g., a ColorRect) showing capture status: green=streaming, yellow=buffering, red=disconnected
```

### Video Capture Implementation

Create a `VideoCaptureStreamer` node:

```
Requirements:
- Use a CameraFeed or the system webcam via CameraServer
- Capture at a configurable resolution (default 640x480) and frame rate (default 5 FPS for ambient monitoring — NOT gaming framerate)
- JPEG-encode each frame before sending (raw RGB at 640x480 is ~900KB per frame; JPEG at quality 70 is ~30-50KB)
- Each frame is sent as a ZMQ multipart message:
    Frame 0: {"type": "video", "ts": <unix_ms>, "seq": <counter>, "w": 640, "h": 480, "fmt": "jpeg", "quality": 70}
    Frame 1: JPEG bytes
- Expose an export variable for the server address: "tcp://<server_ip>:5556"
- Frame rate should be configurable and independent of the game loop
- Consider using a Timer node to control capture rate rather than _process()
- Optional: include a screen capture mode (for Star Citizen overlay analysis later) using get_viewport().get_texture().get_image()
```

### Control Channel (lower priority)

Create a `CaptureControlChannel` node:

```
Requirements:
- ZMQ REQ/REP on tcp://<server_ip>:5557
- On startup, send a handshake: {"cmd": "hello", "client": "selestia", "capabilities": ["audio", "video"]}
- Server responds with config: {"audio_chunk_ms": 100, "video_fps": 5, "video_res": [640, 480], ...}
- Periodic heartbeat every 5 seconds
- Use this to negotiate capture parameters rather than hardcoding them
```

### Project Structure

```
selestia/
├── addons/
│   └── godot_zeromq/          # GDExtension (if using Option A)
├── capture/
│   ├── audio_capture_streamer.gd
│   ├── video_capture_streamer.gd
│   └── capture_control_channel.gd
├── scenes/
│   └── capture_test.tscn      # Test scene with all three nodes, status display
└── ...existing Selestia project files...
```

### Testing

Create a `capture_test.tscn` scene that:
1. Instantiates all three nodes
2. Shows a simple UI with: server IP input, connect/disconnect buttons, status indicators for audio/video/control, a frame counter and bytes-sent counter
3. Can be run standalone to test the capture pipeline before integrating into the main Selestia scene

### Important Notes

- Redot 26 is a Godot 4.4+ fork — use GDScript 4.x syntax
- The desktop is Windows (Ryzen 7 7800X3D, RTX 3060) — ensure Windows compatibility for any native extensions
- Do NOT add any ML/inference dependencies to this project — the desktop stays clean
- Audio quality matters more than video quality for this use case (speech is the primary modality)
- The webcam FPS is deliberately low (5 FPS) — this is ambient monitoring, not real-time video chat
- All server addresses should be configurable via export variables, not hardcoded

---

## AGENT PROMPT 2: MI25 Host Receiver (Server)

### Context

You are working on the `ambient-sub` project (https://github.com/fauxtrot/ambient-sub) — an ambient AI pipeline running on a Dell R630 server with dual AMD MI25 GPUs, Ubuntu 22.04, 16 cores, 96GB RAM. The project has a working Python venv with torch 2.7.1+rocm6.3, whisper, diart, pyannote, ultralytics (YOLO), and other dependencies.

The project already has:
- An agent infrastructure with base classes, event system, and coordinator
- A provider system with registry and router
- A frame structure (JSONL) with session management
- SpacetimeDB integration (HTTP client)
- Various test scripts for audio, YOLO, Whisper, diart

You are adding ZMQ receiver endpoints that accept raw audio and video data from a remote Redot client (Selestia) and route them into the existing inference pipeline.

### Your Task

Create a `capture_receiver` module that:
1. Listens for incoming audio and video data via ZMQ PULL sockets
2. Routes audio data to the Diart/Whisper pipeline
3. Routes video data to the YOLO pipeline
4. Integrates with the existing agent/provider infrastructure

### Architecture

```
ZMQ PULL :5555 (audio)  ──→  AudioReceiver  ──→  audio_queue  ──→  [Diart Thread] ──→ frames
                                                                ──→  [Whisper on utterance]
ZMQ PULL :5556 (video)  ──→  VideoReceiver  ──→  video_queue  ──→  [YOLO Thread]   ──→ frames
ZMQ REP  :5557 (control)──→  ControlHandler ──→  config/heartbeat
```

### Implementation

**Install pyzmq in the existing venv:**
```bash
pip install pyzmq
```

**Create `ambient_subconscious/capture_receiver/`:**

```
ambient_subconscious/capture_receiver/
├── __init__.py
├── receiver.py          # Main receiver orchestrator
├── audio_receiver.py    # ZMQ PULL for audio, feeds Diart/Whisper
├── video_receiver.py    # ZMQ PULL for video, feeds YOLO
├── control_handler.py   # ZMQ REP for config negotiation + heartbeat
└── protocol.py          # Message parsing (header JSON + payload bytes)
```

**`protocol.py` — Message format:**
```python
"""
ZMQ multipart message protocol:
  Frame 0: JSON header (utf-8 encoded)
  Frame 1: Raw payload bytes

Audio header:
  {"type": "audio", "ts": <unix_ms>, "seq": <int>, "sr": 16000, "ch": 1, "fmt": "int16"}
  Payload: raw PCM int16 bytes

Video header:
  {"type": "video", "ts": <unix_ms>, "seq": <int>, "w": 640, "h": 480, "fmt": "jpeg", "quality": 70}
  Payload: JPEG bytes
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioChunk:
    timestamp_ms: int
    sequence: int
    sample_rate: int
    channels: int
    samples: np.ndarray  # float32, shape (num_samples,)

@dataclass  
class VideoFrame:
    timestamp_ms: int
    sequence: int
    width: int
    height: int
    jpeg_bytes: bytes
    # decoded on demand by consumer

def parse_message(frames: list[bytes]) -> AudioChunk | VideoFrame:
    header = json.loads(frames[0].decode('utf-8'))
    payload = frames[1]
    
    if header["type"] == "audio":
        # Convert int16 PCM to float32 numpy array (what Whisper/Diart expect)
        raw = np.frombuffer(payload, dtype=np.int16)
        samples = raw.astype(np.float32) / 32768.0
        return AudioChunk(
            timestamp_ms=header["ts"],
            sequence=header["seq"],
            sample_rate=header.get("sr", 16000),
            channels=header.get("ch", 1),
            samples=samples
        )
    elif header["type"] == "video":
        return VideoFrame(
            timestamp_ms=header["ts"],
            sequence=header["seq"],
            width=header.get("w", 640),
            height=header.get("h", 480),
            jpeg_bytes=payload
        )
```

**`audio_receiver.py` — Core audio ingestion:**
```python
"""
Receives audio chunks via ZMQ PULL and feeds them into the processing pipeline.

The audio accumulates in a rolling buffer. Diart gets the continuous stream
for real-time diarization. When Diart signals an utterance boundary, the
relevant audio segment is queued for Whisper transcription.

This implements the "I hear → I understand" progressive enrichment model:
  1. Diart fires fast (~500ms latency) — reflexive layer
  2. Whisper fires on utterance completion — enrichment layer
"""

import zmq
import threading
import queue
import numpy as np
import logging
from .protocol import parse_message, AudioChunk

logger = logging.getLogger(__name__)

class AudioReceiver:
    def __init__(self, bind_address="tcp://*:5555", buffer_seconds=30):
        self.bind_address = bind_address
        self.buffer_seconds = buffer_seconds
        self.sample_rate = 16000
        
        # Rolling audio buffer (30s default)
        self.audio_buffer = np.zeros(self.sample_rate * buffer_seconds, dtype=np.float32)
        self.buffer_pos = 0
        
        # Queues for downstream consumers
        self.diart_queue = queue.Queue(maxsize=100)   # continuous stream for diarization
        self.whisper_queue = queue.Queue(maxsize=20)   # utterances for transcription
        
        # Stats
        self.chunks_received = 0
        self.bytes_received = 0
        self.last_chunk_ts = 0
        
        self._running = False
        self._thread = None
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info(f"AudioReceiver listening on {self.bind_address}")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _receive_loop(self):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PULL)
        socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout for clean shutdown
        socket.bind(self.bind_address)
        
        while self._running:
            try:
                frames = socket.recv_multipart()
                chunk = parse_message(frames)
                
                if not isinstance(chunk, AudioChunk):
                    logger.warning(f"Expected AudioChunk, got {type(chunk)}")
                    continue
                
                self._ingest_chunk(chunk)
                
            except zmq.Again:
                continue  # timeout, check _running flag
            except Exception as e:
                logger.error(f"AudioReceiver error: {e}")
        
        socket.close()
        ctx.term()
    
    def _ingest_chunk(self, chunk: AudioChunk):
        """Add chunk to rolling buffer and notify consumers."""
        n = len(chunk.samples)
        
        # Append to rolling buffer (circular)
        if self.buffer_pos + n <= len(self.audio_buffer):
            self.audio_buffer[self.buffer_pos:self.buffer_pos + n] = chunk.samples
        else:
            # Wrap around
            overflow = (self.buffer_pos + n) - len(self.audio_buffer)
            self.audio_buffer[self.buffer_pos:] = chunk.samples[:n - overflow]
            self.audio_buffer[:overflow] = chunk.samples[n - overflow:]
        
        self.buffer_pos = (self.buffer_pos + n) % len(self.audio_buffer)
        
        # Stats
        self.chunks_received += 1
        self.bytes_received += n * 2  # int16 = 2 bytes per sample
        self.last_chunk_ts = chunk.timestamp_ms
        
        # Feed to diart queue (drop if full — diart cares about recency, not completeness)
        try:
            self.diart_queue.put_nowait(chunk)
        except queue.Full:
            pass  # diart is behind, skip this chunk
        
        if self.chunks_received % 100 == 0:
            logger.debug(f"Audio: {self.chunks_received} chunks, {self.bytes_received / 1024:.0f}KB received")
```

**`video_receiver.py` — Core video ingestion:**
```python
"""
Receives video frames via ZMQ PULL and feeds them to YOLO.
Much simpler than audio — no rolling buffer needed, just decode and infer.
"""

import zmq
import threading
import queue
import logging
from .protocol import parse_message, VideoFrame

logger = logging.getLogger(__name__)

class VideoReceiver:
    def __init__(self, bind_address="tcp://*:5556"):
        self.bind_address = bind_address
        self.frame_queue = queue.Queue(maxsize=10)  # YOLO consumer
        
        self.frames_received = 0
        self.bytes_received = 0
        
        self._running = False
        self._thread = None
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        logger.info(f"VideoReceiver listening on {self.bind_address}")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _receive_loop(self):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PULL)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.bind(self.bind_address)
        
        while self._running:
            try:
                frames = socket.recv_multipart()
                vframe = parse_message(frames)
                
                if not isinstance(vframe, VideoFrame):
                    logger.warning(f"Expected VideoFrame, got {type(vframe)}")
                    continue
                
                self.frames_received += 1
                self.bytes_received += len(vframe.jpeg_bytes)
                
                # Feed to YOLO queue (drop old frames if YOLO is behind)
                try:
                    self.frame_queue.put_nowait(vframe)
                except queue.Full:
                    # Drop oldest, add newest
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(vframe)
                
                if self.frames_received % 50 == 0:
                    logger.debug(f"Video: {self.frames_received} frames, {self.bytes_received / 1024:.0f}KB received")
                    
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"VideoReceiver error: {e}")
        
        socket.close()
        ctx.term()
```

**`receiver.py` — Orchestrator:**
```python
"""
Main entry point for the capture receiver.
Starts audio, video, and control receivers.
Can be run standalone for testing or imported by the agent coordinator.
"""

import logging
import signal
import time
from .audio_receiver import AudioReceiver
from .video_receiver import VideoReceiver

logger = logging.getLogger(__name__)

class CaptureReceiver:
    def __init__(self, audio_port=5555, video_port=5556, control_port=5557, bind_ip="*"):
        self.audio = AudioReceiver(f"tcp://{bind_ip}:{audio_port}")
        self.video = VideoReceiver(f"tcp://{bind_ip}:{video_port}")
        # TODO: ControlHandler on control_port
    
    def start(self):
        logger.info("Starting capture receivers...")
        self.audio.start()
        self.video.start()
        logger.info("All receivers started.")
    
    def stop(self):
        logger.info("Stopping capture receivers...")
        self.audio.stop()
        self.video.stop()
        logger.info("All receivers stopped.")
    
    def status(self) -> dict:
        return {
            "audio": {
                "chunks_received": self.audio.chunks_received,
                "bytes_received": self.audio.bytes_received,
                "last_ts": self.audio.last_chunk_ts,
                "diart_queue_size": self.audio.diart_queue.qsize(),
                "whisper_queue_size": self.audio.whisper_queue.qsize(),
            },
            "video": {
                "frames_received": self.video.frames_received,
                "bytes_received": self.video.bytes_received,
                "yolo_queue_size": self.video.frame_queue.qsize(),
            }
        }


def main():
    """Standalone test — run the receiver and print stats."""
    logging.basicConfig(level=logging.DEBUG)
    
    receiver = CaptureReceiver()
    receiver.start()
    
    def shutdown(sig, frame):
        receiver.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    print("\nCapture receiver running. Waiting for connections...")
    print("Press Ctrl+C to stop.\n")
    
    while True:
        time.sleep(5)
        s = receiver.status()
        print(f"  Audio: {s['audio']['chunks_received']} chunks, "
              f"{s['audio']['bytes_received']/1024:.0f}KB | "
              f"Video: {s['video']['frames_received']} frames, "
              f"{s['video']['bytes_received']/1024:.0f}KB")


if __name__ == "__main__":
    main()
```

### Integration with Existing Pipeline

The receiver creates queues that existing components consume:

```
receiver.audio.diart_queue   → Feed into Diart streaming diarization (reflexive layer)
receiver.audio.whisper_queue → Feed into Whisper on utterance boundaries (enrichment)
receiver.video.frame_queue   → Feed into YOLO inference (visual modality)
```

The agent coordinator should start the CaptureReceiver as part of its initialization and wire the queues to the appropriate provider agents.

### Testing

**Create `tests/test_capture_receiver.py`:**
- Start the receiver
- Create a ZMQ PUSH test client that sends synthetic audio (sine wave) and a test JPEG
- Verify chunks arrive in the queues with correct parsing
- Verify the rolling buffer accumulates correctly
- Measure throughput: how many chunks/sec at 100ms audio + 5fps video

**Create `tests/test_zmq_sender.py`:**
- A standalone Python script that simulates what the Redot client will send
- Captures audio from the local mic (using sounddevice) and sends via ZMQ PUSH
- Captures a test image and sends as JPEG
- Useful for testing the server pipeline before the Redot client is ready

### Dependencies

```bash
pip install pyzmq
# Already installed: numpy, opencv-python-headless
```

### Important Notes

- The MI25 host is headless Linux — no display, no mic. All capture comes from the remote client.
- Audio MUST arrive as 16kHz mono int16 PCM — this is non-negotiable for Whisper/Diart compatibility
- Video is JPEG-encoded before transmission to reduce bandwidth. Decode on the server with cv2.imdecode()
- The audio rolling buffer enables Diart to maintain temporal context even if individual chunks are dropped
- Queue overflow strategy: audio drops (diart cares about recency), video replaces oldest (YOLO cares about recency)
- Bind on `0.0.0.0` (or `*` in ZMQ) — the Proxmox VM firewall needs ports 5555-5557 opened
- The server IP that the Redot client connects to is the VM's LAN IP, not the Proxmox host IP

---

## Coordination Notes

Both agents should produce independently testable components:

1. **MI25 host first:** Stand up the receiver, verify with the Python test sender (`test_zmq_sender.py`). This unblocks pipeline testing without waiting for the Redot client.

2. **Redot client second:** Build the capture nodes, test against the running server receiver. The `capture_test.tscn` scene should show connection status and bytes flowing.

3. **Integration:** Once both sides are sending/receiving, wire the receiver queues into the existing Diart/Whisper/YOLO providers and watch frames start generating.

The Python test sender is the critical bridge — it lets you develop and test the entire server pipeline using the desktop's mic/webcam through Python, without any Redot dependency. The Redot client replaces it later.