"""
Stream State Manager - manages continuous audio stream with Frame storage.

This is the coordinator between:
- AudioListener (emits diarization events)
- Frame storage (JSONL or SpacetimeDB)
- Enrichment hooks (visual, text)
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
import uuid
import numpy as np
import soundfile as sf

from .frame import Frame
from .audio_source import AudioListener, DiarizationEvent
from .audio_capture import AudioCapture
from .queue_audio_source import QueueAudioSource
import threading
import time
import queue


@dataclass
class SessionMetadata:
    """Metadata for a recording session"""
    session_id: str
    device_name: str
    recording_mode: str  # "ambient", "meeting", "focus"
    started_at: str  # ISO timestamp
    ended_at: Optional[str] = None
    duration_seconds: float = 0.0
    frame_count: int = 0
    storage_path: str = ""
    audio_path: Optional[str] = None  # Path to raw audio recording (if enabled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "device_name": self.device_name,
            "recording_mode": self.recording_mode,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "storage_path": self.storage_path,
            "audio_path": self.audio_path,
        }

    def save(self, path: Path):
        """Save metadata to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SessionMetadata":
        """Load metadata from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class StreamState:
    """
    Manages continuous audio stream with Frame storage.

    Architecture:
    - Wraps AudioListener for diarization events
    - Stores Frames to JSONL (or SpacetimeDB in future)
    - Provides hooks for visual/text enrichment
    - Manages sessions and metadata
    """

    def __init__(
        self,
        storage_format: str = "jsonl",  # "jsonl" or "binary" or "spacetimedb"
        base_storage_path: Optional[str] = None,
        device_name: Optional[str] = None,
        recording_mode: str = "ambient",
        record_audio: bool = False,  # Whether to save raw audio
        device: Optional[str] = None,  # "cuda" or "cpu" for AudioListener
        audio_device: Optional[int] = None,  # Audio input device index
    ):
        """
        Initialize stream state manager.

        Args:
            storage_format: "jsonl" (recommended), "binary", or "spacetimedb"
            base_storage_path: Base directory for session storage
            device_name: Name of capture device (from .env)
            recording_mode: "ambient", "meeting", "focus"
            record_audio: Whether to save raw audio stream
            device: CUDA/CPU device for audio processing
            audio_device: Audio input device index (None = default microphone)
        """
        self.storage_format = storage_format
        self.base_storage_path = Path(base_storage_path or "data/sessions")
        self.device_name = device_name or os.getenv("DEVICE_NAME", "Unknown")
        self.recording_mode = recording_mode
        self.record_audio = record_audio

        # Create base storage directory
        self.base_storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize audio listener
        self.listener = AudioListener(device=device, audio_device=audio_device)

        # Session state
        self.current_session: Optional[SessionMetadata] = None
        self.session_start_time: Optional[float] = None
        self.frame_buffer: List[Frame] = []
        self.frame_file_handle: Optional[Any] = None

        # Audio recording state
        self.audio_chunks: List[np.ndarray] = []
        self.audio_file_path: Optional[Path] = None
        self.queue_audio_source: Optional[Any] = None  # For stopping diart

        # Enrichment hooks (user can register callbacks)
        self.on_frame_created: Optional[Callable[[Frame], None]] = None
        self.on_speaker_change: Optional[Callable[[Frame], None]] = None

        print(f"StreamState initialized (format={storage_format}, device={self.listener.device})")

    def start_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new recording session.

        Args:
            session_id: Optional session ID (auto-generated if None)
            metadata: Additional metadata to store

        Returns:
            Session ID
        """
        # Generate session ID (timestamp-based for chronological sorting)
        if session_id is None:
            timestamp = datetime.utcnow()
            session_id = timestamp.strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]

        # Create session directory
        session_path = self.base_storage_path / session_id
        session_path.mkdir(parents=True, exist_ok=True)

        # Create session metadata
        self.current_session = SessionMetadata(
            session_id=session_id,
            device_name=self.device_name,
            recording_mode=self.recording_mode,
            started_at=datetime.utcnow().isoformat(),
            storage_path=str(session_path),
        )

        # Set up storage based on format
        if self.storage_format == "jsonl":
            frames_path = session_path / "frames.jsonl"
            self.frame_file_handle = open(frames_path, 'w', encoding='utf-8')
        elif self.storage_format == "binary":
            # Binary format would use pickle or msgpack
            frames_path = session_path / "frames.msgpack"
            # Implementation here
            pass
        elif self.storage_format == "spacetimedb":
            # SpacetimeDB would use the client from ambient-listener
            pass

        # Set up audio recording if enabled
        if self.record_audio:
            audio_filename = f"audio.wav"
            self.audio_file_path = session_path / audio_filename
            self.current_session.audio_path = str(self.audio_file_path)
            self.audio_chunks = []  # Reset audio chunks
            print(f"  Audio recording: enabled -> {audio_filename}")

        # Save initial metadata
        self.current_session.save(session_path / "metadata.json")

        self.session_start_time = datetime.utcnow().timestamp()

        print(f"Started session: {session_id}")
        print(f"  Path: {session_path}")
        print(f"  Format: {self.storage_format}")

        return session_id

    def end_session(self):
        """End the current recording session"""
        if not self.current_session:
            return

        self.current_session.ended_at = datetime.utcnow().isoformat()

        if self.session_start_time:
            self.current_session.duration_seconds = (
                datetime.utcnow().timestamp() - self.session_start_time
            )

        # Close storage file
        if self.frame_file_handle:
            self.frame_file_handle.close()
            self.frame_file_handle = None

        # Audio is now saved in real-time by AudioCapture
        # No need to save chunks here - file is already written
        if self.record_audio and self.audio_file_path and self.audio_file_path.exists():
            import os
            audio_size = os.path.getsize(self.audio_file_path)
            # Estimate duration from file size (16-bit mono, 16kHz)
            estimated_duration = audio_size / (2 * 16000)  # bytes / (bytes_per_sample * sample_rate)
            print(f"  Audio saved: {self.audio_file_path.name} ({estimated_duration:.1f}s)")

        # Save final metadata
        session_path = Path(self.current_session.storage_path)
        self.current_session.save(session_path / "metadata.json")

        print(f"\nEnded session: {self.current_session.session_id}")
        print(f"  Duration: {self.current_session.duration_seconds:.1f}s")
        print(f"  Frames: {self.current_session.frame_count}")
        if self.current_session.audio_path:
            print(f"  Audio: {self.current_session.audio_path}")

        self.current_session = None
        self.session_start_time = None

    def store_frame(self, frame: Frame):
        """
        Store a frame to the current session.

        Args:
            frame: Frame to store
        """
        if not self.current_session:
            raise RuntimeError("No active session - call start_session() first")

        # Store based on format
        if self.storage_format == "jsonl":
            self.frame_file_handle.write(frame.to_json() + "\n")
            self.frame_file_handle.flush()  # Ensure immediate write
        elif self.storage_format == "binary":
            # Binary storage (msgpack or pickle)
            pass
        elif self.storage_format == "spacetimedb":
            # SpacetimeDB storage
            pass

        self.current_session.frame_count += 1

        # Call enrichment hook
        if self.on_frame_created:
            self.on_frame_created(frame)

    def _handle_audio_chunk(self, waveform: np.ndarray):
        """
        Internal handler for audio chunks (for raw audio recording).

        Args:
            waveform: Audio waveform chunk from diart
        """
        if self.record_audio and self.current_session:
            self.audio_chunks.append(waveform)

    def _handle_diarization_event(self, event: DiarizationEvent):
        """
        Internal handler for AudioListener events.

        Converts DiarizationEvent -> Frame -> Storage
        """
        if not self.current_session:
            return

        # Create frame from diarization event
        frame = Frame.from_diarization_event(
            event=event,
            session_id=self.current_session.session_id,
            stream_start_time=self.session_start_time or 0.0
        )

        # Store frame
        self.store_frame(frame)

        # Trigger speaker change hook
        if event.event_type == "speaker_change" and self.on_speaker_change:
            self.on_speaker_change(frame)

    def start_listening(self, duration: Optional[float] = None):
        """
        Start listening and storing frames with parallel audio capture and inference.

        Args:
            duration: Duration in seconds (None = infinite)
        """
        if not self.current_session:
            self.start_session()

        print(f"Starting continuous stream (session={self.current_session.session_id})...")

        # Set up audio file path if recording
        if self.record_audio:
            session_path = Path(self.current_session.storage_path)
            self.audio_file_path = session_path / "audio.wav"
            self.current_session.audio_path = str(self.audio_file_path)
            # Update metadata with audio path
            metadata_path = session_path / "metadata.json"
            self.current_session.save(metadata_path)
            print(f"  Audio recording: enabled -> {self.audio_file_path.name}")

        # Queue for passing audio from capture to diart
        audio_queue = queue.Queue(maxsize=100)

        # Start audio capture to file (if enabled)
        audio_capture = None
        if self.record_audio:
            def on_chunk_callback(chunk):
                """Pass audio chunks to diart via queue"""
                try:
                    audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass  # Skip if queue is full

            audio_capture = AudioCapture(
                output_path=self.audio_file_path,
                sample_rate=16000,
                device=self.listener.audio_device,
                on_chunk=on_chunk_callback
            )
            audio_capture.start()

            # Wait for audio capture to start producing data
            time.sleep(0.5)  # Give audio capture time to start

        # Run diart inference in parallel thread
        diart_thread = threading.Thread(
            target=self._run_diart_inference,
            args=(audio_queue, duration),
            daemon=True
        )
        diart_thread.start()

        # Wait for duration
        if duration is not None:
            print(f"Listening for {duration} seconds...")
            time.sleep(duration)

            # Stop audio capture
            if audio_capture:
                audio_capture.stop()

            # Signal diart to stop - clear queue first (it might be full)
            try:
                while True:
                    audio_queue.get_nowait()
            except queue.Empty:
                pass

            # Send stop signal
            audio_queue.put(None, timeout=1.0)

            # Close the queue audio source
            if self.queue_audio_source:
                self.queue_audio_source.close()

            # Wait for diart to finish
            diart_thread.join(timeout=2.0)
        else:
            # Infinite mode - wait for KeyboardInterrupt
            print("Listening indefinitely (Ctrl+C to stop)...")
            try:
                diart_thread.join()
            except KeyboardInterrupt:
                print("\nStopped by user")
                if audio_capture:
                    audio_capture.stop()

                # Clear queue and signal stop
                try:
                    while True:
                        audio_queue.get_nowait()
                except queue.Empty:
                    pass
                audio_queue.put(None, timeout=1.0)

                if self.queue_audio_source:
                    self.queue_audio_source.close()

    def _run_diart_inference(self, audio_queue: queue.Queue, duration: Optional[float]):
        """
        Run diart inference on audio from queue.
        Runs in separate thread.

        Audio flows: Mic → AudioCapture → Queue → QueueAudioSource → Diart
        """
        try:
            # Create queue-based audio source
            queue_source = QueueAudioSource(
                audio_queue=audio_queue,
                sample_rate=16000,
                block_size=512
            )

            # Store for access from main thread
            self.queue_audio_source = queue_source

            # Import diart components
            from diart.inference import StreamingInference

            # Create streaming inference with queue source
            inference = StreamingInference(
                self.listener.pipeline,
                queue_source,
                do_profile=False,
                do_plot=False
            )

            # Track events
            annotation_count = [0]

            def on_annotation(annotation_wav):
                """Process each annotation from diart"""
                try:
                    annotation_count[0] += 1
                    annotation, waveform = annotation_wav

                    # Convert annotation to events
                    for segment, _, label in annotation.itertracks(yield_label=True):
                        # Use segment end time as timestamp
                        timestamp = segment.end

                        # Create diarization event
                        from .audio_source import DiarizationEvent

                        event = DiarizationEvent(
                            timestamp=timestamp,
                            sample_position=int(timestamp * 16000),
                            speaker=label,
                            event_type="speaker_change",
                            confidence=1.0,
                            annotation=annotation
                        )

                        # Call event handler
                        self._handle_diarization_event(event)

                except Exception as e:
                    print(f"[Diart] Annotation error: {e}")
                    import traceback
                    traceback.print_exc()

            # Attach hook and run inference
            inference.attach_hooks(on_annotation)

            try:
                prediction = inference()
            except Exception as inf_error:
                print(f"[Diart] Inference error: {inf_error}")
                import traceback
                traceback.print_exc()
                raise

        except Exception as e:
            print(f"[Diart] Inference error: {e}")
            import traceback
            traceback.print_exc()

    def load_session_frames(self, session_id: str) -> List[Frame]:
        """
        Load all frames from a session.

        Args:
            session_id: Session ID to load

        Returns:
            List of Frames
        """
        session_path = self.base_storage_path / session_id

        if self.storage_format == "jsonl":
            frames_path = session_path / "frames.jsonl"
            frames = []
            with open(frames_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        frames.append(Frame.from_json(line))
            return frames

        # Other formats...
        return []

    def get_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Load session metadata"""
        metadata_path = self.base_storage_path / session_id / "metadata.json"
        if not metadata_path.exists():
            return None
        return SessionMetadata.load(metadata_path)

    def list_sessions(self) -> List[str]:
        """List all session IDs"""
        return [
            d.name for d in self.base_storage_path.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ]
