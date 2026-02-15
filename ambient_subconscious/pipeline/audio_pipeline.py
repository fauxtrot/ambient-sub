"""Audio pipeline: ZMQ receiver → Diart diarization → Whisper transcription → SpacetimeDB.

Flow:
  AudioReceiver (ZMQ PULL :5555)
    → diart_queue (AudioChunk objects)
      → QueueAudioSource (unwraps .samples)
        → Diart StreamingInference (real-time diarization)
          → on utterance boundary: get_recent_audio → Whisper → CreateEntry
"""

import asyncio
import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from ..capture_receiver.audio_receiver import AudioReceiver
from ..capture_receiver.protocol import AudioChunk
from ..stream.queue_audio_source import QueueAudioSource
from ..spacetime.client import SpacetimeClient

logger = logging.getLogger(__name__)


class AudioPipeline:
    """Wires ZMQ audio into Diart + Whisper and publishes to SpacetimeDB."""

    def __init__(
        self,
        spacetime_client: SpacetimeClient,
        bind_address: str = "tcp://*:5555",
        whisper_model: str = "base",
        sample_rate: int = 16000,
        session_id: int = 1,
        device: str = "cuda",
    ):
        self.spacetime_client = spacetime_client
        self.bind_address = bind_address
        self.whisper_model_name = whisper_model
        self.sample_rate = sample_rate
        self.session_id = session_id
        self.device = device

        self.receiver = AudioReceiver(bind_address=bind_address)
        self._whisper_model = None
        self._running = False
        self._diart_thread: Optional[threading.Thread] = None
        self._transcription_thread: Optional[threading.Thread] = None

        # Queue for utterances waiting for Whisper transcription
        self._utterance_queue: queue.Queue = queue.Queue(maxsize=20)

    def start(self):
        """Start the full audio pipeline."""
        self._running = True

        # Start ZMQ receiver
        self.receiver.start()
        logger.info("AudioPipeline: ZMQ receiver started")

        # Start Diart inference thread
        self._diart_thread = threading.Thread(
            target=self._run_diart, daemon=True, name="audio-diart"
        )
        self._diart_thread.start()

        # Start Whisper transcription thread
        self._transcription_thread = threading.Thread(
            target=self._run_transcription, daemon=True, name="audio-whisper"
        )
        self._transcription_thread.start()

        logger.info("AudioPipeline: all threads started")

    def stop(self):
        """Stop the audio pipeline."""
        self._running = False
        self.receiver.stop()

        if self._diart_thread:
            self._diart_thread.join(timeout=5)
        if self._transcription_thread:
            self._utterance_queue.put(None)  # signal stop
            self._transcription_thread.join(timeout=5)

        logger.info("AudioPipeline stopped")

    def _run_diart(self):
        """Run Diart StreamingInference on audio from ZMQ receiver."""
        try:
            # PyTorch 2.7 defaults to weights_only=True which breaks
            # pyannote/diart model loading (omegaconf globals in checkpoints).
            # Patch torch.load before importing diart/pyannote.
            import torch
            _original_torch_load = torch.load
            def _patched_torch_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _original_torch_load(*args, **kwargs)
            torch.load = _patched_torch_load

            from diart import SpeakerDiarization
            from diart.inference import StreamingInference

            # Build pipeline
            pipeline = SpeakerDiarization()

            # QueueAudioSource reads AudioChunks from receiver's diart_queue
            source = QueueAudioSource(
                audio_queue=self.receiver.diart_queue,
                sample_rate=self.sample_rate,
            )

            inference = StreamingInference(
                pipeline, source, do_profile=False, do_plot=False
            )

            # Track active speakers for utterance boundary detection
            prev_speakers: set = set()
            silence_start: Optional[float] = None
            SILENCE_THRESHOLD_S = 1.0  # seconds of silence to trigger utterance end

            def on_annotation(annotation_wav):
                nonlocal prev_speakers, silence_start
                try:
                    annotation, waveform = annotation_wav
                    current_speakers = set()
                    for segment, _, label in annotation.itertracks(yield_label=True):
                        current_speakers.add(label)

                    now = time.time()

                    if not current_speakers:
                        # Silence detected
                        if silence_start is None:
                            silence_start = now
                        elif (now - silence_start) >= SILENCE_THRESHOLD_S and prev_speakers:
                            # Utterance boundary: silence after speech
                            duration = SILENCE_THRESHOLD_S + 1.0  # approximate
                            self._enqueue_utterance(duration, prev_speakers)
                            prev_speakers = set()
                            silence_start = None
                    else:
                        silence_start = None
                        prev_speakers = current_speakers

                except Exception as e:
                    logger.error(f"Diart annotation error: {e}", exc_info=True)

            inference.attach_hooks(on_annotation)
            inference()

        except Exception as e:
            logger.error(f"Diart inference failed: {e}", exc_info=True)

    def _enqueue_utterance(self, duration_s: float, speakers: set):
        """Grab recent audio from the rolling buffer and queue for Whisper."""
        audio = self.receiver.get_recent_audio(duration_s)
        if len(audio) < self.sample_rate * 0.3:  # skip <300ms
            return

        speaker_label = ", ".join(sorted(speakers)) if speakers else "Unknown"

        try:
            self._utterance_queue.put_nowait({
                "audio": audio,
                "duration_s": duration_s,
                "speaker": speaker_label,
                "timestamp_ms": int(time.time() * 1000),
            })
        except queue.Full:
            logger.warning("Utterance queue full, dropping utterance")

    def _run_transcription(self):
        """Consume utterances and transcribe with Whisper."""
        import whisper

        logger.info(f"Loading Whisper model '{self.whisper_model_name}'...")
        self._whisper_model = whisper.load_model(self.whisper_model_name, device=self.device)
        logger.info("Whisper model loaded")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        entry_seq = 0

        while self._running:
            try:
                item = self._utterance_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            try:
                audio = item["audio"]
                result = self._whisper_model.transcribe(
                    audio,
                    language="en",
                    fp16=(self.device != "cpu"),
                )

                text = result.get("text", "").strip()
                if not text:
                    continue

                entry_seq += 1
                entry_id = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{entry_seq:04d}"

                logger.info(
                    f"[Whisper] {item['speaker']}: {text} "
                    f"({item['duration_s']:.1f}s, conf={result.get('language_probability', 0):.2f})"
                )

                # Write to SpacetimeDB via reducer
                loop.run_until_complete(
                    self.spacetime_client.call_reducer(
                        "CreateEntry",
                        sessionId=self.session_id,
                        entryId=entry_id,
                        durationMs=int(item["duration_s"] * 1000),
                        transcript=text,
                        confidence=result.get("language_probability"),
                        audioClipPath=None,
                        recordingStartMs=0,
                        recordingEndMs=int(item["duration_s"] * 1000),
                    )
                )

            except Exception as e:
                logger.error(f"Whisper transcription error: {e}", exc_info=True)

        loop.close()
