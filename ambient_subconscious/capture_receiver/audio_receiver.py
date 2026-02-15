"""Receives audio chunks via ZMQ PULL and feeds them into the processing pipeline.

The audio accumulates in a rolling buffer. Diart gets the continuous stream
for real-time diarization. When Diart signals an utterance boundary, the
relevant audio segment is queued for Whisper transcription.

This implements the "I hear -> I understand" progressive enrichment model:
  1. Diart fires fast (~500ms latency) -- reflexive layer
  2. Whisper fires on utterance completion -- enrichment layer
"""

import logging
import queue
import threading

import numpy as np
import zmq

from .protocol import AudioChunk, parse_message

logger = logging.getLogger(__name__)


class AudioReceiver:
    """ZMQ PULL receiver for raw PCM audio from the Redot client."""

    def __init__(self, bind_address: str = "tcp://*:5555", buffer_seconds: int = 30):
        self.bind_address = bind_address
        self.buffer_seconds = buffer_seconds
        self.sample_rate = 16000

        # Rolling audio buffer (circular, 30s default)
        self.audio_buffer = np.zeros(self.sample_rate * buffer_seconds, dtype=np.float32)
        self.buffer_pos = 0
        self.total_samples_written = 0  # monotonic counter for buffer indexing

        # Queues for downstream consumers
        self.diart_queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=100)
        self.whisper_queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=20)

        # Stats
        self.chunks_received = 0
        self.bytes_received = 0
        self.last_chunk_ts = 0

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True, name="audio-recv")
        self._thread.start()
        logger.info(f"AudioReceiver listening on {self.bind_address}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _receive_loop(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout for clean shutdown
        sock.bind(self.bind_address)

        try:
            while self._running:
                try:
                    raw = sock.recv()
                    chunk = parse_message(raw)

                    if not isinstance(chunk, AudioChunk):
                        logger.warning(f"Expected AudioChunk, got {type(chunk).__name__}")
                        continue

                    self._ingest_chunk(chunk)

                except zmq.Again:
                    continue  # recv timeout, loop back to check _running
                except Exception as e:
                    logger.error(f"AudioReceiver error: {e}", exc_info=True)
        finally:
            sock.close()
            ctx.term()
            logger.info("AudioReceiver stopped.")

    def _ingest_chunk(self, chunk: AudioChunk):
        """Add chunk to rolling buffer and notify consumers."""
        n = len(chunk.samples)
        buf_len = len(self.audio_buffer)

        # Write into circular buffer
        end = self.buffer_pos + n
        if end <= buf_len:
            self.audio_buffer[self.buffer_pos:end] = chunk.samples
        else:
            # Wrap around
            first = buf_len - self.buffer_pos
            self.audio_buffer[self.buffer_pos:] = chunk.samples[:first]
            self.audio_buffer[:n - first] = chunk.samples[first:]

        self.buffer_pos = (self.buffer_pos + n) % buf_len
        self.total_samples_written += n

        # Stats
        self.chunks_received += 1
        self.bytes_received += n * 2  # int16 on the wire = 2 bytes/sample
        self.last_chunk_ts = chunk.timestamp_ms

        # Feed to diart queue (drop if full -- diart cares about recency)
        try:
            self.diart_queue.put_nowait(chunk)
        except queue.Full:
            pass

        if self.chunks_received % 100 == 0:
            logger.debug(
                f"Audio: {self.chunks_received} chunks, "
                f"{self.bytes_received / 1024:.0f} KB received"
            )

    def get_recent_audio(self, seconds: float) -> np.ndarray:
        """Return the most recent N seconds from the rolling buffer."""
        n_samples = min(int(seconds * self.sample_rate), len(self.audio_buffer))
        n_samples = min(n_samples, self.total_samples_written)
        if n_samples == 0:
            return np.zeros(0, dtype=np.float32)

        start = (self.buffer_pos - n_samples) % len(self.audio_buffer)
        if start < self.buffer_pos:
            return self.audio_buffer[start:self.buffer_pos].copy()
        else:
            return np.concatenate([
                self.audio_buffer[start:],
                self.audio_buffer[:self.buffer_pos],
            ])
