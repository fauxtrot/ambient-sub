"""Queue-based audio source for diart"""

import queue
import numpy as np
from typing import Optional
from diart.sources import AudioSource


class QueueAudioSource(AudioSource):
    """
    Audio source that reads from a queue instead of microphone.
    Allows one microphone capture to feed multiple consumers.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        sample_rate: int = 16000,
        block_size: int = 512,
    ):
        """
        Args:
            audio_queue: Queue to read audio chunks from
            sample_rate: Audio sample rate
            block_size: Number of samples per chunk expected by diart
        """
        uri = "queue://audio"
        super().__init__(uri, sample_rate)
        self.audio_queue = audio_queue
        self.block_size = block_size
        self._stopped = False

    def read(self):
        """
        Read audio chunks from queue and push them to the stream.

        This is called by diart's StreamingInference and must block until complete.
        Reads chunks from queue and pushes them via self.stream.on_next().
        """
        chunks_received = 0

        while not self._stopped:
            try:
                # Get chunk from queue (block with timeout)
                chunk = self.audio_queue.get(timeout=0.1)

                # None signals end of stream
                if chunk is None:
                    self._stopped = True
                    break

                # Unwrap AudioChunk from ZMQ receiver if needed
                if hasattr(chunk, 'samples'):
                    chunk = chunk.samples  # AudioChunk.samples is float32 ndarray

                # Ensure chunk is the right shape
                if isinstance(chunk, np.ndarray):
                    chunks_received += 1

                    # Flatten if multi-dimensional
                    if len(chunk.shape) > 1:
                        chunk = chunk.flatten()

                    # Convert to float32 and ensure it's a proper numpy array
                    chunk = np.array(chunk, dtype=np.float32)

                    # Reshape to (1, samples) - diart expects (channels, samples)
                    chunk = chunk.reshape(1, -1)

                    # Push chunk to the stream (RxPY observable)
                    self.stream.on_next(chunk)

            except queue.Empty:
                # No data available, continue waiting
                if self._stopped:
                    break
                continue
            except BaseException as e:
                print(f"[Audio] Stream error: {e}")
                import traceback
                traceback.print_exc()
                # Notify stream of error
                self.stream.on_error(e)
                break

        # Signal completion
        self.stream.on_completed()
        self.close()

    def close(self):
        """Stop reading from queue"""
        self._stopped = True
