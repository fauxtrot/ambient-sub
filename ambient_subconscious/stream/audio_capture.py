"""Real-time audio capture to WAV file"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
from pathlib import Path
from typing import Optional, Callable


class AudioCapture:
    """
    Captures audio from microphone and writes directly to WAV file.
    Runs in separate thread, non-blocking.
    """

    def __init__(
        self,
        output_path: Path,
        sample_rate: int = 16000,
        device: Optional[int] = None,
        on_chunk: Optional[Callable[[np.ndarray], None]] = None
    ):
        """
        Args:
            output_path: Path to output WAV file
            sample_rate: Audio sample rate
            device: Audio input device index (None = default)
            on_chunk: Optional callback for each audio chunk (for parallel processing)
        """
        self.output_path = Path(output_path)
        self.sample_rate = sample_rate
        self.device = device
        self.on_chunk = on_chunk

        self.audio_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.capture_thread = None
        self.writer_thread = None

    def start(self):
        """Start capturing audio"""
        print(f"Starting audio capture to {self.output_path}")

        # Ensure directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Start writer thread (writes queue to file)
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

        # Start capture thread (captures from mic to queue)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        """Stop capturing audio"""
        print("Stopping audio capture...")
        self.stop_flag.set()

        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.writer_thread:
            self.writer_thread.join(timeout=2.0)

        print(f"Audio saved to {self.output_path}")

    def _capture_loop(self):
        """Capture audio from microphone and put in queue"""
        def callback(indata, frames, time, status):
            if status:
                print(f"[Audio] Status: {status}")

            # Copy data (sounddevice reuses buffer)
            chunk = indata.copy()

            # Put in queue for writer
            self.audio_queue.put(chunk)

            # Call user callback if provided (for parallel processing)
            if self.on_chunk:
                try:
                    # Convert to mono if needed
                    if len(chunk.shape) > 1:
                        mono_chunk = chunk[:, 0]
                    else:
                        mono_chunk = chunk
                    self.on_chunk(mono_chunk)
                except Exception as e:
                    print(f"[Audio] Callback error: {e}")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                device=self.device,
                callback=callback
            ):
                print(f"[Audio] Capturing from device {self.device}")
                # Wait until stop flag is set
                while not self.stop_flag.is_set():
                    sd.sleep(100)
        except Exception as e:
            print(f"[Audio] Capture error: {e}")
        finally:
            # Signal writer to finish
            self.audio_queue.put(None)

    def _writer_loop(self):
        """Write audio from queue to WAV file"""
        try:
            with sf.SoundFile(
                str(self.output_path),
                mode='w',
                samplerate=self.sample_rate,
                channels=1,
                format='WAV'
            ) as wav_file:
                print(f"[Audio] Writing to {self.output_path}")

                while True:
                    chunk = self.audio_queue.get()

                    # None signals end of stream
                    if chunk is None:
                        break

                    # Write to file
                    wav_file.write(chunk)

        except Exception as e:
            print(f"[Audio] Writer error: {e}")
