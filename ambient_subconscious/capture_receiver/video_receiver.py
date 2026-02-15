"""Receives video frames via ZMQ PULL and feeds them to YOLO.

Simpler than audio -- no rolling buffer needed. Decode JPEG and infer.
"""

import logging
import queue
import threading

import zmq

from .protocol import VideoFrame, parse_message

logger = logging.getLogger(__name__)


class VideoReceiver:
    """ZMQ PULL receiver for JPEG video frames from the Redot client."""

    def __init__(self, bind_address: str = "tcp://*:5556"):
        self.bind_address = bind_address
        self.frame_queue: queue.Queue[VideoFrame] = queue.Queue(maxsize=10)

        # Stats
        self.frames_received = 0
        self.bytes_received = 0
        self.last_frame_ts = 0

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True, name="video-recv")
        self._thread.start()
        logger.info(f"VideoReceiver listening on {self.bind_address}")

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
                    vframe = parse_message(raw)

                    if not isinstance(vframe, VideoFrame):
                        logger.warning(f"Expected VideoFrame, got {type(vframe).__name__}")
                        continue

                    self.frames_received += 1
                    self.bytes_received += len(vframe.jpeg_bytes)
                    self.last_frame_ts = vframe.timestamp_ms

                    # Feed to YOLO queue -- drop oldest if YOLO is behind
                    try:
                        self.frame_queue.put_nowait(vframe)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.frame_queue.put_nowait(vframe)

                    if self.frames_received % 50 == 0:
                        logger.debug(
                            f"Video: {self.frames_received} frames, "
                            f"{self.bytes_received / 1024:.0f} KB received"
                        )

                except zmq.Again:
                    continue
                except Exception as e:
                    logger.error(f"VideoReceiver error: {e}", exc_info=True)
        finally:
            sock.close()
            ctx.term()
            logger.info("VideoReceiver stopped.")
