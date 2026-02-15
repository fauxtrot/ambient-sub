"""Video pipeline: ZMQ receiver → JPEG decode → YOLO → SpacetimeDB.

Flow:
  VideoReceiver (ZMQ PULL :5556)
    → frame_queue (VideoFrame objects)
      → cv2.imdecode (JPEG → BGR ndarray)
        → YOLO inference → detection list
          → SpacetimeDB CreateFrame via Svelte API
"""

import asyncio
import json
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from ..capture_receiver.video_receiver import VideoReceiver
from ..capture_receiver.protocol import VideoFrame
from ..spacetime.client import SpacetimeClient

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Drains ZMQ video frames, runs YOLO, and publishes to SpacetimeDB."""

    def __init__(
        self,
        spacetime_client: SpacetimeClient,
        bind_address: str = "tcp://*:5556",
        yolo_model: str = "yolo11n",
        yolo_confidence: float = 0.5,
        session_id: int = 1,
        device: str = "cuda",
    ):
        self.spacetime_client = spacetime_client
        self.bind_address = bind_address
        self.yolo_model_name = yolo_model
        self.yolo_confidence = yolo_confidence
        self.session_id = session_id
        self.device = device

        self.receiver = VideoReceiver(bind_address=bind_address)
        self._yolo = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.frames_processed = 0
        self.total_detections = 0

    def start(self):
        """Start the video pipeline."""
        self._running = True
        self.receiver.start()

        self._thread = threading.Thread(
            target=self._run_inference, daemon=True, name="video-yolo"
        )
        self._thread.start()
        logger.info("VideoPipeline started")

    def stop(self):
        """Stop the video pipeline."""
        self._running = False
        self.receiver.stop()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(
            f"VideoPipeline stopped: {self.frames_processed} frames, "
            f"{self.total_detections} detections"
        )

    def _load_yolo(self):
        """Lazy-load YOLO model."""
        if self._yolo is not None:
            return

        from ultralytics import YOLO

        logger.info(f"Loading YOLO model '{self.yolo_model_name}'...")
        self._yolo = YOLO(self.yolo_model_name)
        logger.info("YOLO model loaded")

    def _run_inference(self):
        """Consumer loop: pop frames, decode, run YOLO, publish."""
        self._load_yolo()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self._running:
            try:
                vframe: VideoFrame = self.receiver.frame_queue.get(timeout=1.0)
            except Exception:
                continue

            try:
                # Decode JPEG
                buf = np.frombuffer(vframe.jpeg_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    logger.warning("Failed to decode JPEG frame")
                    continue

                # Run YOLO
                results = self._yolo(
                    img_bgr,
                    conf=self.yolo_confidence,
                    device=self.device,
                    verbose=False,
                )

                # Extract detections
                detections = []
                if results and len(results) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = results[0].names[cls_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            "class": cls_name,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": round(conf, 3),
                        })

                self.frames_processed += 1
                self.total_detections += len(detections)

                detections_json = json.dumps(detections)

                if self.frames_processed % 50 == 0:
                    logger.info(
                        f"Video: {self.frames_processed} frames processed, "
                        f"{len(detections)} objects in latest"
                    )

                # Publish to SpacetimeDB (CreateFrame routes through Svelte API)
                loop.run_until_complete(
                    self.spacetime_client.call_reducer(
                        "CreateFrame",
                        session_id=self.session_id,
                        frame_type="webcam",
                        image_path=f"zmq_frame_{vframe.sequence}",
                        detections=detections_json,
                        reviewed=False,
                        notes=None,
                    )
                )

            except Exception as e:
                logger.error(f"Video inference error: {e}", exc_info=True)
