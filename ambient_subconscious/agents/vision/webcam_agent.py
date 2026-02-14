"""
Webcam capture agent with YOLO object detection.

Captures frames from webcam at configurable rate and runs YOLO detection.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..base import ProviderAgent
from ..events import Event
from ...providers.adapters.yolo_adapter import YOLOAdapter
from ...spacetime.client import SpacetimeClient
from .utils import save_frame_image, format_detections_json

logger = logging.getLogger(__name__)


class WebcamAgent(ProviderAgent):
    """
    Webcam capture agent with YOLO object detection.

    Captures frames at configurable rate, runs YOLO detection,
    and stores results in SpacetimeDB.
    """

    def __init__(
        self,
        agent_id: str,
        spacetime_client: SpacetimeClient,
        camera_index: int = 0,
        fps: float = 2.0,
        resolution: tuple = (1280, 720),
        yolo_model: str = "yolo26n",
        yolo_confidence: float = 0.5,
        output_dir: str = "data/frames",
        warmup_frames: int = 5
    ):
        """
        Initialize webcam agent.

        Args:
            agent_id: Unique agent identifier
            spacetime_client: SpacetimeDB client
            camera_index: Camera device index (0 for default)
            fps: Frames per second to capture
            resolution: Target resolution (width, height)
            yolo_model: YOLO model name (yolo26n, yolo11n, etc.)
            yolo_confidence: Confidence threshold for detections
            output_dir: Directory to save frames
            warmup_frames: Number of frames to skip on startup
        """
        # Define provider capabilities for webcam + YOLO
        provider_capabilities = [
            ("vision_rgb", "object_detection"),
            ("vision_rgb", "scene_understanding")
        ]

        super().__init__(
            agent_id=agent_id,
            provider_capabilities=provider_capabilities,
            spacetime_conn=spacetime_client,
            config=None
        )

        self.spacetime_client = spacetime_client

        self.camera_index = camera_index
        self.fps = fps
        self.resolution = resolution
        self.output_dir = output_dir
        self.warmup_frames = warmup_frames

        # Webcam capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.capture_task: Optional[asyncio.Task] = None

        # YOLO adapter
        self.yolo_adapter = YOLOAdapter(
            model_name=yolo_model,
            confidence_threshold=yolo_confidence
        )

        # Session tracking
        self.session_id: Optional[int] = None

        logger.info(f"WebcamAgent initialized: camera={camera_index}, fps={fps}, model={yolo_model}")

    async def setup(self) -> None:
        """Setup webcam and YOLO."""
        logger.info(f"Setting up WebcamAgent: {self.agent_id}")

        # Use default session ID for now
        # TODO: Implement session management via Svelte API
        self.session_id = 1
        logger.info(f"Session ID: {self.session_id}")

        # Open webcam
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            raise RuntimeError(f"Could not open camera {self.camera_index}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Warmup camera
        logger.info(f"Warming up camera ({self.warmup_frames} frames)...")
        for _ in range(self.warmup_frames):
            self.cap.read()
        await asyncio.sleep(0.5)

        logger.info("[OK] Camera ready")

        # Start capture loop
        self.running = True
        self.capture_task = asyncio.create_task(self._capture_loop())

    async def cleanup(self) -> None:
        """Cleanup webcam resources."""
        logger.info(f"Cleaning up WebcamAgent: {self.agent_id}")

        self.running = False

        # Wait for capture task to finish
        if self.capture_task:
            await self.capture_task

        # Release camera
        if self.cap:
            self.cap.release()

        logger.info("[OK] WebcamAgent cleaned up")

    async def process(self, event: Event) -> None:
        """Process events (not used for provider agents)."""
        pass

    async def capture(self) -> Optional[dict]:
        """
        Capture a single frame.

        Returns:
            Dictionary with frame data and detections, or None if capture failed
        """
        if not self.cap or not self.cap.isOpened():
            logger.warning("Camera not initialized")
            return None

        # Capture frame
        ret, frame = self.cap.read()

        if not ret:
            logger.warning("Failed to capture frame")
            return None

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get timestamp
        timestamp = time.time()

        # Run YOLO detection
        detection_start = time.time()
        token = self.yolo_adapter.process(frame_rgb, timestamp=timestamp)
        detection_time = (time.time() - detection_start) * 1000

        # Extract detections
        detections = []
        if token.annotations and "object_detection" in token.annotations:
            detections = token.annotations["object_detection"]["value"]

        # Save frame to disk
        image_path = save_frame_image(
            frame_rgb,
            frame_type="webcam",
            session_id=self.session_id,
            timestamp=timestamp,
            output_dir=self.output_dir
        )

        return {
            'timestamp': timestamp,
            'image_path': image_path,
            'detections': detections,
            'detection_time': detection_time
        }

    async def publish_to_spacetime(self, data: dict) -> None:
        """
        Publish captured frame to SpacetimeDB.

        Args:
            data: Frame data from capture()
        """
        try:
            # Format detections as JSON
            detections_json = format_detections_json(data['detections'])

            # Call CreateFrame reducer
            await self.spacetime_client.call_reducer(
                "CreateFrame",
                session_id=self.session_id,
                frame_type="webcam",
                image_path=data['image_path'],
                detections=detections_json,
                reviewed=False,
                notes=None
            )

        except Exception as e:
            logger.error(f"Failed to publish frame to SpacetimeDB: {e}")

    async def _capture_loop(self) -> None:
        """Main capture loop."""
        frame_interval = 1.0 / self.fps
        frame_count = 0

        logger.info(f"Starting capture loop: {self.fps} fps (interval: {frame_interval:.2f}s)")

        while self.running:
            loop_start = time.time()

            try:
                # Capture frame using the abstract method
                frame_data = await self.capture()

                if frame_data is None:
                    logger.warning("Failed to capture frame, retrying...")
                    await asyncio.sleep(0.1)
                    continue

                # Publish to SpacetimeDB
                await self.publish_to_spacetime(frame_data)

                frame_count += 1

                # Log every 10 frames
                if frame_count % 10 == 0:
                    logger.info(
                        f"Frame {frame_count}: {len(frame_data['detections'])} objects detected "
                        f"({frame_data['detection_time']:.1f}ms)"
                    )

            except Exception as e:
                logger.error(f"Error in capture loop: {e}", exc_info=True)

            # Maintain frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def get_agent_info(self) -> dict:
        """Get agent information."""
        return {
            "agent_id": self.agent_id,
            "agent_type": "WebcamAgent",
            "status": "running" if self.running else "stopped",
            "camera_index": self.camera_index,
            "fps": self.fps,
            "resolution": self.resolution,
            "yolo_model": self.yolo_adapter.model_name,
            "session_id": self.session_id
        }
