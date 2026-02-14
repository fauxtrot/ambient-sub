"""
Screen capture agent with YOLO object detection.

Captures screen at configurable rate and runs YOLO detection to understand user context.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import ImageGrab

from ..base import ProviderAgent
from ..events import Event
from ...providers.adapters.yolo_adapter import YOLOAdapter
from ...spacetime.client import SpacetimeClient
from .utils import save_frame_image, format_detections_json, get_active_window_info

logger = logging.getLogger(__name__)


class ScreenCaptureAgent(ProviderAgent):
    """
    Screen capture agent with YOLO object detection.

    Captures screen at configurable rate, runs YOLO detection to understand
    user context (coding, browsing, gaming, etc.), and stores results in SpacetimeDB.
    """

    def __init__(
        self,
        agent_id: str,
        spacetime_client: SpacetimeClient,
        fps: float = 1.0,
        yolo_model: str = "yolo26n",
        yolo_confidence: float = 0.5,
        output_dir: str = "data/frames",
        capture_active_window_only: bool = False,
        privacy_mode_apps: Optional[list] = None
    ):
        """
        Initialize screen capture agent.

        Args:
            agent_id: Unique agent identifier
            spacetime_client: SpacetimeDB client
            fps: Frames per second to capture (typically 0.5-2.0)
            yolo_model: YOLO model name (yolo26n, yolo11n, etc.)
            yolo_confidence: Confidence threshold for detections
            output_dir: Directory to save frames
            capture_active_window_only: If True, capture only active window
            privacy_mode_apps: List of app names to skip (e.g., ["banking.exe"])
        """
        # Define provider capabilities for screen capture + YOLO
        provider_capabilities = [
            ("vision_rgb", "object_detection"),
            ("vision_rgb", "scene_understanding"),
            ("vision_rgb", "context_inference")
        ]

        super().__init__(
            agent_id=agent_id,
            provider_capabilities=provider_capabilities,
            spacetime_conn=spacetime_client,
            config=None
        )

        self.spacetime_client = spacetime_client

        self.fps = fps
        self.output_dir = output_dir
        self.capture_active_window_only = capture_active_window_only
        self.privacy_mode_apps = privacy_mode_apps or []

        # Capture state
        self.running = False
        self.capture_task: Optional[asyncio.Task] = None

        # YOLO adapter
        self.yolo_adapter = YOLOAdapter(
            model_name=yolo_model,
            confidence_threshold=yolo_confidence
        )

        # Session tracking
        self.session_id: Optional[int] = None

        logger.info(f"ScreenCaptureAgent initialized: fps={fps}, model={yolo_model}")
        if privacy_mode_apps:
            logger.info(f"Privacy mode enabled for: {privacy_mode_apps}")

    async def setup(self) -> None:
        """Setup screen capture."""
        logger.info(f"Setting up ScreenCaptureAgent: {self.agent_id}")

        # Use default session ID for now
        # TODO: Implement session management via Svelte API
        self.session_id = 1
        logger.info(f"Session ID: {self.session_id}")

        # Start capture loop
        self.running = True
        self.capture_task = asyncio.create_task(self._capture_loop())

        logger.info("[OK] Screen capture ready")

    async def cleanup(self) -> None:
        """Cleanup screen capture resources."""
        logger.info(f"Cleaning up ScreenCaptureAgent: {self.agent_id}")

        self.running = False

        # Wait for capture task to finish
        if self.capture_task:
            await self.capture_task

        logger.info("[OK] ScreenCaptureAgent cleaned up")

    async def process(self, event: Event) -> None:
        """Process events (not used for provider agents)."""
        pass

    async def capture(self) -> Optional[dict]:
        """
        Capture a single screen frame.

        Returns:
            Dictionary with frame data, detections, and context, or None if skipped
        """
        # Get active window info
        window_info = get_active_window_info()
        process_name = window_info.get("process_name", "unknown")

        # Check privacy mode
        if self._should_skip_capture(process_name):
            logger.debug(f"Skipping capture: privacy mode for {process_name}")
            return None

        # Capture screenshot
        screenshot = ImageGrab.grab()

        # Convert to numpy array (RGB)
        frame_rgb = np.array(screenshot)

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

        # Infer context from detections
        context = self._infer_context(detections, window_info)

        # Save frame to disk
        image_path = save_frame_image(
            frame_rgb,
            frame_type="screen",
            session_id=self.session_id,
            timestamp=timestamp,
            output_dir=self.output_dir
        )

        return {
            'timestamp': timestamp,
            'image_path': image_path,
            'detections': detections,
            'window_info': window_info,
            'context': context,
            'detection_time': detection_time
        }

    async def publish_to_spacetime(self, data: dict) -> None:
        """
        Publish captured screen frame to SpacetimeDB.

        Args:
            data: Frame data from capture()
        """
        try:
            # Format detections as JSON
            detections_json = format_detections_json(data['detections'])

            # Add context and window info to notes
            notes = f"context: {data['context']}, window: {data['window_info'].get('window_title', 'unknown')}"

            # Call CreateFrame reducer
            await self.spacetime_client.call_reducer(
                "CreateFrame",
                session_id=self.session_id,
                frame_type="screen",
                image_path=data['image_path'],
                detections=detections_json,
                reviewed=False,
                notes=notes
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
                    # Skipped due to privacy mode
                    await asyncio.sleep(frame_interval)
                    continue

                # Publish to SpacetimeDB
                await self.publish_to_spacetime(frame_data)

                frame_count += 1

                # Log every 5 frames
                if frame_count % 5 == 0:
                    process_name = frame_data['window_info'].get('process_name', 'unknown')
                    logger.info(
                        f"Frame {frame_count}: {len(frame_data['detections'])} objects detected, "
                        f"context: {frame_data['context']}, window: {process_name} "
                        f"({frame_data['detection_time']:.1f}ms)"
                    )

            except Exception as e:
                logger.error(f"Error in capture loop: {e}", exc_info=True)

            # Maintain frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def _should_skip_capture(self, process_name: str) -> bool:
        """Check if capture should be skipped for privacy."""
        for app in self.privacy_mode_apps:
            if app.lower() in process_name.lower():
                return True
        return False

    def _infer_context(self, detections: list, window_info: dict) -> str:
        """
        Infer user context from detections and window info.

        Args:
            detections: YOLO detections
            window_info: Active window information

        Returns:
            Context string (e.g., "coding", "browsing", "gaming")
        """
        window_title = window_info.get("window_title", "").lower()
        process_name = window_info.get("process_name", "").lower()

        # Detect based on process name / window title
        if any(x in process_name for x in ["code", "vscode", "visual studio", "pycharm", "vim"]):
            return "coding"
        elif any(x in process_name for x in ["chrome", "firefox", "edge", "browser"]):
            return "browsing"
        elif any(x in process_name for x in ["discord", "slack", "teams", "zoom"]):
            return "communicating"
        elif any(x in window_title for x in ["terminal", "powershell", "cmd"]):
            return "terminal"
        elif any(x in process_name for x in ["game", "steam"]):
            return "gaming"

        # Detect based on YOLO detections
        detected_classes = {det["class"] for det in detections}

        if "laptop" in detected_classes or "keyboard" in detected_classes:
            return "working"
        elif "tv" in detected_classes or "remote" in detected_classes:
            return "watching"
        elif "book" in detected_classes:
            return "reading"

        return "unknown"

    def get_agent_info(self) -> dict:
        """Get agent information."""
        return {
            "agent_id": self.agent_id,
            "agent_type": "ScreenCaptureAgent",
            "status": "running" if self.running else "stopped",
            "fps": self.fps,
            "yolo_model": self.yolo_adapter.model_name,
            "privacy_mode": len(self.privacy_mode_apps) > 0,
            "session_id": self.session_id
        }
