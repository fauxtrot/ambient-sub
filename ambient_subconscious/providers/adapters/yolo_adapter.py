"""
YOLO Provider Adapter for object detection.

Wraps YOLO models (v26n, v11n, v12n, v8n) for vision-based object detection
and scene understanding.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from ..base import ProviderAdapter
from ..m4_token import M4Token

logger = logging.getLogger(__name__)


class YOLOAdapter(ProviderAdapter):
    """
    Adapter for YOLO object detection models.

    Capabilities provided:
    - ("vision_rgb", "object_detection")
    - ("vision_rgb", "scene_understanding")
    """

    def __init__(
        self,
        model_name: str = "yolo26n",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        class_filter: Optional[List[str]] = None,
        model_cache_dir: str = ".models"
    ):
        """
        Initialize YOLO adapter.

        Args:
            model_name: YOLO model to use (yolo26n, yolo11n, yolo12n, yolov8n)
            device: Device to run on ("cuda" or "cpu")
            confidence_threshold: Minimum confidence for detections (0-1)
            class_filter: Optional list of class names to detect (None = all classes)
            model_cache_dir: Directory to cache model weights
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.class_filter = class_filter
        self.model_cache_dir = model_cache_dir

        # Load model
        logger.info(f"Loading YOLO model: {model_name}")
        try:
            model_path = f"{model_cache_dir}/{model_name}.pt"
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"YOLO model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

        # Model metadata
        self.class_names = self.model.names  # Dict: {class_id: class_name}

        logger.info(f"YOLO adapter initialized: {model_name} on {self.device}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        if class_filter:
            logger.info(f"Class filter: {class_filter}")

    def get_capabilities(self) -> List[Tuple[str, str]]:
        """Get capabilities provided by this adapter."""
        return [
            ("vision_rgb", "object_detection"),
            ("vision_rgb", "scene_understanding")
        ]

    def translate_input(self, input_data: Any) -> Any:
        """
        Translate input to YOLO format.

        Args:
            input_data: Can be:
                - numpy array (H, W, C) in RGB
                - PIL Image
                - file path string
                - M4Token with raw_data["image"]

        Returns:
            Input in format acceptable by YOLO
        """
        if isinstance(input_data, M4Token):
            # Extract image from M4Token
            if input_data.raw_data and "image" in input_data.raw_data:
                return input_data.raw_data["image"]
            else:
                raise ValueError("M4Token missing image in raw_data")

        # YOLO accepts numpy arrays, PIL Images, and file paths directly
        return input_data

    def translate_output(
        self,
        provider_output,
        original_input: Any,
        timestamp: float,
        duration_ms: float
    ) -> M4Token:
        """
        Translate YOLO output to M4Token.

        Args:
            provider_output: YOLO Results object
            original_input: Original input data
            timestamp: Timestamp of frame
            duration_ms: Processing duration

        Returns:
            M4Token with object detection annotations
        """
        # Extract detections from YOLO results
        results = provider_output[0]  # Single image result
        boxes = results.boxes

        # Build detection list
        detections = []
        confidences = []

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = self.class_names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            # Skip if below confidence threshold
            if conf < self.confidence_threshold:
                continue

            # Skip if not in class filter
            if self.class_filter and class_name not in self.class_filter:
                continue

            # Add detection
            detection = {
                "class": class_name,
                "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],  # x1, y1, x2, y2
                "confidence": conf,
                "class_id": cls_id
            }
            detections.append(detection)
            confidences.append(conf)

        # Calculate average confidence
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Create M4Token
        token = M4Token(
            modality="vision_rgb",
            source=f"yolo_{self.model_name}",
            timestamp=timestamp,
            duration_ms=duration_ms
        )

        # Add object detection annotation
        token.add_annotation(
            capability="object_detection",
            value=detections,
            confidence=avg_confidence,
            model=self.model_name,
            device=self.device,
            latency_ms=duration_ms
        )

        # Add scene understanding annotation (high-level summary)
        scene_summary = self._generate_scene_summary(detections)
        token.add_annotation(
            capability="scene_understanding",
            value=scene_summary,
            confidence=avg_confidence,
            model=self.model_name
        )

        # Store raw image reference if available
        if isinstance(original_input, np.ndarray):
            token.raw_data = {
                "image_shape": original_input.shape,
                "num_detections": len(detections)
            }

        return token

    def _generate_scene_summary(self, detections: List[Dict]) -> str:
        """
        Generate high-level scene understanding from detections.

        Args:
            detections: List of detection dictionaries

        Returns:
            Human-readable scene summary
        """
        if not detections:
            return "empty scene"

        # Count object types
        object_counts = {}
        for det in detections:
            class_name = det["class"]
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

        # Generate summary
        if len(object_counts) == 0:
            return "empty scene"
        elif len(object_counts) == 1:
            class_name, count = list(object_counts.items())[0]
            if count == 1:
                return f"{class_name}"
            else:
                return f"multiple {class_name}s"
        else:
            # Multiple object types
            top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            summary_parts = [f"{count} {name}{'s' if count > 1 else ''}" for name, count in top_objects]
            return ", ".join(summary_parts)

    def process(
        self,
        input_data: Any,
        timestamp: Optional[float] = None,
        duration_ms: Optional[float] = None
    ) -> M4Token:
        """
        Process input through YOLO and return M4Token.

        Args:
            input_data: Input image data
            timestamp: Optional timestamp (uses current time if None)
            duration_ms: Optional duration (calculated if None)

        Returns:
            M4Token with detections
        """
        if timestamp is None:
            timestamp = time.time()

        # Translate input
        yolo_input = self.translate_input(input_data)

        # Run YOLO inference
        start = time.time()
        results = self.model(yolo_input, verbose=False)
        inference_time = (time.time() - start) * 1000  # ms

        if duration_ms is None:
            duration_ms = inference_time

        # Translate output to M4Token
        token = self.translate_output(
            provider_output=results,
            original_input=input_data,
            timestamp=timestamp,
            duration_ms=duration_ms
        )

        return token

    def get_metadata(self) -> Dict[str, Any]:
        """Get adapter metadata."""
        return {
            "adapter_type": "YOLOAdapter",
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "class_filter": self.class_filter,
            "num_classes": len(self.class_names),
            "capabilities": self.get_capabilities()
        }
