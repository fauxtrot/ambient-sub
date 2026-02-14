"""
Shared utilities for vision agents.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def save_frame_image(
    image: np.ndarray,
    frame_type: str,
    session_id: int,
    timestamp: float,
    output_dir: str = "data/frames"
) -> str:
    """
    Save frame image to disk.

    Args:
        image: Image as numpy array (H, W, C) in RGB
        frame_type: Type of frame ("webcam" or "screen")
        session_id: Session ID
        timestamp: Unix timestamp
        output_dir: Base directory for frames

    Returns:
        Relative path to saved image
    """
    # Create output directory
    output_path = Path(output_dir) / frame_type / str(session_id)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename from timestamp
    dt = datetime.fromtimestamp(timestamp)
    filename = f"{dt.strftime('%Y%m%d_%H%M%S')}_{int((timestamp % 1) * 1000):03d}.jpg"
    file_path = output_path / filename

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save with JPEG compression
    cv2.imwrite(str(file_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Return relative path
    relative_path = str(file_path.relative_to(Path(output_dir).parent))
    logger.debug(f"Saved frame: {relative_path}")

    return relative_path


def format_detections_json(detections: List[Dict[str, Any]]) -> str:
    """
    Format YOLO detections as JSON string for SpacetimeDB.

    Args:
        detections: List of detection dictionaries from YOLO

    Returns:
        JSON string representation
    """
    # Simplify detections for storage
    simplified = []
    for det in detections:
        simplified.append({
            "class": det["class"],
            "bbox": det["bbox"],
            "confidence": round(det["confidence"], 3)
        })

    return json.dumps(simplified)


def resize_frame(
    image: np.ndarray,
    target_width: int = 640,
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize frame to target resolution.

    Args:
        image: Input image (H, W, C)
        target_width: Target width in pixels
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if maintain_aspect:
        # Calculate new height to maintain aspect ratio
        aspect_ratio = h / w
        target_height = int(target_width * aspect_ratio)
    else:
        target_height = target_width

    # Resize using high-quality interpolation
    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_LANCZOS4
    )

    return resized


def crop_to_roi(
    image: np.ndarray,
    bbox: tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop image to region of interest.

    Args:
        image: Input image (H, W, C)
        bbox: Bounding box (x1, y1, x2, y2)

    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw detection bounding boxes on image for visualization.

    Args:
        image: Input image (H, W, C) in RGB
        detections: List of detection dictionaries
        color: Box color (R, G, B)
        thickness: Box line thickness

    Returns:
        Image with boxes drawn
    """
    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for det in detections:
        bbox = det["bbox"]
        class_name = det["class"]
        confidence = det["confidence"]

        x1, y1, x2, y2 = map(int, bbox)

        # Draw box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color[::-1], thickness)

        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 10, label_size[1] + 10)

        cv2.rectangle(
            img_bgr,
            (x1, label_y - label_size[1] - 10),
            (x1 + label_size[0] + 10, label_y),
            color[::-1],
            -1
        )
        cv2.putText(
            img_bgr,
            label,
            (x1 + 5, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # Convert back to RGB
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def get_active_window_info() -> Dict[str, str]:
    """
    Get information about the active window (Windows only).

    Returns:
        Dictionary with window_title, process_name, etc.
    """
    try:
        import win32gui
        import win32process
        import psutil

        # Get foreground window
        hwnd = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(hwnd)

        # Get process info
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            process = psutil.Process(pid)
            process_name = process.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_name = "unknown"

        return {
            "window_title": window_title,
            "process_name": process_name,
            "pid": pid
        }

    except ImportError:
        # Win32 APIs not available (not Windows or packages not installed)
        return {
            "window_title": "unknown",
            "process_name": "unknown",
            "pid": -1
        }
    except Exception as e:
        logger.warning(f"Could not get active window info: {e}")
        return {
            "window_title": "unknown",
            "process_name": "unknown",
            "pid": -1
        }


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.

    Args:
        pil_image: PIL Image object

    Returns:
        Numpy array (H, W, C) in RGB
    """
    return np.array(pil_image)


def numpy_to_pil(np_image: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.

    Args:
        np_image: Numpy array (H, W, C) in RGB

    Returns:
        PIL Image object
    """
    return Image.fromarray(np_image)
