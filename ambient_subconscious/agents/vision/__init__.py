"""
Vision agents for webcam and screen capture.
"""

from .utils import save_frame_image, format_detections_json
from .webcam_agent import WebcamAgent
from .screen_capture_agent import ScreenCaptureAgent

__all__ = [
    'save_frame_image',
    'format_detections_json',
    'WebcamAgent',
    'ScreenCaptureAgent'
]
