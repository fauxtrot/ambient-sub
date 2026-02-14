"""Configuration management for stream storage"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class StreamConfig:
    """Load configuration from environment variables"""

    @staticmethod
    def get_storage_path() -> str:
        """Get base storage path for sessions"""
        return os.getenv(
            "STREAM_STORAGE_PATH",
            "data/sessions"
        )

    @staticmethod
    def get_storage_format() -> str:
        """Get storage format (jsonl, binary, spacetimedb)"""
        return os.getenv("STREAM_STORAGE_FORMAT", "jsonl")

    @staticmethod
    def get_device_name() -> str:
        """Get device name for session metadata"""
        return os.getenv("DEVICE_NAME", "Unknown")

    @staticmethod
    def get_recording_mode() -> str:
        """Get recording mode (ambient, meeting, focus)"""
        return os.getenv("RECORDING_MODE", "ambient")

    @staticmethod
    def get_inference_device() -> str:
        """Get CUDA/CPU device for audio processing"""
        device = os.getenv("INFERENCE_DEVICE", "gpu")
        # Convert "gpu" to "cuda" for PyTorch
        if device.lower() == "gpu":
            return "cuda"
        return device.lower()

    @staticmethod
    def get_record_audio() -> bool:
        """Whether to record raw audio stream"""
        return os.getenv("RECORD_RAW_AUDIO", "false").lower() == "true"

    @staticmethod
    def get_recordings_path() -> str:
        """Get path for audio recordings"""
        return os.getenv("RECORDINGS_PATH", "recordings")

    @staticmethod
    def get_audio_input_device() -> Optional[int]:
        """Get audio input device index (None = default)"""
        device_str = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
        if device_str:
            try:
                return int(device_str)
            except ValueError:
                return None
        return None

    @staticmethod
    def get_recording_duration() -> Optional[float]:
        """Get recording duration in seconds (None = unlimited)"""
        duration_str = os.getenv("RECORDING_DURATION", "").strip()
        if duration_str:
            try:
                duration = float(duration_str)
                # Return None for 0 or negative (unlimited)
                return duration if duration > 0 else None
            except ValueError:
                return None
        return None
