"""ZMQ message protocol — single-frame length-prefixed format.

Wire format (one ZMQ frame per message):
  Bytes 0-3:   header_len  (big-endian uint32)
  Bytes 4-7:   payload_len (big-endian uint32)
  Bytes 8..8+N:    JSON header (UTF-8 encoded)
  Bytes 8+N..end:  raw binary payload

Audio header:
  {"type": "audio", "ts": <unix_ms>, "seq": <int>, "sr": 16000, "ch": 1, "fmt": "int16"}
  Payload: raw PCM int16 bytes (little-endian)

Video header:
  {"type": "video", "ts": <unix_ms>, "seq": <int>, "w": 640, "h": 480, "fmt": "jpeg", "quality": 70}
  Payload: JPEG bytes
"""

import json
import logging
import struct
from dataclasses import dataclass
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Decoded audio chunk ready for pipeline consumption."""
    timestamp_ms: int
    sequence: int
    sample_rate: int
    channels: int
    samples: np.ndarray  # float32, shape (num_samples,)


@dataclass
class VideoFrame:
    """Decoded video frame header + raw JPEG bytes."""
    timestamp_ms: int
    sequence: int
    width: int
    height: int
    jpeg_bytes: bytes
    # Decoded on demand by consumer via cv2.imdecode()


def unpack_frame(raw: bytes) -> tuple[dict, bytes]:
    """Unpack a single ZMQ frame into (header_dict, payload_bytes).

    Wire layout:
      [4B header_len][4B payload_len][header JSON][payload]
    """
    if len(raw) < 8:
        raise ValueError(f"Frame too short: {len(raw)} bytes (need at least 8)")

    header_len = struct.unpack(">I", raw[0:4])[0]
    payload_len = struct.unpack(">I", raw[4:8])[0]

    expected = 8 + header_len + payload_len
    if len(raw) < expected:
        raise ValueError(
            f"Frame truncated: got {len(raw)} bytes, "
            f"expected {expected} (header={header_len}, payload={payload_len})"
        )

    header_json = raw[8 : 8 + header_len].decode("utf-8")
    payload = raw[8 + header_len : 8 + header_len + payload_len]

    return json.loads(header_json), payload


def parse_message(raw: bytes) -> Union[AudioChunk, VideoFrame]:
    """Parse a single ZMQ frame into an AudioChunk or VideoFrame.

    Accepts the raw bytes from socket.recv() (single frame, not multipart).
    Raises ValueError on malformed messages.
    """
    header, payload = unpack_frame(raw)
    msg_type = header.get("type")

    if msg_type == "audio":
        pcm = np.frombuffer(payload, dtype=np.int16)
        samples = pcm.astype(np.float32) / 32768.0
        return AudioChunk(
            timestamp_ms=header["ts"],
            sequence=header["seq"],
            sample_rate=header.get("sr", 16000),
            channels=header.get("ch", 1),
            samples=samples,
        )
    elif msg_type == "video":
        return VideoFrame(
            timestamp_ms=header["ts"],
            sequence=header["seq"],
            width=header.get("w", 640),
            height=header.get("h", 480),
            jpeg_bytes=payload,
        )
    else:
        raise ValueError(f"Unknown message type: {msg_type!r}")
