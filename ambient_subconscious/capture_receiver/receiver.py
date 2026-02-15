"""Main entry point for the capture receiver.

Starts audio, video, and control receivers.
Can be run standalone for testing or imported by the agent coordinator.
"""

import logging
import signal
import time

from .audio_receiver import AudioReceiver
from .control_handler import ControlHandler
from .video_receiver import VideoReceiver

logger = logging.getLogger(__name__)


class CaptureReceiver:
    """Orchestrates all ZMQ receiver endpoints."""

    def __init__(
        self,
        audio_port: int = 5555,
        video_port: int = 5556,
        control_port: int = 5557,
        bind_ip: str = "*",
    ):
        self.audio = AudioReceiver(f"tcp://{bind_ip}:{audio_port}")
        self.video = VideoReceiver(f"tcp://{bind_ip}:{video_port}")
        self.control = ControlHandler(f"tcp://{bind_ip}:{control_port}")

    def start(self):
        logger.info("Starting capture receivers...")
        self.audio.start()
        self.video.start()
        self.control.start()
        logger.info(
            f"All receivers started.  "
            f"Audio :{self.audio.bind_address}  "
            f"Video :{self.video.bind_address}  "
            f"Control :{self.control.bind_address}"
        )

    def stop(self):
        logger.info("Stopping capture receivers...")
        self.audio.stop()
        self.video.stop()
        self.control.stop()
        logger.info("All receivers stopped.")

    def status(self) -> dict:
        return {
            "audio": {
                "chunks_received": self.audio.chunks_received,
                "bytes_received": self.audio.bytes_received,
                "last_ts": self.audio.last_chunk_ts,
                "diart_queue_size": self.audio.diart_queue.qsize(),
                "whisper_queue_size": self.audio.whisper_queue.qsize(),
            },
            "video": {
                "frames_received": self.video.frames_received,
                "bytes_received": self.video.bytes_received,
                "last_ts": self.video.last_frame_ts,
                "yolo_queue_size": self.video.frame_queue.qsize(),
            },
            "control": {
                "connected_clients": list(self.control.connected_clients.keys()),
            },
        }


def main():
    """Standalone test -- run the receiver and print stats."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    receiver = CaptureReceiver()
    receiver.start()

    def shutdown(sig, frame):
        receiver.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("\nCapture receiver running. Waiting for connections...")
    print("  Audio  : tcp://*:5555")
    print("  Video  : tcp://*:5556")
    print("  Control: tcp://*:5557")
    print("Press Ctrl+C to stop.\n")

    while True:
        time.sleep(5)
        s = receiver.status()
        print(
            f"  Audio: {s['audio']['chunks_received']} chunks, "
            f"{s['audio']['bytes_received'] / 1024:.0f} KB  "
            f"(diart_q={s['audio']['diart_queue_size']}, "
            f"whisper_q={s['audio']['whisper_queue_size']})  |  "
            f"Video: {s['video']['frames_received']} frames, "
            f"{s['video']['bytes_received'] / 1024:.0f} KB  "
            f"(yolo_q={s['video']['yolo_queue_size']})  |  "
            f"Clients: {s['control']['connected_clients']}"
        )


if __name__ == "__main__":
    main()
