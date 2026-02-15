"""ZMQ REP handler for config negotiation and heartbeat.

The Redot client sends:
  {"cmd": "hello", "client": "selestia", "capabilities": ["audio", "video"]}
  {"cmd": "heartbeat", "client": "selestia", "uptime_s": ...}

The server responds with capture configuration.
"""

import json
import logging
import threading
import time

import zmq

logger = logging.getLogger(__name__)


class ControlHandler:
    """ZMQ REP socket for client config negotiation and heartbeat."""

    def __init__(
        self,
        bind_address: str = "tcp://*:5557",
        audio_chunk_ms: int = 100,
        video_fps: int = 5,
        video_res: tuple[int, int] = (640, 480),
    ):
        self.bind_address = bind_address
        self.config = {
            "audio_chunk_ms": audio_chunk_ms,
            "audio_sample_rate": 16000,
            "audio_channels": 1,
            "audio_format": "int16",
            "video_fps": video_fps,
            "video_res": list(video_res),
            "video_format": "jpeg",
            "video_quality": 70,
        }

        # Client tracking
        self.connected_clients: dict[str, dict] = {}
        self.last_heartbeat: dict[str, float] = {}

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._handle_loop, daemon=True, name="control")
        self._thread.start()
        logger.info(f"ControlHandler listening on {self.bind_address}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _handle_loop(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.bind(self.bind_address)

        try:
            while self._running:
                try:
                    raw = sock.recv()
                    msg = json.loads(raw.decode("utf-8"))
                    reply = self._handle_message(msg)
                    sock.send(json.dumps(reply).encode("utf-8"))
                except zmq.Again:
                    continue
                except json.JSONDecodeError as e:
                    logger.warning(f"Control: invalid JSON: {e}")
                    sock.send(json.dumps({"error": "invalid JSON"}).encode("utf-8"))
                except Exception as e:
                    logger.error(f"ControlHandler error: {e}", exc_info=True)
                    try:
                        sock.send(json.dumps({"error": str(e)}).encode("utf-8"))
                    except Exception:
                        pass
        finally:
            sock.close()
            ctx.term()
            logger.info("ControlHandler stopped.")

    def _handle_message(self, msg: dict) -> dict:
        cmd = msg.get("cmd", "")
        client = msg.get("client", "unknown")

        if cmd == "hello":
            caps = msg.get("capabilities", [])
            self.connected_clients[client] = {
                "capabilities": caps,
                "connected_at": time.time(),
            }
            self.last_heartbeat[client] = time.time()
            logger.info(f"Client connected: {client} (capabilities: {caps})")
            return {"status": "ok", "config": self.config}

        elif cmd == "heartbeat":
            self.last_heartbeat[client] = time.time()
            return {"status": "ok", "ts": int(time.time() * 1000)}

        elif cmd == "get_config":
            return {"status": "ok", "config": self.config}

        else:
            logger.warning(f"Control: unknown command {cmd!r} from {client}")
            return {"error": f"unknown command: {cmd}"}
