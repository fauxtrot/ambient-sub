"""ZMQ port monitor — listens on all 3 channels and prints incoming data.

No inference, no routing — just shows what arrives.

Wire format (single ZMQ frame, length-prefixed):
  [4B header_len BE][4B payload_len BE][JSON header UTF-8][payload bytes]

Usage:
    python tests/zmq_monitor.py
    python tests/zmq_monitor.py --bind 0.0.0.0
"""

import argparse
import json
import signal
import struct
import sys
import threading
import time

import numpy as np
import zmq


def unpack_frame(raw: bytes) -> tuple[dict, bytes]:
    """Unpack length-prefixed single frame into (header, payload)."""
    if len(raw) < 8:
        raise ValueError(f"Frame too short: {len(raw)} bytes")
    header_len = struct.unpack(">I", raw[0:4])[0]
    payload_len = struct.unpack(">I", raw[4:8])[0]
    header_json = raw[8 : 8 + header_len].decode("utf-8")
    payload = raw[8 + header_len : 8 + header_len + payload_len]
    return json.loads(header_json), payload


def monitor_audio(bind_ip: str, port: int = 5555):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.bind(f"tcp://{bind_ip}:{port}")

    chunks = 0
    total_bytes = 0
    while True:
        try:
            raw = sock.recv()
            header, payload = unpack_frame(raw)
            chunks += 1
            total_bytes += len(payload)

            samples = np.frombuffer(payload, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            peak = np.max(np.abs(samples)) if len(samples) > 0 else 0

            print(
                f"  AUDIO #{chunks:>5d}  seq={header.get('seq', '?'):>5}  "
                f"ts={header.get('ts', 0)}  "
                f"samples={len(samples):>5d}  "
                f"rms={rms:>6.0f}  peak={peak:>5d}  "
                f"({len(payload)} bytes)"
            )
        except zmq.Again:
            continue
        except Exception as e:
            print(f"  AUDIO ERROR: {e}")


def monitor_video(bind_ip: str, port: int = 5556):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.bind(f"tcp://{bind_ip}:{port}")

    frames_count = 0
    total_bytes = 0
    while True:
        try:
            raw = sock.recv()
            header, payload = unpack_frame(raw)
            frames_count += 1
            total_bytes += len(payload)

            print(
                f"  VIDEO #{frames_count:>5d}  seq={header.get('seq', '?'):>5}  "
                f"ts={header.get('ts', 0)}  "
                f"{header.get('w', '?')}x{header.get('h', '?')}  "
                f"fmt={header.get('fmt', '?')}  "
                f"({len(payload) / 1024:.1f} KB)"
            )
        except zmq.Again:
            continue
        except Exception as e:
            print(f"  VIDEO ERROR: {e}")


def monitor_control(bind_ip: str, port: int = 5557):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.bind(f"tcp://{bind_ip}:{port}")

    config = {
        "audio_chunk_ms": 100,
        "audio_sample_rate": 16000,
        "audio_channels": 1,
        "audio_format": "int16",
        "video_fps": 5,
        "video_res": [640, 480],
        "video_format": "jpeg",
        "video_quality": 70,
    }

    while True:
        try:
            raw = sock.recv()
            # Control channel may be plain JSON or length-prefixed
            # Try plain JSON first (simpler for REQ/REP)
            try:
                msg = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Try length-prefixed format
                header, _ = unpack_frame(raw)
                msg = header

            cmd = msg.get("cmd", "?")
            client = msg.get("client", "?")

            if cmd == "hello":
                print(f"  CTRL   HELLO from '{client}'  caps={msg.get('capabilities', [])}")
                reply = {"status": "ok", "config": config}
            elif cmd == "heartbeat":
                print(f"  CTRL   HEARTBEAT from '{client}'")
                reply = {"status": "ok", "ts": int(time.time() * 1000)}
            else:
                print(f"  CTRL   {cmd} from '{client}'  -> {json.dumps(msg)}")
                reply = {"status": "ok"}

            sock.send(json.dumps(reply).encode("utf-8"))
        except zmq.Again:
            continue
        except Exception as e:
            print(f"  CTRL   ERROR: {e}")
            try:
                sock.send(json.dumps({"error": str(e)}).encode("utf-8"))
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="ZMQ port monitor")
    parser.add_argument("--bind", default="*", help="Bind IP (default: * = all interfaces)")
    parser.add_argument("--audio-port", type=int, default=5555)
    parser.add_argument("--video-port", type=int, default=5556)
    parser.add_argument("--control-port", type=int, default=5557)
    args = parser.parse_args()

    print(f"ZMQ Monitor -- listening on all channels")
    print(f"  Audio  : tcp://{args.bind}:{args.audio_port}")
    print(f"  Video  : tcp://{args.bind}:{args.video_port}")
    print(f"  Control: tcp://{args.bind}:{args.control_port}")
    print(f"Waiting for data... (Ctrl+C to stop)\n")

    threads = [
        threading.Thread(target=monitor_audio, args=(args.bind, args.audio_port), daemon=True),
        threading.Thread(target=monitor_video, args=(args.bind, args.video_port), daemon=True),
        threading.Thread(target=monitor_control, args=(args.bind, args.control_port), daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
