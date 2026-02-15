"""Synthetic ZMQ sender — simulates the Redot client for pipeline testing.

Uses the same single-frame length-prefixed wire format as the Godot client:
  [4B header_len BE][4B payload_len BE][JSON header UTF-8][payload bytes]

Usage:
    python tests/test_zmq_sender.py                  # default localhost
    python tests/test_zmq_sender.py 192.168.1.100    # remote server IP
    python tests/test_zmq_sender.py --duration 30    # run for 30 seconds
"""

import argparse
import json
import struct
import sys
import threading
import time

import numpy as np
import zmq


def pack_frame(header: dict, payload: bytes) -> bytes:
    """Pack header + payload into the length-prefixed wire format."""
    header_bytes = json.dumps(header).encode("utf-8")
    return (
        struct.pack(">I", len(header_bytes))
        + struct.pack(">I", len(payload))
        + header_bytes
        + payload
    )


def make_audio_chunk(seq: int, chunk_ms: int = 100, sample_rate: int = 16000) -> bytes:
    """Generate a sine-wave audio chunk as a packed single frame."""
    n_samples = int(sample_rate * chunk_ms / 1000)
    t = np.linspace(seq * chunk_ms / 1000, (seq + 1) * chunk_ms / 1000, n_samples, endpoint=False)
    freq = 440 + 100 * np.sin(2 * np.pi * 0.5 * t)
    wave = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)

    header = {
        "type": "audio",
        "ts": int(time.time() * 1000),
        "seq": seq,
        "sr": sample_rate,
        "ch": 1,
        "fmt": "int16",
    }
    return pack_frame(header, wave.tobytes())


def make_video_frame(seq: int, width: int = 640, height: int = 480) -> bytes:
    """Generate a synthetic JPEG video frame as a packed single frame."""
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(40, 80, width, dtype=np.uint8)
    img[:, :, 1] = np.linspace(60, 120, width, dtype=np.uint8)
    img[:, :, 2] = 50
    cv2.putText(img, f"Frame {seq}", (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(img, time.strftime("%H:%M:%S"), (20, height // 2 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])

    header = {
        "type": "video",
        "ts": int(time.time() * 1000),
        "seq": seq,
        "w": width,
        "h": height,
        "fmt": "jpeg",
        "quality": 70,
    }
    return pack_frame(header, jpeg.tobytes())


def run_control(server_ip: str, port: int = 5557):
    """Send hello + periodic heartbeats on the control channel."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    sock.connect(f"tcp://{server_ip}:{port}")

    hello = {"cmd": "hello", "client": "test_sender", "capabilities": ["audio", "video"]}
    sock.send(json.dumps(hello).encode("utf-8"))
    try:
        reply = json.loads(sock.recv().decode("utf-8"))
        print(f"  Control: connected -- server config: {reply.get('config', {})}")
    except zmq.Again:
        print("  Control: no reply from server (timeout)")
        sock.close()
        ctx.term()
        return

    while True:
        time.sleep(5)
        try:
            hb = {"cmd": "heartbeat", "client": "test_sender"}
            sock.send(json.dumps(hb).encode("utf-8"))
            sock.recv()
        except Exception:
            break

    sock.close()
    ctx.term()


def main():
    parser = argparse.ArgumentParser(description="Synthetic ZMQ sender for pipeline testing")
    parser.add_argument("server", nargs="?", default="localhost", help="Server IP (default: localhost)")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds (default: 10)")
    parser.add_argument("--audio-port", type=int, default=5555)
    parser.add_argument("--video-port", type=int, default=5556)
    parser.add_argument("--control-port", type=int, default=5557)
    parser.add_argument("--chunk-ms", type=int, default=100, help="Audio chunk size in ms")
    parser.add_argument("--video-fps", type=int, default=5, help="Video frames per second")
    args = parser.parse_args()

    ctx = zmq.Context()

    audio_sock = ctx.socket(zmq.PUSH)
    audio_sock.connect(f"tcp://{args.server}:{args.audio_port}")

    video_sock = ctx.socket(zmq.PUSH)
    video_sock.connect(f"tcp://{args.server}:{args.video_port}")

    ctrl_thread = threading.Thread(
        target=run_control, args=(args.server, args.control_port), daemon=True
    )
    ctrl_thread.start()

    print(f"\nSending to {args.server} for {args.duration}s...")
    print(f"  Audio: {args.chunk_ms}ms chunks @ 16kHz -> port {args.audio_port}")
    print(f"  Video: {args.video_fps} FPS JPEG -> port {args.video_port}")
    print()

    audio_seq = 0
    video_seq = 0
    audio_interval = args.chunk_ms / 1000.0
    video_interval = 1.0 / args.video_fps
    next_audio = time.time()
    next_video = time.time()
    start = time.time()
    audio_bytes = 0
    video_bytes = 0

    try:
        while time.time() - start < args.duration:
            now = time.time()

            if now >= next_audio:
                frame = make_audio_chunk(audio_seq, args.chunk_ms)
                audio_sock.send(frame)
                audio_bytes += len(frame)
                audio_seq += 1
                next_audio += audio_interval

            if now >= next_video:
                frame = make_video_frame(video_seq)
                video_sock.send(frame)
                video_bytes += len(frame)
                video_seq += 1
                next_video += video_interval

            sleep_time = min(next_audio - time.time(), next_video - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start
    print(f"\nDone. Sent in {elapsed:.1f}s:")
    print(f"  Audio: {audio_seq} chunks ({audio_bytes / 1024:.0f} KB)")
    print(f"  Video: {video_seq} frames ({video_bytes / 1024:.0f} KB)")

    audio_sock.close()
    video_sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
