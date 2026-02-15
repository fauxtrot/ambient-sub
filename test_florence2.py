#!/usr/bin/env python3
"""Quick test: ZMQ video frame → Florence2 caption.

Binds to the ZMQ video port, waits for ONE frame from the game client,
saves it as a JPEG, runs Florence2 on it, and prints the result.

Usage:
    python test_florence2.py                          # defaults
    python test_florence2.py --port 5556              # custom port
    python test_florence2.py --device cpu              # CPU inference
    python test_florence2.py --model microsoft/Florence-2-large  # bigger model
    python test_florence2.py --task detailed           # MORE_DETAILED_CAPTION
    python test_florence2.py --save frame.jpg          # custom save path
"""

import argparse
import json
import struct
import sys
import time

import zmq


def parse_video_frame(raw: bytes):
    """Parse ZMQ frame → (header_dict, jpeg_bytes) or None."""
    if len(raw) < 8:
        return None
    header_len = struct.unpack(">I", raw[0:4])[0]
    payload_len = struct.unpack(">I", raw[4:8])[0]
    if len(raw) < 8 + header_len + payload_len:
        return None
    header = json.loads(raw[8 : 8 + header_len])
    if header.get("type") != "video":
        return None
    payload = raw[8 + header_len : 8 + header_len + payload_len]
    return header, payload


def main():
    parser = argparse.ArgumentParser(description="Test Florence2 on a ZMQ video frame")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ video port (default 5556)")
    parser.add_argument("--bind", default="*", help="ZMQ bind address (default *)")
    parser.add_argument("--model", default="microsoft/Florence-2-base", help="Florence2 model name")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--task", choices=["caption", "detailed"], default="caption",
                        help="Caption task (caption = short, detailed = long)")
    parser.add_argument("--save", default="test_frame.jpg", help="Save captured frame to this path")
    args = parser.parse_args()

    # ── Step 1: Load Florence2 ──
    print(f"[1/3] Loading Florence2 '{args.model}' on {args.device}...")
    t0 = time.time()

    from ambient_subconscious.models.florence2 import Florence2Model
    model = Florence2Model(model_name=args.model, device=args.device)
    # Force initialization now so we can time it
    model._ensure_initialized()

    if not model.available:
        print("ERROR: Florence2 failed to load. Check dependencies (pip install einops timm)")
        sys.exit(1)

    load_time = time.time() - t0
    print(f"    Model loaded in {load_time:.1f}s")

    # ── Step 2: Wait for a ZMQ video frame ──
    print(f"\n[2/3] Waiting for video frame on tcp://{args.bind}:{args.port}...")
    print("    (make sure the game client is running and sending video)")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVTIMEO, 30000)  # 30s timeout
    sock.bind(f"tcp://{args.bind}:{args.port}")

    jpeg_bytes = None
    header = None
    try:
        while True:
            try:
                raw = sock.recv()
            except zmq.Again:
                print("    TIMEOUT: No frame received in 30s. Is the game client sending video?")
                sys.exit(1)

            result = parse_video_frame(raw)
            if result is None:
                continue  # skip non-video messages

            header, jpeg_bytes = result
            break
    finally:
        sock.close()
        ctx.term()

    print(f"    Frame received: {header.get('w', '?')}x{header.get('h', '?')}, "
          f"{len(jpeg_bytes)} bytes, seq={header.get('seq', '?')}")

    # Save the frame
    with open(args.save, "wb") as f:
        f.write(jpeg_bytes)
    print(f"    Saved to: {args.save}")

    # ── Step 3: Run Florence2 ──
    task_name = "caption" if args.task == "caption" else "detailed"
    print(f"\n[3/3] Running Florence2 ({task_name})...")

    t0 = time.time()
    if task_name == "caption":
        result = model.caption(jpeg_bytes)
    else:
        result = model.describe(jpeg_bytes)
    inference_time = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  Florence2 ({inference_time:.2f}s): {result}")
    print(f"{'=' * 60}")

    # Also try the other task for comparison
    other = "detailed" if task_name == "caption" else "caption"
    print(f"\n  (also running {other} for comparison...)")
    t0 = time.time()
    if other == "caption":
        result2 = model.caption(jpeg_bytes)
    else:
        result2 = model.describe(jpeg_bytes)
    t2 = time.time() - t0
    print(f"  Florence2 {other} ({t2:.2f}s): {result2}")


if __name__ == "__main__":
    main()
