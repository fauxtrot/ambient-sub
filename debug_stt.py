#!/usr/bin/env python3
"""Standalone ZMQ audio → VAD → Whisper debug tool.

Connects to the same ZMQ audio stream as the main pipeline.
Shows live energy levels, VAD state, and Whisper transcriptions.

Usage:
    python debug_stt.py                     # defaults: tcp://*:5555, base, cuda
    python debug_stt.py --port 5555         # custom port
    python debug_stt.py --model small       # bigger Whisper model
    python debug_stt.py --device cpu        # CPU-only
    python debug_stt.py --threshold 0.005   # lower VAD threshold
    python debug_stt.py --save              # save utterances as WAV files
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import zmq


# ── ZMQ protocol (matches capture_receiver/protocol.py) ──────────────

def parse_audio(raw: bytes):
    """Parse ZMQ frame → (header_dict, float32_samples) or None."""
    if len(raw) < 8:
        return None
    header_len = struct.unpack(">I", raw[0:4])[0]
    payload_len = struct.unpack(">I", raw[4:8])[0]
    if len(raw) < 8 + header_len + payload_len:
        return None
    header = json.loads(raw[8 : 8 + header_len])
    if header.get("type") != "audio":
        return None
    payload = raw[8 + header_len : 8 + header_len + payload_len]
    pcm = np.frombuffer(payload, dtype=np.int16)
    samples = pcm.astype(np.float32) / 32768.0
    return header, samples


# ── Simple energy VAD (matches stream/vad_energy.py logic) ───────────

class SimpleVAD:
    def __init__(self, sample_rate=16000, threshold=0.01, ema_alpha=0.3,
                 min_dur=0.5, max_dur=30.0, silence_dur=0.8, frame_ms=25):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.frame_len = int(frame_ms * sample_rate / 1000)
        self.min_samples = int(min_dur * sample_rate)
        self.max_samples = int(max_dur * sample_rate)
        self.silence_frames_needed = int(silence_dur * 1000 / frame_ms)

        self.ema = 0.0
        self.in_utt = False
        self.utt_start = 0
        self.silence_count = 0
        self.utt_audio = []

    def feed(self, samples):
        """Feed samples, yield (event, audio_or_None) tuples."""
        offset = 0
        while offset + self.frame_len <= len(samples):
            frame = samples[offset : offset + self.frame_len]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * rms

            if not self.in_utt:
                if self.ema > self.threshold:
                    self.in_utt = True
                    self.utt_start = 0
                    self.silence_count = 0
                    self.utt_audio = [frame]
                    yield ("start", None, self.ema, rms)
            else:
                self.utt_audio.append(frame)
                total = sum(len(a) for a in self.utt_audio)

                if self.ema < self.threshold:
                    self.silence_count += 1
                else:
                    self.silence_count = 0

                if self.silence_count >= self.silence_frames_needed or total >= self.max_samples:
                    if total >= self.min_samples:
                        audio = np.concatenate(self.utt_audio)
                        yield ("end", audio, self.ema, rms)
                    else:
                        yield ("discard", None, self.ema, rms)
                    self.in_utt = False
                    self.utt_audio = []
                else:
                    yield ("continue", None, self.ema, rms)

            offset += self.frame_len


# ── Energy meter bar ─────────────────────────────────────────────────

def energy_bar(ema, threshold, width=40):
    """Render a text energy meter."""
    # Clamp to [0, 0.1] for display
    level = min(ema / 0.1, 1.0)
    thresh_pos = min(threshold / 0.1, 1.0)
    filled = int(level * width)
    thresh_col = int(thresh_pos * width)
    bar = ""
    for i in range(width):
        if i == thresh_col:
            bar += "|"
        elif i < filled:
            bar += "#"
        else:
            bar += "."
    return bar


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Debug STT: ZMQ audio → VAD → Whisper")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ audio port (default 5555)")
    parser.add_argument("--bind", default="*", help="ZMQ bind address (default *)")
    parser.add_argument("--connect", default=None, help="ZMQ connect address instead of bind (e.g. tcp://192.168.0.220:5555)")
    parser.add_argument("--model", default="base", help="Whisper model (tiny/base/small/medium)")
    parser.add_argument("--device", default="cuda", help="Whisper device (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=0.01, help="VAD energy threshold")
    parser.add_argument("--silence", type=float, default=0.8, help="Silence duration to end utterance (s)")
    parser.add_argument("--save", action="store_true", help="Save utterance WAVs to debug_audio/")
    parser.add_argument("--no-whisper", action="store_true", help="VAD only, skip Whisper")
    args = parser.parse_args()

    # ── ZMQ setup ──
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVTIMEO, 1000)

    if args.connect:
        sock.connect(args.connect)
        print(f"[ZMQ] Connected to {args.connect}")
    else:
        addr = f"tcp://{args.bind}:{args.port}"
        sock.bind(addr)
        print(f"[ZMQ] Bound to {addr}")

    # ── Whisper setup ──
    whisper_model = None
    if not args.no_whisper:
        print(f"[Whisper] Loading model '{args.model}' on {args.device}...")
        import whisper
        whisper_model = whisper.load_model(args.model, device=args.device)
        print(f"[Whisper] Model loaded.")

    # ── VAD setup ──
    vad = SimpleVAD(
        threshold=args.threshold,
        silence_dur=args.silence,
    )
    print(f"[VAD] threshold={args.threshold}, silence={args.silence}s")

    # ── Save dir ──
    if args.save:
        Path("debug_audio").mkdir(exist_ok=True)
        print("[Save] Utterances will be saved to debug_audio/")

    print()
    print("=" * 60)
    print("  Listening for ZMQ audio... speak into the mic")
    print(f"  Energy bar: # = level, | = threshold ({args.threshold})")
    print("=" * 60)
    print()

    chunk_count = 0
    utt_count = 0
    last_meter = 0

    try:
        while True:
            try:
                raw = sock.recv()
            except zmq.Again:
                continue

            result = parse_audio(raw)
            if result is None:
                continue

            header, samples = result
            chunk_count += 1

            # Feed to VAD
            for event, audio, ema, rms in vad.feed(samples):
                now = time.time()

                if event == "start":
                    sys.stdout.write(f"\r\033[K  >> Utterance started (ema={ema:.4f})\n")
                    sys.stdout.flush()

                elif event == "end":
                    duration = len(audio) / 16000
                    utt_count += 1
                    sys.stdout.write(f"\r\033[K  >> Utterance #{utt_count}: {duration:.1f}s, {len(audio)} samples\n")
                    sys.stdout.flush()

                    # Save WAV
                    if args.save and audio is not None:
                        import scipy.io.wavfile as wav
                        fname = f"debug_audio/utt_{utt_count:04d}_{duration:.1f}s.wav"
                        wav.write(fname, 16000, (audio * 32768).astype(np.int16))
                        print(f"     Saved: {fname}")

                    # Whisper
                    if whisper_model is not None and audio is not None:
                        print(f"     Transcribing ({len(audio)/16000:.1f}s)...", end="", flush=True)
                        t0 = time.time()
                        result = whisper_model.transcribe(audio, language="en", fp16=False)
                        elapsed = time.time() - t0
                        text = result.get("text", "").strip()
                        print(f"\r\033[K     STT ({elapsed:.1f}s): \"{text}\"")

                        # Also show segments for debug
                        for seg in result.get("segments", []):
                            print(f"          [{seg['start']:.1f}-{seg['end']:.1f}] "
                                  f"(p={seg['no_speech_prob']:.2f}) {seg['text'].strip()}")
                    print()

                elif event == "discard":
                    sys.stdout.write(f"\r\033[K  >> Utterance too short, discarded\n")
                    sys.stdout.flush()

            # Live energy meter (throttled to ~4 Hz)
            now = time.time()
            if now - last_meter > 0.25:
                bar = energy_bar(vad.ema, args.threshold)
                state = "SPEECH" if vad.in_utt else "quiet"
                sys.stdout.write(f"\r  [{bar}] ema={vad.ema:.4f} {state}  chunks={chunk_count}")
                sys.stdout.flush()
                last_meter = now

    except KeyboardInterrupt:
        print(f"\n\nDone. {chunk_count} chunks received, {utt_count} utterances detected.")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
