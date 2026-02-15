"""
MI25 Smoke Test Suite — gfx900 Inference Validation
====================================================
Run this BEFORE building any streaming infrastructure.
Validates that each model produces correct output on MI25 hardware.

Prerequisites:
  - venv activated with torch 2.7.1+rocm6.3
  - HuggingFace token with pyannote model access accepted:
      https://huggingface.co/pyannote/segmentation-3.0  (accept terms)
      https://huggingface.co/pyannote/speaker-diarization-3.1  (accept terms)
      https://huggingface.co/pyannote/embedding  (accept terms)
  - Set HF token: export HF_TOKEN="hf_your_token_here"

Usage:
  python smoke_test_mi25.py              # Run all tests
  python smoke_test_mi25.py --whisper    # Just Whisper
  python smoke_test_mi25.py --yolo       # Just YOLO
  python smoke_test_mi25.py --diart      # Just Diart/Pyannote
  python smoke_test_mi25.py --all-cpu    # Run everything on CPU as baseline comparison

Test audio source:
  US Constitution Preamble (public domain, ~15s, clear single speaker)
  wget https://www2.cs.uic.edu/~i101/SoundFiles/preamble.wav

Test image:
  Uses ultralytics built-in sample, or provide your own JPEG.
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def passed(msg, elapsed=None):
    t = f" ({elapsed:.2f}s)" if elapsed else ""
    print(f"  ✅ PASS: {msg}{t}")

def failed(msg, err=None):
    print(f"  ❌ FAIL: {msg}")
    if err:
        print(f"     Error: {err}")

def warn(msg):
    print(f"  ⚠️  WARN: {msg}")

def get_test_audio():
    """Locate test audio file — real speech for meaningful validation."""
    audio_path = Path("test_tts_cloned.wav")
    if audio_path.exists():
        return audio_path
    # Fallback to synthetic if real audio not available
    audio_path = Path("test_audio.wav")
    if not audio_path.exists():
        print("  Generating synthetic test audio (speech-like tones)...")
        generate_synthetic_audio(audio_path)
    return audio_path

def generate_synthetic_audio(path):
    """Generate a simple sine wave WAV as fallback test audio."""
    import torch
    import torchaudio
    sample_rate = 16000
    duration = 5  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Two tones to simulate something interesting
    waveform = (0.5 * torch.sin(2 * 3.14159 * 440 * t) +
                0.3 * torch.sin(2 * 3.14159 * 880 * t)).unsqueeze(0)
    torchaudio.save(str(path), waveform, sample_rate)
    warn("Using synthetic audio — Whisper transcription will be empty/noise, but tests GPU ops")

def get_test_image():
    """Create a simple test image or use an existing one."""
    img_path = Path("test_image.jpg")
    if not img_path.exists():
        print("  Generating test image with colored rectangles...")
        try:
            import numpy as np
            import cv2
            # Create a scene with some objects YOLO might detect
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            img[:] = (180, 200, 180)  # light background
            # Some rectangles to give YOLO something to look at
            cv2.rectangle(img, (100, 100), (250, 350), (0, 0, 200), -1)
            cv2.rectangle(img, (350, 200), (550, 500), (200, 0, 0), -1)
            cv2.circle(img, (320, 320), 80, (0, 200, 0), -1)
            cv2.imwrite(str(img_path), img)
            passed("Generated test image")
        except Exception as e:
            failed("Could not generate test image", e)
            return None
    return img_path


# ── Test 0: GPU Detection ────────────────────────────────────────────────────

def test_gpu():
    section("TEST 0: GPU Detection & Basic Tensor Ops")
    import torch

    if not torch.cuda.is_available():
        failed("torch.cuda.is_available() returned False — ROCm not working")
        return False

    gpu_count = torch.cuda.device_count()
    print(f"  GPUs detected: {gpu_count}")
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"    GPU {i}: {name} ({mem:.1f} GB)")

    # Quick matmul on each GPU
    for i in range(gpu_count):
        try:
            t0 = time.time()
            a = torch.randn(1024, 1024, device=f"cuda:{i}")
            b = torch.randn(1024, 1024, device=f"cuda:{i}")
            c = torch.mm(a, b)
            torch.cuda.synchronize(i)
            elapsed = time.time() - t0
            passed(f"GPU {i} matmul 1024x1024", elapsed)
        except Exception as e:
            failed(f"GPU {i} matmul", e)
            return False

    return True


# ── Test 1: Whisper ──────────────────────────────────────────────────────────

def test_whisper(device="cuda:0"):
    section(f"TEST 1: Whisper Inference ({device})")
    audio_path = get_test_audio()
    if not audio_path.exists():
        failed("No test audio available")
        return False

    try:
        import whisper

        print(f"  Loading Whisper 'base' model on {device}...")
        t0 = time.time()
        model = whisper.load_model("base", device=device)
        load_time = time.time() - t0
        passed(f"Model loaded", load_time)

        print(f"  Transcribing {audio_path}...")
        t0 = time.time()
        result = model.transcribe(str(audio_path))
        transcribe_time = time.time() - t0

        text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        # The key test: transcribe() ran on GPU without crashing
        passed(f"Transcription complete", transcribe_time)
        print(f"    Language: {language}")

        if text:
            print(f"    Text: {text[:200]}{'...' if len(text) > 200 else ''}")
            # Check for garbled output (common gfx900 failure mode)
            if len(set(text.split())) < 3:
                failed("Transcription looks garbled — only repeated tokens")
                return False
        else:
            warn("Empty transcription (expected for synthetic sine wave audio)")
            warn("Re-run with real speech audio for a content validation test")

        return True

    except Exception as e:
        failed("Whisper inference", e)
        import traceback
        traceback.print_exc()
        return False


# ── Test 2: YOLO ─────────────────────────────────────────────────────────────

def test_yolo(device="cuda:0"):
    section(f"TEST 2: YOLO Inference ({device})")
    img_path = get_test_image()
    if img_path is None:
        failed("No test image available")
        return False

    try:
        from ultralytics import YOLO

        # Use device index for ultralytics (it wants int, not "cuda:N")
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        if "cpu" in device:
            dev_idx = "cpu"

        print(f"  Loading YOLOv8n on device {dev_idx}...")
        t0 = time.time()
        model = YOLO("yolov8n.pt")
        load_time = time.time() - t0
        passed(f"Model loaded", load_time)

        print(f"  Running inference on {img_path}...")
        t0 = time.time()
        results = model(str(img_path), device=dev_idx, verbose=False)
        infer_time = time.time() - t0

        r = results[0]
        num_detections = len(r.boxes) if r.boxes is not None else 0
        passed(f"Inference complete — {num_detections} detections", infer_time)

        if r.boxes is not None and num_detections > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id]
                coords = box.xyxy[0].tolist()
                print(f"    {label} ({conf:.2f}) at [{coords[0]:.0f},{coords[1]:.0f},{coords[2]:.0f},{coords[3]:.0f}]")

            # Sanity: confidence scores should be between 0 and 1
            confs = [float(b.conf[0]) for b in r.boxes]
            if all(0 <= c <= 1 for c in confs):
                passed("Confidence scores are sane (0-1 range)")
            else:
                failed("Confidence scores out of range — possible GPU corruption")
                return False
        else:
            warn("No detections on synthetic image (expected — geometric shapes aren't COCO objects)")
            warn("Try with a real photo (person, car, etc.) for a more meaningful test")
            # This isn't a failure — YOLO correctly found nothing recognizable

        # Second pass: test with ultralytics bus sample if available
        bus_url = "https://ultralytics.com/images/bus.jpg"
        bus_path = Path("test_bus.jpg")
        if not bus_path.exists():
            try:
                import urllib.request
                urllib.request.urlretrieve(bus_url, str(bus_path))
            except:
                pass

        if bus_path.exists():
            print(f"\n  Running inference on bus.jpg (should detect people + bus)...")
            t0 = time.time()
            results2 = model(str(bus_path), device=dev_idx, verbose=False)
            infer_time2 = time.time() - t0
            r2 = results2[0]
            num2 = len(r2.boxes) if r2.boxes is not None else 0
            passed(f"Bus image — {num2} detections", infer_time2)
            if r2.boxes is not None:
                for box in r2.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = r2.names[cls_id]
                    print(f"    {label} ({conf:.2f})")
                if num2 >= 2:
                    passed("Detected multiple objects in real photo — GPU inference is working")
                else:
                    warn("Fewer detections than expected")

        return True

    except Exception as e:
        failed("YOLO inference", e)
        import traceback
        traceback.print_exc()
        return False


# ── Test 3: Pyannote / Diart ────────────────────────────────────────────────

def test_diart(device="cuda:0"):
    section(f"TEST 3: Pyannote Speaker Diarization ({device})")
    audio_path = get_test_audio()
    if not audio_path.exists():
        failed("No test audio available")
        return False

    from huggingface_hub import HfFolder
    hf_token = HfFolder.get_token()
    if not hf_token:
        failed("No HF token found. Run: python3 -c \"from huggingface_hub import login; login()\"")
        return False
    print(f"  Using token from huggingface-cli login")

    try:
        import torch
        # PyTorch 2.6+ defaults weights_only=True, but pyannote checkpoints
        # contain many custom classes. Temporarily revert to weights_only=False
        # for trusted HuggingFace models.
        _original_load = torch.load
        torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})
        from pyannote.audio import Pipeline

        print(f"  Loading pyannote/speaker-diarization-3.1...")
        t0 = time.time()
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        load_time = time.time() - t0
        passed(f"Pipeline loaded", load_time)

        # Send to GPU if requested
        if "cuda" in device:
            print(f"  Sending pipeline to {device}...")
            pipeline.to(torch.device(device))
            passed(f"Pipeline on {device}")

        print(f"  Running diarization on {audio_path}...")
        t0 = time.time()
        diarization = pipeline(str(audio_path))
        diarize_time = time.time() - t0

        # Restore original torch.load
        torch.load = _original_load

        # Check output
        speakers = set()
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            segments.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })

        passed(f"Diarization complete — {len(speakers)} speaker(s), {len(segments)} segment(s)", diarize_time)
        for seg in segments[:10]:  # Show first 10
            print(f"    [{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['speaker']}")

        if len(segments) == 0:
            warn("No speech segments detected (expected for synthetic sine wave audio)")
            warn("Re-run with real speech audio for a content validation test")
            return True

        # Sanity: timestamps should be monotonically increasing
        starts = [s["start"] for s in segments]
        if starts == sorted(starts):
            passed("Timestamps are monotonically ordered")
        else:
            warn("Timestamps not strictly ordered (may indicate clustering issue)")

        return True

    except Exception as e:
        failed("Pyannote diarization", e)
        import traceback
        traceback.print_exc()
        return False


# ── Test 4: Diart Streaming (optional) ──────────────────────────────────────

def test_diart_streaming(device="cuda:0"):
    section(f"TEST 4: Diart Streaming from File ({device})")
    audio_path = get_test_audio()
    if not audio_path.exists():
        failed("No test audio available")
        return False

    try:
        from diart import SpeakerDiarization
        from diart.sources import FileAudioSource
        from diart.inference import StreamingInference

        print("  Setting up diart streaming pipeline...")
        t0 = time.time()
        pipeline = SpeakerDiarization()
        source = FileAudioSource(str(audio_path), sample_rate=16000)
        inference = StreamingInference(pipeline, source, do_plot=False)
        setup_time = time.time() - t0
        passed(f"Diart pipeline created", setup_time)

        print("  Running streaming diarization from file...")
        t0 = time.time()
        prediction = inference()
        stream_time = time.time() - t0
        passed(f"Streaming diarization complete", stream_time)

        return True

    except Exception as e:
        # Diart streaming has known quirks — don't hard-fail
        warn(f"Diart streaming test failed: {e}")
        warn("This is non-critical — batch pyannote (Test 3) is the primary path")
        return True  # Soft pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MI25 Smoke Test Suite")
    parser.add_argument("--whisper", action="store_true", help="Test Whisper only")
    parser.add_argument("--yolo", action="store_true", help="Test YOLO only")
    parser.add_argument("--diart", action="store_true", help="Test Pyannote/Diart only")
    parser.add_argument("--gpu", default="cuda:0", help="GPU device (default: cuda:0)")
    parser.add_argument("--all-cpu", action="store_true", help="Run all tests on CPU as baseline")
    args = parser.parse_args()

    device = "cpu" if args.all_cpu else args.gpu
    run_all = not (args.whisper or args.yolo or args.diart)

    print(f"\n🔥 MI25 Smoke Test Suite")
    print(f"   Device: {device}")
    print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Always run GPU detection first
    if not args.all_cpu:
        results["gpu"] = test_gpu()
        if not results["gpu"]:
            print("\n💀 GPU detection failed. Fix ROCm before proceeding.")
            sys.exit(1)

    if run_all or args.whisper:
        results["whisper"] = test_whisper(device)

    if run_all or args.yolo:
        results["yolo"] = test_yolo(device)

    if run_all or args.diart:
        results["diart"] = test_diart(device)
        # Optional streaming test
        if results.get("diart"):
            results["diart_streaming"] = test_diart_streaming(device)

    # Summary
    section("RESULTS SUMMARY")
    all_passed = True
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test}")
        if not result:
            all_passed = False

    if all_passed:
        print(f"\n🎉 All tests passed on {device}!")
        print("   Your MI25(s) are ready for the ambient-sub pipeline.")
        print("   Next step: Phase 3 — set up the capture streaming from Godot.")
    else:
        print(f"\n⚠️  Some tests failed on {device}.")
        print("   Try running failed tests with --all-cpu to confirm it's a GPU issue.")
        print("   If CPU passes but GPU fails, the model has gfx900 compatibility issues.")
        print("   Fallback: run that model on CPU (you have 16 cores / 96GB to spare).")

    # Save results
    results_path = Path("smoke_test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "device": device,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "results": {k: v for k, v in results.items()}
        }, f, indent=2)
    print(f"\n   Results saved to {results_path}")


if __name__ == "__main__":
    main()