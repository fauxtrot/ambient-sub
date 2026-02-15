"""MI25 VRAM Budget — load all ambient pipeline models on GPU 0 and report usage."""
import torch
import time

print("=== MI25 VRAM Budget Test ===\n")

def report(label, device=0):
    alloc = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"  [{label:30s}]  Alloc: {alloc:.2f} GB  Reserved: {reserved:.2f} GB  / {total:.0f} GB")

device = "cuda:0"
report("Baseline (empty)")

# 1. Whisper base
print("\n--- Loading Whisper base ---")
t0 = time.time()
import whisper
whisper_model = whisper.load_model("base", device=device)
print(f"  Loaded in {time.time()-t0:.1f}s")
report("+ Whisper base")

# 2. YOLO v8n
print("\n--- Loading YOLOv8n ---")
t0 = time.time()
from ultralytics import YOLO
import numpy as np
yolo_model = YOLO("yolov8n.pt")
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
yolo_model(dummy, device=0, verbose=False)
print(f"  Loaded in {time.time()-t0:.1f}s")
report("+ Whisper + YOLO")

# 3. Pyannote speaker diarization
print("\n--- Loading Pyannote speaker-diarization-3.1 ---")
t0 = time.time()
try:
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    # Patch hf_hub_download to accept use_auth_token (pyannote compat)
    import huggingface_hub
    _orig_dl = huggingface_hub.hf_hub_download
    def _patched_dl(*a, **kw):
        kw.pop("use_auth_token", None)
        return _orig_dl(*a, **kw)
    huggingface_hub.hf_hub_download = _patched_dl
    # Also patch the one imported in pyannote
    import pyannote.audio.core.pipeline as _pac
    _pac.hf_hub_download = _patched_dl

    from pyannote.audio import Pipeline
    from huggingface_hub import get_token
    hf_token = get_token()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    pipeline.to(torch.device(device))
    torch.load = _orig
    print(f"  Loaded in {time.time()-t0:.1f}s")
    report("+ Whisper + YOLO + Pyannote")
except Exception as e:
    torch.load = _orig
    print(f"  SKIP: pyannote load failed ({e})")
    report("+ Whisper + YOLO (pyannote skipped)")

# 4. Florence-2
print("\n--- Loading Florence-2-base ---")
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoProcessor
florence_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True, attn_implementation="eager"
).to(device)
florence_processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True
)
print(f"  Loaded in {time.time()-t0:.1f}s")
report("+ All models + Florence-2-base")

# Florence-2 inference benchmark
print("\n--- Florence-2 Inference Benchmark ---")
from PIL import Image
# Create a test image
test_img = Image.new("RGB", (640, 480), color=(120, 150, 180))
tasks = ["<CAPTION>", "<DETAILED_CAPTION>", "<OD>"]
for task in tasks:
    inputs = florence_processor(text=task, images=test_img, return_tensors="pt").to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output_ids = florence_model.generate(
            **inputs, max_new_tokens=256, do_sample=False
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    text = florence_processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    result = florence_processor.post_process_generation(text, task=task, image_size=test_img.size)
    print(f"  {task:25s}  {elapsed:.2f}s  →  {str(result)[:100]}")
report("After Florence-2 inference")

# Summary
print("\n=== SUMMARY ===")
alloc, reserved, total = (
    torch.cuda.memory_allocated(0) / 1024**3,
    torch.cuda.memory_reserved(0) / 1024**3,
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
)
free = total - reserved
print(f"  GPU 0 Used:     {reserved:.2f} GB / {total:.0f} GB")
print(f"  GPU 0 Free:     {free:.2f} GB")
print(f"  GPU 1:          Untouched (reserved for LLM via llama.cpp)")
print(f"  Headroom:       {free:.1f} GB for KV cache / batch inference")
