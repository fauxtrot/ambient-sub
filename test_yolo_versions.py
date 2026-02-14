"""Test YOLO version availability and performance"""

import torch
from ultralytics import YOLO
import numpy as np
import time
from PIL import ImageGrab

def test_yolo_model(model_name):
    """Test loading and inference with a specific YOLO model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print('='*60)

    try:
        # Load model
        print(f"Loading {model_name}...")
        start = time.time()
        model = YOLO(f'.models/{model_name}.pt')
        load_time = time.time() - start
        print(f"[OK] Model loaded in {load_time*1000:.1f}ms")

        # Check device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        print(f"[OK] Using device: {device}")

        # Capture screenshot for testing
        print("Capturing screenshot for inference test...")
        screenshot = ImageGrab.grab()
        img_array = np.array(screenshot)

        # Run inference
        print(f"Running inference on {screenshot.size} image...")
        start = time.time()
        results = model(img_array, verbose=False)
        inference_time = time.time() - start
        print(f"[OK] Inference completed in {inference_time*1000:.1f}ms")

        # Show detections
        detections = results[0].boxes
        print(f"[OK] Detected {len(detections)} objects")

        if len(detections) > 0:
            print("\nTop 5 detections:")
            for i, box in enumerate(detections[:5]):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                print(f"  {i+1}. {model.names[cls_id]}: {conf:.2f} at [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")

        return True, inference_time * 1000  # Success, inference time in ms

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False, None


if __name__ == "__main__":
    print("YOLO Version Testing")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Test models in priority order
    models_to_test = [
        'yolo26n',  # Latest nano
        'yolo11n',  # Fallback 1
        'yolo12n',  # Fallback 2
        'yolov8n',  # Original test model
    ]

    results = {}
    for model_name in models_to_test:
        success, inf_time = test_yolo_model(model_name)
        results[model_name] = {'success': success, 'inference_ms': inf_time}

        if success:
            print(f"\n[OK] {model_name} is available and working!")
            break

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Find best available model
    for model_name, result in results.items():
        if result['success']:
            print(f"\n[RECOMMENDED] Use {model_name}")
            print(f"  Inference time: {result['inference_ms']:.1f}ms")
            print(f"  Model file: .models/{model_name}.pt")
            break
    else:
        print("\n[ERROR] No YOLO models could be loaded!")
