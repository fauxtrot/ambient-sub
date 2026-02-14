"""Test YOLOv8 model loading and inference"""

import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw


def test_yolo_loading():
    """Test loading YOLOv8 model"""
    print("Loading YOLOv8 nano model (will download if needed)...")
    import os
    os.makedirs('.models', exist_ok=True)
    model = YOLO('.models/yolov8n.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)

    print(f"Model loaded: {model.model.__class__.__name__}")
    print(f"Number of classes: {len(model.names)}")
    print(f"Sample classes: {list(model.names.values())[:10]}")

    return model


def test_yolo_inference():
    """Test YOLOv8 inference on a dummy image"""
    model = test_yolo_loading()

    # Create a test image
    print("\nCreating test image (640x480 RGB)...")
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("Running inference...")
    results = model(test_image, verbose=False)

    print(f"Inference complete. Detections: {len(results[0].boxes)}")

    if len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes[:5]):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  Detection {i}: {model.names[cls_id]} ({conf:.2f})")
    else:
        print("  (No objects detected in random noise, as expected)")

    # Test feature extraction
    print("\nExtracting features from backbone...")
    # Access the model before the detection head
    with torch.no_grad():
        # Get intermediate features
        features = model.model.model[:10](torch.tensor(test_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0)
    print(f"Feature shape: {features.shape}")
    print("\nYOLOv8 working!")

    return model


if __name__ == "__main__":
    test_yolo_inference()
