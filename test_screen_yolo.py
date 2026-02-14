"""Quick test of YOLO26n with screen capture"""

import numpy as np
import time
from PIL import ImageGrab
from ultralytics import YOLO
import torch
import cv2

def test_screen_yolo():
    """Capture screen and run YOLO detection"""

    print("="*60)
    print("Screen Capture YOLO Test")
    print("="*60)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLO model
    print("Loading YOLO26n...")
    model = YOLO('.models/yolo26n.pt')
    model.to(device)
    print("[OK] Model loaded")

    # Capture screenshot
    print("\nCapturing screenshot...")
    start = time.time()
    screenshot = ImageGrab.grab()
    capture_time = (time.time() - start) * 1000

    print(f"[OK] Screenshot captured in {capture_time:.1f}ms")
    print(f"[OK] Resolution: {screenshot.size}")

    # Convert to numpy array
    frame = np.array(screenshot)
    frame_rgb = frame  # PIL uses RGB by default

    # Run YOLO
    print("\nRunning YOLO detection...")
    start = time.time()
    results = model(frame_rgb, verbose=False)
    inference_time = (time.time() - start) * 1000

    print(f"[OK] Inference completed in {inference_time:.1f}ms")

    # Extract detections
    boxes = results[0].boxes
    print(f"\n[OK] Detected {len(boxes)} objects")

    if len(boxes) > 0:
        print("\nDetections:")
        print("-" * 60)

        # Count objects by class
        object_counts = {}

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            object_counts[class_name] = object_counts.get(class_name, 0) + 1

            print(f"{i+1:2d}. {class_name:15s} - confidence: {conf:.3f}")
            print(f"    bbox: [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")

        # Draw boxes on frame
        print("\nDrawing bounding boxes...")
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            # Draw box
            cv2.rectangle(frame_bgr, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 3)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_y = max(xyxy[1] - 10, label_size[1] + 10)

            cv2.rectangle(
                frame_bgr,
                (xyxy[0], label_y - label_size[1] - 10),
                (xyxy[0] + label_size[0] + 10, label_y),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                frame_bgr,
                label,
                (xyxy[0] + 5, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

        # Save annotated image
        output_path = 'screen_yolo_test.jpg'
        cv2.imwrite(output_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"\n[OK] Saved annotated image: {output_path}")

        print("\n" + "="*60)
        print("Screen Analysis Summary")
        print("="*60)
        print(f"\nYOLO detected these objects on your screen:")
        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {count}x {obj}")

    else:
        print("\n[INFO] No objects detected on screen")
        print("This might happen if:")
        print("  - Mostly text/UI elements (YOLO trained on real-world objects)")
        print("  - Desktop/blank screen")
        print("  - Very high-level view with small objects")

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


if __name__ == "__main__":
    test_screen_yolo()
