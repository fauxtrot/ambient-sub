"""Quick test of YOLO26n with webcam capture"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

def test_webcam_yolo():
    """Capture from webcam and run YOLO detection"""

    print("="*60)
    print("Webcam YOLO Test")
    print("="*60)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLO model
    print("Loading YOLO26n...")
    model = YOLO('.models/yolo26n.pt')
    model.to(device)
    print("[OK] Model loaded")

    # Open webcam
    print("\nOpening webcam (camera 0)...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        print("Try checking:")
        print("  - Camera permissions")
        print("  - Other apps using camera")
        print("  - Different camera index (try 1, 2, etc.)")
        return

    print("[OK] Webcam opened")

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm up camera
    print("Warming up camera...")
    for _ in range(5):
        cap.read()
    time.sleep(0.5)

    # Capture frame
    print("\nCapturing frame...")
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Could not capture frame!")
        cap.release()
        return

    print(f"[OK] Captured frame: {frame.shape}")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            print(f"{i+1:2d}. {class_name:15s} - confidence: {conf:.3f}")
            print(f"    bbox: [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")

        # Draw boxes on frame
        print("\nDrawing bounding boxes...")
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            # Draw box
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(xyxy[1] - 10, label_size[1] + 10)

            cv2.rectangle(
                frame,
                (xyxy[0], label_y - label_size[1] - 10),
                (xyxy[0] + label_size[0] + 10, label_y),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                frame,
                label,
                (xyxy[0] + 5, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        # Save annotated image
        output_path = 'webcam_yolo_test.jpg'
        cv2.imwrite(output_path, frame)
        print(f"\n[OK] Saved annotated image: {output_path}")

    else:
        print("\n[INFO] No objects detected")
        print("This might happen if:")
        print("  - Room is empty/dark")
        print("  - Objects are out of frame")
        print("  - Confidence threshold too high")

    # Cleanup
    cap.release()

    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

    if len(boxes) > 0:
        print(f"\nYOLO detected objects in your room:")
        object_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

        for obj, count in sorted(object_counts.items()):
            print(f"  - {count}x {obj}")


if __name__ == "__main__":
    test_webcam_yolo()
