"""Test screenshot capture"""

import numpy as np
from PIL import ImageGrab
import time


def test_screenshot():
    """Test screenshot capture"""
    print("Capturing screenshot...")
    start = time.time()
    screenshot = ImageGrab.grab()
    elapsed = time.time() - start

    print(f"Screenshot captured in {elapsed*1000:.1f}ms")
    print(f"Size: {screenshot.size}")
    print(f"Mode: {screenshot.mode}")

    # Convert to numpy array
    img_array = np.array(screenshot)
    print(f"Array shape: {img_array.shape}")
    print(f"Memory: {img_array.nbytes / 1024:.1f} KB")

    print("\nScreenshot capture working!")

    return screenshot


if __name__ == "__main__":
    test_screenshot()
