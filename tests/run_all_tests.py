"""Run all component tests"""

import sys
import subprocess


def run_test(test_name, test_file):
    """Run a single test script"""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"\nWARNING: {test_name} timed out")
        return False
    except Exception as e:
        print(f"\nERROR in {test_name}: {e}")
        return False


def main():
    """Run all tests in sequence"""
    tests = [
        ("CUDA/GPU Detection", "tests/test_cuda.py"),
        ("Audio Capture", "tests/test_audio_capture.py"),
        ("Screenshot Capture", "tests/test_screenshot.py"),
        ("YOLOv8 Loading", "tests/test_yolo.py"),
        ("Whisper STT", "tests/test_whisper.py"),
        # ("Diart Streaming", "tests/test_diart.py"),  # Commented out - interactive
        # ("Stream State", "tests/test_stream_state.py"),  # Commented out - interactive
    ]

    print("Ambient Subconscious - Component Tests")
    print("="*60)

    results = []
    for test_name, test_file in tests:
        success = run_test(test_name, test_file)
        results.append((test_name, success))

    # Summary
    print(f"\n\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {test_name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
