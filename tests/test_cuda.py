"""Test CUDA availability and GPU detection"""

import torch


def test_cuda():
    """Check if CUDA is available and show GPU info"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

        # Test tensor creation on GPU
        x = torch.randn(100, 100).cuda()
        print(f"\nTest tensor created on GPU: {x.device}")
        print("CUDA working!")
    else:
        print("\nWARNING: CUDA not available. GPU acceleration will not work.")
        print("Check that you have CUDA-enabled PyTorch installed.")


if __name__ == "__main__":
    test_cuda()
