"""
Test script for the provider system with DiartAdapter.

This demonstrates the capability-based provider architecture:
1. Register DiartAdapter for speaker identification
2. Process audio through the router
3. Get M4 tokens with annotations

Run this to verify the foundation is working correctly.
"""

import numpy as np
import torch
from ambient_subconscious.providers import (
    CapabilityRegistry,
    CapabilityRouter,
    M4Token,
)
from ambient_subconscious.providers.adapters import DiartAdapter


def generate_test_audio(duration_sec: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate simple test audio (sine wave)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    # 440 Hz sine wave (A4 note)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


def main():
    print("=" * 60)
    print("Testing Provider System with DiartAdapter")
    print("=" * 60)

    # Step 1: Create registry and router
    print("\n[1] Creating registry and router...")
    registry = CapabilityRegistry()
    router = CapabilityRouter(registry)
    print(f"   Created: {registry}")
    print(f"   Created: {router}")

    # Step 2: Register DiartAdapter
    print("\n[2] Registering DiartAdapter...")
    try:
        diart_adapter = DiartAdapter(device="cuda" if torch.cuda.is_available() else "cpu")

        # Register for its capabilities
        for modality, capability in diart_adapter.get_capabilities():
            registry.register(modality, capability, diart_adapter)
            print(f"   Registered: ({modality}, {capability})")

        print(f"\n   Registry summary:")
        summary = registry.get_registry_summary()
        print(f"   - Total capabilities: {summary['total_capabilities']}")
        print(f"   - Total providers: {summary['total_providers']}")

    except Exception as e:
        print(f"   ⚠️  Error initializing Diart: {e}")
        print("   (This is expected if diart models aren't downloaded yet)")
        print("\n   To fix: Run one of the existing tests that downloads diart models:")
        print("   python tests\\test_diart.py")
        return

    # Step 3: Generate test audio
    print("\n[3] Generating test audio...")
    test_audio = generate_test_audio(duration_sec=2.0)
    print(f"   Generated: {test_audio.shape} samples, {test_audio.dtype}")

    # Step 4: Process through router
    print("\n[4] Processing audio through router...")
    print("   (This may take a moment for first inference...)")

    try:
        # Process speaker_id capability
        token = router.process_capability(
            modality="audio",
            capability="speaker_id",
            input_data=test_audio,
            timeout_sec=30.0
        )

        print(f"\n   ✓ Got M4Token: {token}")
        print(f"\n   Token details:")
        print(f"   - Modality: {token.modality}")
        print(f"   - Source: {token.source}")
        print(f"   - Timestamp: {token.timestamp:.2f}s")
        print(f"   - Duration: {token.duration_ms}ms")
        print(f"   - Annotations: {list(token.annotations.keys())}")

        # Print annotations
        for cap, ann in token.annotations.items():
            print(f"\n   Annotation '{cap}':")
            for key, val in ann.items():
                print(f"      {key}: {val}")

    except Exception as e:
        print(f"   ⚠️  Error processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Test multi-capability processing
    print("\n[5] Testing multi-capability processing...")
    try:
        tokens = router.process_multi_capability(
            capabilities=[("audio", "speaker_id"), ("audio", "is_speech")],
            input_data=test_audio,
            parallel=True
        )

        print(f"   ✓ Got {len(tokens)} tokens")
        for i, token in enumerate(tokens):
            if token:
                print(f"   Token {i+1}: {list(token.annotations.keys())}")
            else:
                print(f"   Token {i+1}: Failed")

    except Exception as e:
        print(f"   ⚠️  Error in multi-capability: {e}")

    # Step 6: Get routing stats
    print("\n[6] Router statistics:")
    stats = router.get_routing_stats()
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"   {key}: {val:.2f}")
        else:
            print(f"   {key}: {val}")

    print("\n" + "=" * 60)
    print("✓ Provider system test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add more adapters (WhisperAdapter, YOLOAdapter)")
    print("2. Implement M4StreamCollector for data collection")
    print("3. Add A/B evaluation for comparing providers")


if __name__ == "__main__":
    main()
