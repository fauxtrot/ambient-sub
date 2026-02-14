"""Quick encodec playback test - 10 seconds"""
import subprocess
import sys

print("=" * 60)
print("ENCODEC AUDIO RECONSTRUCTION TEST")
print("=" * 60)
print("\nThis will:")
print("  1. Record 10 seconds of audio from your microphone")
print("  2. Encode to Encodec tokens")
print("  3. Decode back to audio")
print("  4. Play both original and reconstructed for comparison")
print("\n" + "=" * 60)

input("\nPress Enter to start recording (10 seconds)...")

# Run encoding.py with playback
subprocess.run([
    sys.executable,
    "tests/encoding.py",
    "--duration", "10",
    "--bandwidth", "6.0",
    "--playback",
    "--output", "reconstructed.wav",
    "--tokens", "tokens.json"
])

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print("\nFiles created:")
print("  - input.wav (original recording)")
print("  - reconstructed.wav (after encode/decode)")
print("  - tokens.json (discrete token representation)")
print("\nYou can replay anytime with:")
print("  python tests/encoding.py --input input.wav --playback")
