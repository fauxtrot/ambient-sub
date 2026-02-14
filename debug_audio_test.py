"""Debug test to diagnose audio capture issues"""
from ambient_subconscious.stream import AudioListener
import time

print("=" * 60)
print("AUDIO DEBUG TEST")
print("=" * 60)

listener = AudioListener()
print(f"\nDevice: {listener.device}")
print(f"Audio device: {listener.audio_device}")
print(f"Sample rate: {listener.sample_rate}")

callback_count = [0]
event_count = [0]

def handle_event(event):
    """Simple event handler with counter"""
    event_count[0] += 1
    print(f"\n>>> EVENT #{event_count[0]}: {event.event_type} | speaker={event.speaker} | time={event.timestamp:.2f}s")

print("\n" + "=" * 60)
print("Starting 10-second test...")
print("SPEAK INTO YOUR MICROPHONE NOW!")
print("=" * 60 + "\n")

try:
    listener.listen(on_event=handle_event, duration=10)
    time.sleep(11)
except KeyboardInterrupt:
    print("\nStopped by user")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print(f"Total events captured: {event_count[0]}")
print("\nIf you see [DEBUG] messages above, the callback is working.")
print("If events = 0, check if [DEBUG] messages appeared.")
print("If no [DEBUG] messages, the callback was never triggered (audio/device issue).")
