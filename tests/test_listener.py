"""Test the core audio listener"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

from ambient_subconscious.stream.audio_source import AudioListener, DiarizationEvent


def test_listener(duration=10):
    """Test the audio listener for a short duration"""
    print("Testing Audio Listener")
    print("="*60)

    listener = AudioListener()

    event_count = 0
    speakers_seen = set()
    events_by_type = {}

    def handle_event(event: DiarizationEvent):
        nonlocal event_count, speakers_seen
        event_count += 1

        if event.speaker:
            speakers_seen.add(event.speaker)

        events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1

        if event_count <= 5 or event.event_type == "speaker_change":
            print(f"[{event.timestamp:6.2f}s] {event.event_type:15s} speaker={event.speaker}")

    print(f"\nListening for {duration} seconds...")
    print("Speak into your microphone!\n")

    try:
        listener.listen(on_event=handle_event, duration=duration)
        import time
        time.sleep(duration + 1)
    except KeyboardInterrupt:
        print("\nStopped by user")

    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}")
    print(f"Total events: {event_count}")
    print(f"Speakers detected: {len(speakers_seen)}")
    print(f"Speaker IDs: {speakers_seen}")
    print(f"\nEvents by type:")
    for event_type, count in sorted(events_by_type.items()):
        print(f"  {event_type:15s}: {count}")

    if event_count > 0:
        print("\nListener working!")
    else:
        print("\nWARNING: No events received. Check microphone permissions.")


if __name__ == "__main__":
    test_listener()
