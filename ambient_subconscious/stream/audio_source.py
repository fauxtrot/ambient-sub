"""
Audio source with real-time speaker diarization.

This is the foundation of the system - the continuous audio river
that provides the timebase for all other events.
"""

import os
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
import torch
from diart import SpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.inference import StreamingInference
import rx.operators as ops

# Configure model cache
os.environ['HF_HOME'] = os.path.abspath('.models/huggingface')
os.environ['TORCH_HOME'] = os.path.abspath('.models/torch')


@dataclass
class DiarizationEvent:
    """Event emitted by the audio source"""
    timestamp: float  # seconds from stream start
    sample_position: int  # sample index in audio stream
    speaker: Optional[str]  # speaker ID (e.g., "speaker_0")
    event_type: str  # "speaker_change", "speech_start", "speech_end"
    confidence: float  # confidence score
    annotation: object  # full pyannote annotation


class AudioListener:
    """
    Core audio streaming listener with real-time diarization.

    This is the foundation of the system. Audio flows continuously,
    and diarization events are emitted as they occur.
    """

    def __init__(self, device: Optional[str] = None, audio_device: Optional[int] = None):
        """
        Initialize the audio listener.

        Args:
            device: 'cuda' or 'cpu'. Auto-detected if None.
            audio_device: Audio input device index. None = default mic.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.audio_device = audio_device
        self.sample_rate = 16000  # diart default

        # Initialize diarization pipeline
        print(f"Initializing speaker diarization on {device}...")
        self.pipeline = SpeakerDiarization()  # No device parameter in diart 0.9.2

        # State tracking
        self.current_speakers = set()
        self.stream_start_time = None
        self.sample_count = 0

        print("Audio listener ready")

    def _emit_event(
        self,
        annotation,
        event_type: str,
        speaker: Optional[str] = None
    ) -> DiarizationEvent:
        """Create a diarization event from annotation"""
        timestamp = annotation[0].end  # pyannote uses (start, end) tuples
        sample_position = int(timestamp * self.sample_rate)

        # Extract active speakers at this timestamp
        active_speakers = set()
        for segment, _, label in annotation.itertracks(yield_label=True):
            if segment.start <= timestamp <= segment.end:
                active_speakers.add(label)

        # Determine speaker and confidence
        if speaker is None and active_speakers:
            speaker = list(active_speakers)[0]

        confidence = 1.0  # TODO: extract confidence from pyannote

        return DiarizationEvent(
            timestamp=timestamp,
            sample_position=sample_position,
            speaker=speaker,
            event_type=event_type,
            confidence=confidence,
            annotation=annotation
        )

    def listen(
        self,
        on_event: Callable[[DiarizationEvent], None],
        duration: Optional[float] = None,
        on_audio_chunk: Optional[Callable[[np.ndarray], None]] = None
    ):
        """
        Start listening and emit diarization events.

        Args:
            on_event: Callback function for each diarization event
            duration: Duration in seconds (None = infinite)
            on_audio_chunk: Optional callback for raw audio chunks (for saving to file)
        """
        print(f"Starting audio stream (duration={duration})...")

        # Create microphone source with optional device selection
        if self.audio_device is not None:
            print(f"Using audio device: {self.audio_device}")
            mic = MicrophoneAudioSource(self.sample_rate, device=self.audio_device)
        else:
            mic = MicrophoneAudioSource(self.sample_rate)

        # Create streaming inference
        inference = StreamingInference(self.pipeline, mic, do_profile=False, do_plot=False)

        # Track duration
        import time
        start_time = time.time()

        def on_annotation(annotation_wav):
            """Process each annotation from diart (called via hook)"""
            try:
                print(f"[DEBUG] on_annotation called at {time.time() - start_time:.2f}s")
                annotation, waveform = annotation_wav
                print(f"[DEBUG] Annotation type: {type(annotation)}, has {len(list(annotation.itertracks()))} tracks")

                # Pass audio chunk to callback if provided (for saving raw audio)
                if on_audio_chunk is not None:
                    on_audio_chunk(waveform)

                # Check duration
                if duration is not None and time.time() - start_time >= duration:
                    print(f"[DEBUG] Duration limit reached, stopping")
                    return

                # Track speaker changes
                active_speakers = set()
                for segment, _, label in annotation.itertracks(yield_label=True):
                    timestamp = annotation.end
                    print(f"[DEBUG] Track: segment={segment}, label={label}, timestamp={timestamp:.2f}s")
                    if segment.start <= timestamp <= segment.end:
                        active_speakers.add(label)

                print(f"[DEBUG] Active speakers: {active_speakers}, Previous: {self.current_speakers}")

                # Detect speaker changes
                new_speakers = active_speakers - self.current_speakers
                left_speakers = self.current_speakers - active_speakers

                if new_speakers:
                    print(f"[DEBUG] New speakers detected: {new_speakers}")
                if left_speakers:
                    print(f"[DEBUG] Speakers left: {left_speakers}")

                for speaker in new_speakers:
                    event = self._emit_event(annotation, "speaker_change", speaker)
                    print(f"[DEBUG] Emitting speaker_change event for {speaker}")
                    on_event(event)

                for speaker in left_speakers:
                    event = self._emit_event(annotation, "speaker_end", speaker)
                    print(f"[DEBUG] Emitting speaker_end event for {speaker}")
                    on_event(event)

                self.current_speakers = active_speakers

                # Emit general update
                event = self._emit_event(annotation, "update")
                print(f"[DEBUG] Emitting update event")
                on_event(event)

            except Exception as e:
                print(f"[ERROR] Exception in on_annotation: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        # Attach hook to process annotations in real-time
        inference.attach_hooks(on_annotation)

        print("Listening...")

        # Run inference with timeout if duration specified
        if duration is not None:
            import threading

            # Flag to stop inference
            stop_flag = {"stop": False}

            def run_inference():
                try:
                    inference()
                except Exception as e:
                    if not stop_flag["stop"]:
                        print(f"[ERROR] Inference error: {e}")

            # Start inference in thread
            inference_thread = threading.Thread(target=run_inference, daemon=True)
            inference_thread.start()

            # Wait for duration
            inference_thread.join(timeout=duration)

            # Stop if still running
            if inference_thread.is_alive():
                print(f"\nDuration limit ({duration}s) reached, stopping...")
                stop_flag["stop"] = True
                # Force stop the microphone source
                try:
                    mic.close()
                except:
                    pass
        else:
            # Run inference indefinitely (blocks until complete)
            try:
                prediction = inference()
            except KeyboardInterrupt:
                print("\nStopped by user")


def main():
    """Test the audio listener"""
    import time

    listener = AudioListener()

    event_count = 0
    speakers_seen = set()

    def handle_event(event: DiarizationEvent):
        nonlocal event_count, speakers_seen
        event_count += 1

        if event.speaker:
            speakers_seen.add(event.speaker)

        if event_count % 10 == 0:
            print(f"[{event.timestamp:6.2f}s] {event.event_type:15s} speaker={event.speaker} (total_events={event_count})")

    print("\nSpeak into your microphone to test speaker detection...")
    print("Press Ctrl+C to stop\n")

    try:
        listener.listen(on_event=handle_event, duration=30)
        time.sleep(31)  # Wait for stream to complete
    except KeyboardInterrupt:
        print("\nStopped by user")

    print(f"\nFinal stats:")
    print(f"  Total events: {event_count}")
    print(f"  Speakers detected: {len(speakers_seen)}")
    print(f"  Speaker IDs: {speakers_seen}")


if __name__ == "__main__":
    main()
