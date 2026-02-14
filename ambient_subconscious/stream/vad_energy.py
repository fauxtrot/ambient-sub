"""
Energy-based Voice Activity Detection (VAD) with RMS/EMA smoothing.

Detects utterance boundaries based on audio energy levels using:
- RMS (Root Mean Square) for instantaneous energy
- EMA (Exponential Moving Average) for smoothing
"""

import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VADEnergyMonitor:
    """
    Energy-based VAD with RMS/EMA smoothing.

    Detects utterance boundaries based on audio energy levels.
    Uses exponential moving average to smooth energy measurements
    and reduce false positives from transient noise.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: int = 25,
        energy_threshold: float = 0.01,
        ema_alpha: float = 0.3,
        min_utterance_duration: float = 0.5,
        max_utterance_duration: float = 30.0,
        silence_duration: float = 0.8
    ):
        """
        Initialize VAD energy monitor.

        Args:
            sample_rate: Audio sample rate (Hz)
            frame_length_ms: Frame size for RMS calculation (ms)
            energy_threshold: Energy level to trigger utterance (RMS amplitude)
            ema_alpha: EMA smoothing factor (0-1, higher = more reactive to changes)
            min_utterance_duration: Minimum utterance length (seconds)
            max_utterance_duration: Maximum utterance length before force-split (seconds)
            silence_duration: Continuous silence duration to end utterance (seconds)
        """
        self.sample_rate = sample_rate
        self.frame_length = int(frame_length_ms * sample_rate / 1000)
        self.energy_threshold = energy_threshold
        self.ema_alpha = ema_alpha
        self.min_utterance_duration = min_utterance_duration
        self.max_utterance_duration = max_utterance_duration
        self.silence_frames_threshold = int(silence_duration * 1000 / frame_length_ms)

        # State
        self.ema_energy = 0.0
        self.in_utterance = False
        self.utterance_start_sample = 0
        self.silence_frame_count = 0

        logger.info(
            f"VADEnergyMonitor initialized: threshold={energy_threshold}, "
            f"ema_alpha={ema_alpha}, min_duration={min_utterance_duration}s"
        )

    def process_audio_chunk(
        self,
        audio: np.ndarray,
        sample_position: int
    ) -> Dict[str, Any]:
        """
        Process audio chunk and detect utterance boundaries.

        Args:
            audio: Audio samples (1D numpy array)
            sample_position: Current sample position in stream

        Returns:
            Dictionary with:
                - event: 'utterance_start' | 'utterance_end' | 'utterance_continue' | None
                - rms: Instantaneous RMS energy
                - ema_energy: Smoothed energy level
                - sample_position: Current sample position
                - utterance_duration: Duration of current utterance (if in_utterance)
        """
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0.0

        # Update EMA (exponential moving average)
        self.ema_energy = (1 - self.ema_alpha) * self.ema_energy + self.ema_alpha * rms

        event = None
        utterance_duration = 0.0

        if not self.in_utterance:
            # Waiting for speech - check for utterance start
            if self.ema_energy > self.energy_threshold:
                self.in_utterance = True
                self.utterance_start_sample = sample_position
                self.silence_frame_count = 0
                event = 'utterance_start'

                logger.debug(
                    f"Utterance started at sample {sample_position} "
                    f"(energy={self.ema_energy:.4f})"
                )
        else:
            # In utterance - check for end conditions
            utterance_duration = (sample_position - self.utterance_start_sample) / self.sample_rate

            # Track silence
            if self.ema_energy < self.energy_threshold:
                self.silence_frame_count += 1
            else:
                self.silence_frame_count = 0

            # Check end conditions
            silence_exceeded = self.silence_frame_count >= self.silence_frames_threshold
            max_duration_exceeded = utterance_duration >= self.max_utterance_duration

            if silence_exceeded or max_duration_exceeded:
                # End utterance only if meets minimum duration
                if utterance_duration >= self.min_utterance_duration:
                    event = 'utterance_end'

                    reason = "silence" if silence_exceeded else "max_duration"
                    logger.debug(
                        f"Utterance ended at sample {sample_position} "
                        f"(duration={utterance_duration:.2f}s, reason={reason})"
                    )
                else:
                    # Too short - discard
                    logger.debug(
                        f"Utterance too short ({utterance_duration:.2f}s), discarding"
                    )
                    event = 'utterance_discard'

                self.in_utterance = False
                self.silence_frame_count = 0
            else:
                event = 'utterance_continue'

        return {
            'event': event,
            'rms': float(rms),
            'ema_energy': float(self.ema_energy),
            'sample_position': sample_position,
            'utterance_duration': utterance_duration,
            'in_utterance': self.in_utterance
        }

    def reset(self):
        """Reset VAD state (e.g., when starting new session)."""
        self.ema_energy = 0.0
        self.in_utterance = False
        self.utterance_start_sample = 0
        self.silence_frame_count = 0
        logger.info("VADEnergyMonitor reset")
