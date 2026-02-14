"""TTS engine wrapper using Kyutai's Pocket TTS model.

Adapted from ambient-listener/python_experiment/models/pocket_tts_wrapper.py.
Same lazy-init pattern, same model — kept in sync but independent.
"""
import logging
import os

import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TTSEngine:
    """Pocket TTS wrapper for the executive agent.

    Produces float32 mono PCM at the configured sample rate (default 24kHz),
    which is exactly what the stage server's /speak endpoint expects.
    """

    PREDEFINED_VOICES = [
        "alba", "marius", "javert", "jean",
        "fantine", "cosette", "eponine", "azelma"
    ]

    DEFAULT_SPEAKER_WAV = "Selestia_fischl.ogg"

    def __init__(
        self,
        voice: str = "fantine",
        speaker_wav: Optional[str] = None,
        sample_rate: int = 24000
    ):
        self.voice_name = voice
        self.speaker_wav = speaker_wav or os.environ.get("POCKET_SPEAKER_WAV") or self.DEFAULT_SPEAKER_WAV
        self._target_sample_rate = sample_rate
        self._model = None
        self._voice_state = None
        self._initialized = False
        self._model_sample_rate = None

    def _ensure_initialized(self):
        """Lazy initialization of model and voice state."""
        if self._initialized:
            return

        try:
            from pocket_tts import TTSModel

            logger.info("Loading Pocket TTS model...")
            self._model = TTSModel.load_model()
            self._model_sample_rate = self._model.sample_rate
            logger.info(f"  Model sample rate: {self._model_sample_rate}")

            # Load voice state
            if self.speaker_wav and Path(self.speaker_wav).exists():
                logger.info(f"  Loading custom voice from: {self.speaker_wav}")
                self._voice_state = self._model.get_state_for_audio_prompt(self.speaker_wav)
                logger.info("  Custom voice loaded successfully")
            else:
                voice_name = self.voice_name if self.voice_name in self.PREDEFINED_VOICES else "alba"
                logger.info(f"  Loading predefined voice: {voice_name}")
                try:
                    self._voice_state = self._model.get_state_for_audio_prompt(voice_name)
                    logger.info(f"  Voice '{voice_name}' loaded successfully")
                except Exception as e:
                    logger.warning(f"  Failed to load voice '{voice_name}': {e}")
                    logger.info("  Trying fallback voice 'alba'...")
                    self._voice_state = self._model.get_state_for_audio_prompt("alba")

            self._initialized = True
            logger.info("Pocket TTS model loaded successfully.")

        except ImportError as e:
            logger.warning(f"pocket-tts not available: {e}")
            logger.warning("TTS will be disabled. Install with: pip install pocket-tts")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to load Pocket TTS model: {e}", exc_info=True)
            self._initialized = True

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize speech from text (blocking — run in a thread).

        Returns:
            float32 mono PCM numpy array, or None if TTS unavailable.
        """
        self._ensure_initialized()

        if self._model is None or self._voice_state is None:
            logger.info(f"[TTS disabled] Would say: {text}")
            return None

        try:
            audio_tensor = self._model.generate_audio(self._voice_state, text)
            audio = audio_tensor.numpy().astype(np.float32).squeeze()

            # Resample if model sample rate differs from target
            if self._model_sample_rate != self._target_sample_rate:
                audio = self._resample(audio, self._model_sample_rate, self._target_sample_rate)

            # Normalize to prevent clipping
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            elif max_val > 0:
                audio = audio / max(max_val, 0.5)

            return audio

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}", exc_info=True)
            return None

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        try:
            import scipy.signal as signal
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            return signal.resample(audio, target_length).astype(np.float32)
        except ImportError:
            # Fallback: linear interpolation
            ratio = target_sr / orig_sr
            target_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @property
    def available(self) -> bool:
        """Check if TTS is available."""
        self._ensure_initialized()
        return self._model is not None and self._voice_state is not None

    @property
    def sample_rate(self) -> int:
        """Return the output sample rate."""
        return self._target_sample_rate
