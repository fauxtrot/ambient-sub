"""Async HTTP client for the Redot stage server API."""
import asyncio
import base64
import logging
import uuid
from typing import Optional

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


class StageClient:
    """Async client wrapping the stage server at 127.0.0.1:7860."""

    # Godot project default mix rate — resample to this before sending
    STAGE_MIX_RATE = 44100

    def __init__(self, host: str = "127.0.0.1", port: int = 7860):
        self.base_url = f"http://{host}:{port}"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def health(self) -> bool:
        """Check if stage server is reachable (GET /health)."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def get_reactions(self) -> list[str]:
        """GET /reactions - list available expression presets."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/reactions") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Handle both list and dict responses
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict):
                        return data.get("reactions", [])
                return []
        except Exception as e:
            logger.warning(f"Failed to get reactions: {e}")
            return []

    async def get_status(self) -> dict:
        """GET /status - current server state."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/status") as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}
        except Exception as e:
            logger.warning(f"Failed to get status: {e}")
            return {}

    async def react(self, name: str, instant: bool = False):
        """POST /react - trigger an expression preset."""
        try:
            session = await self._get_session()
            payload = {"name": name}
            if instant:
                payload["instant"] = True
            async with session.post(f"{self.base_url}/react", json=payload) as resp:
                if resp.status == 200:
                    logger.debug(f"Reacted: {name}")
                else:
                    text = await resp.text()
                    logger.warning(f"React failed ({resp.status}): {text}")
        except Exception as e:
            logger.warning(f"Failed to react '{name}': {e}")

    async def reset_reaction(self):
        """POST /react/reset - reset expression to neutral."""
        try:
            session = await self._get_session()
            async with session.post(f"{self.base_url}/react/reset") as resp:
                if resp.status == 200:
                    logger.debug("Reaction reset")
        except Exception as e:
            logger.warning(f"Failed to reset reaction: {e}")

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        try:
            import scipy.signal as signal
            target_length = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, target_length).astype(np.float32)
        except ImportError:
            ratio = target_sr / orig_sr
            target_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    async def speak_chunked(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
        chunk_seconds: float = 0.5,
        speak_id: Optional[str] = None,
        callback_url: Optional[str] = None,
    ) -> str:
        """
        Stream audio to the stage server in chunks via POST /speak.

        Resamples to STAGE_MIX_RATE (44100) to match the Godot audio bus,
        then chunks and streams. First chunk includes sample_rate, last
        chunk includes done: true.

        Args:
            audio: float32 mono PCM numpy array
            sample_rate: Source sample rate
            chunk_seconds: Chunk duration for streaming
            speak_id: Unique utterance ID (auto-generated if None)
            callback_url: URL the stage server POSTs to when playback finishes

        Returns:
            The speak_id for this utterance.
        """
        if speak_id is None:
            speak_id = f"utt_{uuid.uuid4().hex[:8]}"

        # Resample to match Godot's audio bus rate
        if sample_rate != self.STAGE_MIX_RATE:
            audio = self._resample(audio, sample_rate, self.STAGE_MIX_RATE)
            logger.debug(f"Resampled {sample_rate} -> {self.STAGE_MIX_RATE} Hz")
            sample_rate = self.STAGE_MIX_RATE

        chunk_size = int(sample_rate * chunk_seconds)
        total_samples = len(audio)
        num_chunks = max(1, (total_samples + chunk_size - 1) // chunk_size)

        session = await self._get_session()

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_samples)
            chunk = audio[start:end]

            # Encode float32 PCM as base64
            audio_b64 = base64.b64encode(chunk.astype(np.float32).tobytes()).decode()

            payload = {"audio": audio_b64}

            # First chunk: include sample rate, id, and callback_url
            if i == 0:
                payload["sample_rate"] = sample_rate
                payload["id"] = speak_id
                if callback_url:
                    payload["callback_url"] = callback_url

            # Last chunk: signal done
            if i == num_chunks - 1:
                payload["done"] = True

            try:
                async with session.post(
                    f"{self.base_url}/speak", json=payload
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(f"Speak chunk {i+1}/{num_chunks} failed ({resp.status}): {text}")
                        return speak_id
            except Exception as e:
                logger.warning(f"Speak chunk {i+1}/{num_chunks} error: {e}")
                return speak_id

            # Small delay between chunks to avoid overwhelming the server
            if i < num_chunks - 1:
                await asyncio.sleep(0.05)

        logger.debug(f"Sent {num_chunks} audio chunks ({total_samples} samples, id={speak_id})")
        return speak_id

    async def speak_status(self) -> dict:
        """GET /speak/status — current speech playback state."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/speak/status") as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}
        except Exception as e:
            logger.warning(f"Failed to get speak status: {e}")
            return {}

    async def stop_speaking(self):
        """POST /speak/stop - interrupt current speech."""
        try:
            session = await self._get_session()
            async with session.post(f"{self.base_url}/speak/stop") as resp:
                if resp.status == 200:
                    logger.debug("Speech stopped")
        except Exception as e:
            logger.warning(f"Failed to stop speaking: {e}")

    async def mirror(self) -> Optional[bytes]:
        """GET /mirror — capture viewport screenshot as JPEG bytes."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/mirror") as resp:
                if resp.status == 200:
                    data = await resp.read()
                    logger.debug(f"Mirror captured: {len(data)} bytes")
                    return data
                else:
                    logger.warning(f"Mirror failed ({resp.status})")
                return None
        except Exception as e:
            logger.warning(f"Failed to get mirror: {e}")
            return None
