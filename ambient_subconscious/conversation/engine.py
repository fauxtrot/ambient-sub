"""Conversation engine — hear → understand → respond → speak loop.

Runs parallel to ExecutiveAgent. Taps into the existing AudioReceiver's
rolling buffer, runs its own VAD, transcribes with Whisper, calls the LLM,
parses skill tags, and executes skills (speak, emote).
"""

import asyncio
import logging
import logging.handlers
import os
import re
import socket
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
from aiohttp import web

from ..capture_receiver.audio_receiver import AudioReceiver
from ..executive.stage_client import StageClient
from ..executive.tts import TTSEngine
from ..llm.openai_client import OpenAIClient
from ..stream.vad_energy import VADEnergyMonitor
from .skills import Skill, SkillRegistry

logger = logging.getLogger(__name__)

# --- Dedicated conversation log ---
# Clean, human-readable log showing only the hear/think/speak flow.
conv_log = logging.getLogger("conversation")
conv_log.propagate = False  # Don't leak into the root/system logger


def _setup_conv_log():
    """Set up the conversation log file handler (idempotent)."""
    if conv_log.handlers:
        return
    os.makedirs("logs", exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        "logs/conversation.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    conv_log.addHandler(handler)
    conv_log.setLevel(logging.INFO)


_setup_conv_log()

# How often we poll the AudioReceiver buffer (seconds)
_POLL_INTERVAL = 0.05  # 50ms → 20 Hz


class ConversationEngine:
    """Simple conversation loop: ZMQ audio → VAD → Whisper → LLM → Skills."""

    def __init__(
        self,
        audio_receiver: AudioReceiver,
        config: Dict[str, Any],
        llm_config: Optional[Dict[str, Any]] = None,
        tts_config: Optional[Dict[str, Any]] = None,
        stage_config: Optional[Dict[str, Any]] = None,
        florence2_config: Optional[Dict[str, Any]] = None,
    ):
        self._receiver = audio_receiver
        self._config = config

        # --- VAD ---
        vad_cfg = config.get("vad", {})
        self._vad = VADEnergyMonitor(
            sample_rate=audio_receiver.sample_rate,
            energy_threshold=vad_cfg.get("energy_threshold", 0.01),
            ema_alpha=vad_cfg.get("ema_alpha", 0.3),
            min_utterance_duration=vad_cfg.get("min_utterance_duration", 0.5),
            max_utterance_duration=vad_cfg.get("max_utterance_duration", 30.0),
            silence_duration=vad_cfg.get("silence_duration", 0.8),
        )

        # --- LLM ---
        llm_config = llm_config or config.get("llm", {})
        self._llm = OpenAIClient(
            host=llm_config.get("host", "http://localhost:8080"),
            model=llm_config.get("model", "qwen2.5-7b"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 512),
        )

        # --- TTS ---
        tts_config = tts_config or config.get("tts", {})
        self._tts = TTSEngine(
            voice=tts_config.get("voice", "fantine"),
            speaker_wav=tts_config.get("speaker_wav"),
            sample_rate=tts_config.get("sample_rate", 24000),
        )

        # --- Stage client ---
        stage_config = stage_config or config.get("stage", {})
        self._stage = StageClient(
            host=stage_config.get("host", "127.0.0.1"),
            port=stage_config.get("port", 7860),
        )

        # --- Florence2 (vision-language model for mirror skill) ---
        florence2_config = florence2_config or {}
        self._florence2 = None
        logger.info(f"Florence2 config: enabled={florence2_config.get('enabled', False)}, config={florence2_config}")
        if florence2_config.get("enabled", False):
            from ..models.florence2 import Florence2Model
            self._florence2 = Florence2Model(
                model_name=florence2_config.get("model", "microsoft/Florence-2-base"),
                device=florence2_config.get("device", "cuda"),
            )
            logger.info(f"Florence2 model created (lazy init, will load on first use)")
        else:
            logger.info("Florence2 disabled — mirror skill will not be registered")

        # --- Conversation history ---
        self._system_prompt = config.get(
            "system_prompt",
            "You are Selestia, a friendly and curious AI companion. "
            "Respond naturally and concisely. Use [emote:name] tags to express "
            "emotions and [speak] before text you want to say aloud.",
        )
        self._history: deque = deque(maxlen=config.get("history_max", 20))

        # --- Skill registry ---
        self._skills = SkillRegistry()

        # --- Callback server for speech completion ---
        self._callback_port = config.get("callback_port", 8765)
        self._server_ip = self._detect_lan_ip()
        self._speak_events: Dict[str, asyncio.Event] = {}
        self._callback_app: Optional[web.Application] = None
        self._callback_runner: Optional[web.AppRunner] = None

        # --- State ---
        self._running = False
        self._last_processed_sample = 0
        self._utterance_audio: List[np.ndarray] = []
        self._whisper_model = None
        self._use_faster_whisper = False
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="conv")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start the conversation loop + callback server."""
        logger.info("ConversationEngine starting...")

        # Pre-flight: check LLM
        llm_ok = await self._llm.is_available()
        logger.info(f"  LLM available: {llm_ok}")

        # Pre-flight: check stage
        stage_ok = await self._stage.health()
        logger.info(f"  Stage server reachable: {stage_ok}")

        # Register built-in skills
        self._register_builtin_skills()

        # Start callback server
        await self._start_callback_server()

        # Sync sample counter with receiver
        self._last_processed_sample = self._receiver.total_samples_written

        self._running = True
        logger.info("ConversationEngine started — entering listen loop")
        conv_log.info("=" * 50)
        conv_log.info("Conversation Engine ONLINE")
        conv_log.info(f"  LLM: {'OK' if llm_ok else 'UNAVAILABLE'}")
        conv_log.info(f"  Stage: {'OK' if stage_ok else 'UNAVAILABLE'}")
        conv_log.info(f"  Callback: http://{self._server_ip}:{self._callback_port}")
        conv_log.info("=" * 50)
        conv_log.info("Listening...")

        try:
            await self._listen_loop()
        except asyncio.CancelledError:
            logger.info("ConversationEngine cancelled")
        except Exception as e:
            logger.error(f"ConversationEngine error: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        """Graceful shutdown."""
        self._running = False

        if self._callback_runner:
            await self._callback_runner.cleanup()
            self._callback_runner = None

        await self._stage.close()
        self._thread_pool.shutdown(wait=False)

        logger.info("ConversationEngine stopped")

    # ------------------------------------------------------------------
    # Built-in skills
    # ------------------------------------------------------------------

    def _register_builtin_skills(self):
        """Register speak and emote skills."""

        async def _speak_handler(text: str = "", **_kw):
            """Synthesize text and stream to stage server."""
            if not text:
                return
            logger.info(f"[speak] {text[:80]}{'...' if len(text) > 80 else ''}")
            conv_log.info(f"  >> SPEAK: {text}")

            # Run TTS in thread (blocking model)
            loop = asyncio.get_running_loop()
            t0 = time.monotonic()
            audio = await loop.run_in_executor(
                self._thread_pool, self._tts.synthesize, text
            )
            tts_ms = (time.monotonic() - t0) * 1000
            if audio is None:
                logger.warning("TTS returned None — skipping speak")
                conv_log.info(f"  >> TTS failed ({tts_ms:.0f}ms)")
                return

            conv_log.info(f"  >> TTS OK ({tts_ms:.0f}ms, {len(audio)} samples)")

            # Build callback URL
            callback_url = f"http://{self._server_ip}:{self._callback_port}/speak_done"

            # Stream to stage
            speak_id = await self._stage.speak_chunked(
                audio,
                sample_rate=self._tts.sample_rate,
                callback_url=callback_url,
            )
            conv_log.info(f"  >> Streaming to stage ({speak_id})")

            # Wait for playback to finish (with timeout)
            event = self._register_speak(speak_id)
            try:
                await asyncio.wait_for(event.wait(), timeout=60.0)
                logger.debug(f"Playback finished: {speak_id}")
                conv_log.info(f"  >> Playback done ({speak_id})")
            except asyncio.TimeoutError:
                logger.warning(f"Playback callback timeout for {speak_id}")
                conv_log.info(f"  >> Playback TIMEOUT ({speak_id})")
            finally:
                self._speak_events.pop(speak_id, None)

        async def _emote_handler(name: str = "", **_kw):
            """Send an emote to the stage server."""
            if not name:
                return
            logger.info(f"[emote] {name}")
            conv_log.info(f"  >> EMOTE: {name}")
            await self._stage.react(name)

        self._skills.register(Skill(
            name="speak",
            description="Speak text aloud to the user",
            parameters={"text": "What to say"},
            handler=_speak_handler,
        ))

        self._skills.register(Skill(
            name="emote",
            description="Display a facial expression / emote",
            parameters={"name": "Expression name (e.g. happy, excited, thinking)"},
            handler=_emote_handler,
        ))

        # --- Mirror skill (requires Florence2) ---
        if self._florence2 is not None:
            async def _mirror_handler(**_kw):
                """Fetch mirror image and describe it with Florence2."""
                logger.info("[mirror] Fetching viewport screenshot...")
                conv_log.info("  >> MIRROR: fetching...")

                image_bytes = await self._stage.mirror()
                if not image_bytes:
                    logger.warning("[mirror] Failed to get image from stage")
                    conv_log.info("  >> MIRROR: failed (no image)")
                    return

                conv_log.info(f"  >> MIRROR: got {len(image_bytes)} bytes, captioning...")

                # Run Florence2 in thread (blocking model)
                loop = asyncio.get_running_loop()
                t0 = time.monotonic()
                description = await loop.run_in_executor(
                    self._thread_pool, self._florence2.caption, image_bytes
                )
                caption_ms = (time.monotonic() - t0) * 1000

                if not description:
                    logger.warning("[mirror] Florence2 returned empty caption")
                    conv_log.info(f"  >> MIRROR: empty caption ({caption_ms:.0f}ms)")
                    return

                logger.info(f"[mirror] Caption ({caption_ms:.0f}ms): {description}")
                conv_log.info(f"  >> MIRROR ({caption_ms:.0f}ms): {description}")

                # Inject into conversation so LLM can react
                self._history.append({
                    "role": "system",
                    "content": f"[Mirror view: {description}]"
                })

                # Trigger a follow-up LLM call so Selestia reacts to what she saw
                conv_log.info("  >> MIRROR: prompting reaction...")
                await self._respond()

            self._skills.register(Skill(
                name="mirror",
                description="Look at yourself — captures the game viewport showing your avatar and describes your appearance",
                parameters={},
                handler=_mirror_handler,
            ))

        logger.info(f"Built-in skills registered: {self._skills.names}")

    # ------------------------------------------------------------------
    # Listen loop — poll AudioReceiver, run VAD
    # ------------------------------------------------------------------

    async def _listen_loop(self):
        """Poll audio buffer, run VAD, dispatch utterances."""
        sample_rate = self._receiver.sample_rate
        frame_samples = self._vad.frame_length  # samples per VAD frame

        while self._running:
            # How much new audio has arrived?
            total_now = self._receiver.total_samples_written
            new_samples = total_now - self._last_processed_sample

            if new_samples < frame_samples:
                await asyncio.sleep(_POLL_INTERVAL)
                continue

            # Grab new audio from the rolling buffer
            seconds_new = new_samples / sample_rate
            audio = self._receiver.get_recent_audio(seconds_new)
            self._last_processed_sample = total_now

            # Feed to VAD in frame-sized chunks
            offset = 0
            while offset + frame_samples <= len(audio):
                frame = audio[offset : offset + frame_samples]
                sample_pos = total_now - len(audio) + offset

                result = self._vad.process_audio_chunk(frame, sample_pos)
                event = result["event"]

                if event == "utterance_start":
                    self._utterance_audio.clear()
                    self._utterance_audio.append(frame)

                elif event == "utterance_continue":
                    self._utterance_audio.append(frame)

                elif event == "utterance_end":
                    self._utterance_audio.append(frame)
                    utterance = np.concatenate(self._utterance_audio)
                    self._utterance_audio.clear()
                    duration = len(utterance) / sample_rate
                    logger.info(f"Utterance captured: {duration:.2f}s ({len(utterance)} samples)")
                    conv_log.info(f"--- Utterance detected ({duration:.1f}s) ---")
                    await self._process_utterance(utterance)

                elif event == "utterance_discard":
                    self._utterance_audio.clear()

                offset += frame_samples

            await asyncio.sleep(_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Process utterance: Whisper → LLM → Skills
    # ------------------------------------------------------------------

    async def _process_utterance(self, audio: np.ndarray):
        """Transcribe, call LLM, parse and execute skills."""

        # --- Whisper transcription (blocking, run in thread) ---
        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(
            self._thread_pool, self._transcribe, audio
        )

        if not transcript or not transcript.strip():
            logger.debug("Empty transcription, skipping")
            conv_log.info("  (empty transcription — skipped)")
            return

        logger.info(f"Transcribed: \"{transcript}\"")
        conv_log.info(f"  USER: {transcript}")

        # Add to conversation history
        self._history.append({"role": "user", "content": transcript})

        # LLM → skills
        await self._respond()

    async def _respond(self):
        """Call LLM with current history, parse tags, execute skills.

        Reusable — called after user utterance and after async skill
        results (like mirror) that need a follow-up reaction.
        """
        messages = self._build_messages()
        t0 = time.monotonic()
        response = await self._llm.chat(messages)
        llm_ms = (time.monotonic() - t0) * 1000

        if not response or not response.strip():
            logger.warning("Empty LLM response")
            conv_log.info(f"  LLM: (empty response, {llm_ms:.0f}ms)")
            return

        # Strip Qwen3 <think>…</think> blocks (thinking mode leakage)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if not response:
            logger.warning("LLM response was only a think block")
            conv_log.info(f"  LLM: (think-only response, {llm_ms:.0f}ms)")
            return

        logger.info(f"LLM response: \"{response[:120]}{'...' if len(response) > 120 else ''}\"")
        conv_log.info(f"  SELESTIA ({llm_ms:.0f}ms): {response}")

        # Add to history (full response with tags — tags stripped for next turn)
        self._history.append({"role": "assistant", "content": response})

        # --- Parse skill tags and execute ---
        clean_text, invocations = self._skills.parse_tags(response)

        if not invocations:
            # No explicit tags — treat the whole response as speech
            if clean_text:
                await self._skills.execute("speak", text=clean_text)
            return

        # Execute invocations in order (skip mirror to prevent loops)
        for skill_name, arg in invocations:
            if skill_name == "mirror":
                continue  # don't re-trigger mirror from a mirror response
            elif skill_name == "speak":
                # [speak] means "say the remaining clean text"
                await self._skills.execute("speak", text=clean_text)
            elif skill_name == "emote":
                await self._skills.execute("emote", name=arg or "neutral")
            else:
                await self._skills.execute(skill_name, **({"arg": arg} if arg else {}))

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run STT on audio (blocking — called from thread pool).

        Uses faster-whisper (CTranslate2) if available, falls back to
        OpenAI Whisper.  faster-whisper is typically 4-8x faster on CPU.
        """
        if self._whisper_model is None:
            model_name = self._config.get("whisper_model", "base")
            device = self._config.get("whisper_device", "cpu")

            try:
                from faster_whisper import WhisperModel
                # CTranslate2: int8 on CPU is fast; float16 if CUDA
                compute = "int8" if device == "cpu" else "float16"
                logger.info(f"Loading faster-whisper '{model_name}' on {device} ({compute})...")
                self._whisper_model = WhisperModel(
                    model_name, device=device, compute_type=compute
                )
                self._use_faster_whisper = True
                logger.info("faster-whisper loaded for ConversationEngine")
            except ImportError:
                import whisper
                logger.info(f"faster-whisper not installed, falling back to openai-whisper")
                logger.info(f"Loading Whisper model '{model_name}' on {device}...")
                self._whisper_model = whisper.load_model(model_name, device=device)
                self._use_faster_whisper = False
                logger.info("Whisper model loaded for ConversationEngine")

        if self._use_faster_whisper:
            segments, _info = self._whisper_model.transcribe(
                audio, language="en", beam_size=1, vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        else:
            result = self._whisper_model.transcribe(
                audio, language="en", fp16=False,
            )
            return result.get("text", "").strip()

    def _build_messages(self) -> List[Dict[str, str]]:
        """Build the message list for the LLM."""
        skill_block = self._skills.list_for_prompt()
        system = self._system_prompt
        if skill_block:
            system += "\n\n" + skill_block

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(self._history)
        return messages

    # ------------------------------------------------------------------
    # Callback server — receives POST /speak_done from stage server
    # ------------------------------------------------------------------

    def _register_speak(self, speak_id: str) -> asyncio.Event:
        """Register a speak_id and return an Event to wait on."""
        event = asyncio.Event()
        self._speak_events[speak_id] = event
        return event

    async def _start_callback_server(self):
        """Start a tiny aiohttp server for speech completion callbacks."""
        app = web.Application()
        app.router.add_post("/speak_done", self._handle_speak_done)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._callback_port)
        await site.start()

        self._callback_app = app
        self._callback_runner = runner
        logger.info(f"Callback server listening on :{self._callback_port}")

    async def _handle_speak_done(self, request: web.Request) -> web.Response:
        """Handle POST /speak_done from the stage server."""
        try:
            data = await request.json()
            speak_id = data.get("id", "")
            status = data.get("status", "unknown")
            logger.debug(f"Speak callback: id={speak_id}, status={status}")

            event = self._speak_events.get(speak_id)
            if event:
                event.set()
            else:
                logger.debug(f"No waiter for speak_id={speak_id}")

            return web.json_response({"ok": True})
        except Exception as e:
            logger.warning(f"Callback handler error: {e}")
            return web.json_response({"error": str(e)}, status=400)

    @staticmethod
    def _detect_lan_ip() -> str:
        """Auto-detect this machine's LAN IP by opening a UDP socket."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
