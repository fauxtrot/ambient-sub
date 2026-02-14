"""
Executive Agent with dual-LLM architecture.

Conversational (Personaplex) + Reasoning (Ollama) for high-level decisions.
Speaks and emotes via stage server integration.
"""

import asyncio
import json
import logging
import re
import time
from collections import deque
from typing import Dict, List, Optional, Any

import aiohttp
import numpy as np

from ..llm import PersonaplexClient, OllamaClient
from .stage_client import StageClient
from .tts import TTSEngine

logger = logging.getLogger(__name__)

# Regex for tokenizing the response into an ordered sequence of actions
ACTION_RE = re.compile(r"\[emote:(\w+)\]|\[speak\]", re.IGNORECASE)


class ExecutiveAgent:
    """
    Executive agent with chatbot-style dialogue + editable context.

    Architecture:
    - Dialogue history: Chat-style conversation log
    - Side context: Editable context enriched by providers
    - Dual LLMs: Conversational (fast) + Reasoning (deep)
    - Stage server: Emote and speak via VTuber avatar
    """

    def __init__(
        self,
        llm_config: Dict[str, Any],
        conversational_config: Dict[str, Any],
        reasoning_config: Dict[str, Any],
        svelte_api_url: str = "http://localhost:5174",
        update_interval: int = 5,
        context_window_seconds: int = 30,
        stage_config: Optional[Dict[str, Any]] = None,
        tts_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize executive agent.

        Args:
            llm_config: Base LLM configuration (model, host)
            conversational_config: Conversational mode config (prompt, temperature)
            reasoning_config: Reasoning mode config (prompt, temperature, triggers)
            svelte_api_url: URL of Svelte API for querying data
            update_interval: Seconds between updates
            context_window_seconds: Time window for recent data
            stage_config: Stage server config (enabled, host, port)
            tts_config: TTS config (provider, voice, speaker_wav, sample_rate)
        """
        self.svelte_api_url = svelte_api_url.rstrip('/')
        self.update_interval = update_interval
        self.context_window_seconds = context_window_seconds

        # Initialize single LLM client
        self.llm = OllamaClient(
            host=llm_config.get('host', 'http://localhost:11434'),
            model=llm_config.get('model', 'deepseek-r1:8b'),
            temperature=conversational_config.get('temperature', 0.7)
        )

        # Store prompts for different modes
        self.conversational_prompt = conversational_config.get('system_prompt',
            "You are an observant AI assistant. Respond naturally and concisely.")
        self.conversational_temperature = conversational_config.get('temperature', 0.7)
        self.conversational_max_tokens = conversational_config.get('max_tokens', 512)

        self.reasoning_prompt = reasoning_config.get('system_prompt',
            "You are a strategic reasoning agent. Provide deep analysis.")
        self.reasoning_temperature = reasoning_config.get('temperature', 0.7)
        self.reasoning_max_tokens = reasoning_config.get('max_tokens', 1024)

        self.reasoning_enabled = reasoning_config.get('enabled', True)
        self.reasoning_triggers = set(reasoning_config.get('triggers', [
            'think', 'analyze', 'reason', 'decide', 'complex'
        ]))

        # Stage server (emotes + speech)
        stage_config = stage_config or {}
        self.stage_enabled = stage_config.get('enabled', True)
        if self.stage_enabled:
            self.stage = StageClient(
                host=stage_config.get('host', '127.0.0.1'),
                port=stage_config.get('port', 7860),
            )
        else:
            self.stage = None

        # TTS engine
        tts_config = tts_config or {}
        self.tts = TTSEngine(
            voice=tts_config.get('voice', 'fantine'),
            speaker_wav=tts_config.get('speaker_wav'),
            sample_rate=tts_config.get('sample_rate', 24000),
        )

        # Available reactions (fetched from stage server on startup)
        self.available_reactions: List[str] = []

        # Track background action tasks
        self._action_task: Optional[asyncio.Task] = None

        # Dialogue history (deque for memory efficiency)
        self.dialogue_history = deque(maxlen=50)

        # Side context (editable by executive)
        self.context = {
            "current_environment": "",
            "recent_audio": "",
            "recent_visual": "",
            "user_state": "",
            "agent_state": "initializing",
            "notes": []
        }

        self.running = False

        logger.info(f"ExecutiveAgent initialized: interval={update_interval}s, window={context_window_seconds}s")

    async def start(self):
        """Start executive agent main loop."""
        self.running = True
        logger.info("Executive agent starting...")

        # Check LLM availability
        llm_available = await self.llm.is_available()

        logger.info(f"LLM available: {llm_available}")

        if not llm_available:
            logger.error("LLM not available - executive agent cannot start")
            return

        # Connect to stage server and fetch reactions
        if self.stage_enabled and self.stage:
            stage_ok = await self.stage.health()
            if stage_ok:
                self.available_reactions = await self.stage.get_reactions()
                logger.info(f"Stage server connected: {len(self.available_reactions)} reactions available")
            else:
                logger.warning("Stage server not reachable - emote/speak disabled")
                self.stage_enabled = False

        self.context["agent_state"] = "active"

        try:
            await self._run_loop()
        except KeyboardInterrupt:
            logger.info("Executive agent interrupted by user")
        except Exception as e:
            logger.error(f"Executive agent error: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        """Stop executive agent."""
        self.running = False
        self.context["agent_state"] = "stopped"
        logger.info("Executive agent stopping...")

        # Cancel any in-flight action task
        if self._action_task and not self._action_task.done():
            self._action_task.cancel()

        # Close stage client session
        if self.stage:
            await self.stage.close()

        logger.info("Executive agent stopped")

    async def _run_loop(self):
        """Main executive loop."""
        logger.info("Executive agent loop started")

        while self.running:
            loop_start = time.time()

            try:
                # 1. Poll recent provider data
                frames, entries = await self._poll_provider_data()

                # 2. Update side context from providers
                self._update_context_from_providers(frames, entries)

                # 3. Generate observation/thought
                response = await self._think()

                # 4. Apply context edits
                if response.get("context_edit"):
                    self._apply_context_edit(response["context_edit"])

                # 5. Output thought to console
                message = response.get("message", "")
                if message:
                    print(f"[Executive:Conversational] {message}")

                # 6. Execute stage actions (emote/speak) as background task
                actions = response.get("actions", [])
                if actions:
                    # Cancel previous action if still running
                    if self._action_task and not self._action_task.done():
                        self._action_task.cancel()
                    self._action_task = asyncio.create_task(
                        self._execute_actions(actions)
                    )

                # 7. Add to dialogue history
                self.dialogue_history.append({
                    "role": "assistant",
                    "content": message,
                    "timestamp": time.time()
                })

            except Exception as e:
                logger.error(f"Error in executive loop: {e}", exc_info=True)

            # Maintain update interval
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.update_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _execute_actions(self, actions: List[Dict[str, str]]):
        """Execute an ordered list of emote/speak actions sequentially.

        Runs as a background asyncio.Task so it doesn't block the main loop.
        """
        if not self.stage_enabled or not self.stage:
            return

        try:
            for action in actions:
                if action["type"] == "emote":
                    emote = action["value"]
                    if emote in self.available_reactions:
                        await self.stage.react(emote)
                        logger.info(f"[Stage] emote: {emote}")
                    else:
                        logger.warning(f"[Stage] unknown reaction '{emote}', skipping")

                elif action["type"] == "speak":
                    text = action["value"]
                    logger.info(f"[Stage] speak: {text}")

                    audio = await asyncio.to_thread(self.tts.synthesize, text)

                    if audio is not None:
                        await self.stage.speak_chunked(audio, self.tts.sample_rate)
                    else:
                        # TTS failed — send brief silence so server doesn't hang
                        silence = np.zeros(int(self.tts.sample_rate * 0.1), dtype=np.float32)
                        await self.stage.speak_chunked(silence, self.tts.sample_rate)

        except asyncio.CancelledError:
            logger.debug("[Stage] action task cancelled")
        except Exception as e:
            logger.error(f"[Stage] action error: {e}", exc_info=True)

    async def _poll_provider_data(self):
        """
        Poll recent frames and entries from Svelte API.

        Returns:
            Tuple of (frames, entries)
        """
        since_timestamp = time.time() - self.context_window_seconds

        try:
            async with aiohttp.ClientSession() as session:
                # Query frames
                frames_task = session.get(
                    f"{self.svelte_api_url}/api/frame/query",
                    params={"since": since_timestamp, "limit": 20}
                )

                # Query entries
                entries_task = session.get(
                    f"{self.svelte_api_url}/api/entry/query",
                    params={"since": since_timestamp, "limit": 10}
                )

                # Execute in parallel
                frames_resp, entries_resp = await asyncio.gather(
                    frames_task, entries_task, return_exceptions=True
                )

                frames = []
                entries = []

                if not isinstance(frames_resp, Exception):
                    frames_data = await frames_resp.json()
                    frames = frames_data.get("frames", [])

                if not isinstance(entries_resp, Exception):
                    entries_data = await entries_resp.json()
                    entries = entries_data.get("entries", [])

                return frames, entries

        except Exception as e:
            logger.warning(f"Failed to poll provider data: {e}")
            return [], []

    def _update_context_from_providers(self, frames: List[Dict], entries: List[Dict]):
        """Update side context from provider input tokens."""
        # Aggregate visual detections from recent frames
        if frames:
            objects = []
            for frame in frames:
                try:
                    detections = json.loads(frame.get("detections", "[]"))
                    objects.extend([d["class"] for d in detections])
                except (json.JSONDecodeError, KeyError):
                    pass

            # Count frequency
            from collections import Counter
            object_counts = Counter(objects)
            top_objects = [obj for obj, _ in object_counts.most_common(5)]

            if top_objects:
                self.context["recent_visual"] = f"Detected: {', '.join(top_objects)}"
            else:
                self.context["recent_visual"] = "No objects detected"

        # Aggregate recent audio transcripts
        if entries:
            latest_entry = entries[0]
            transcript = latest_entry.get("transcript", "")
            speaker = latest_entry.get("speaker", "Unknown")

            self.context["recent_audio"] = f"{speaker}: {transcript}" if transcript else "No recent audio"

    async def _think(self) -> Dict[str, Any]:
        """
        Generate response using conversational mode (or reasoning if triggered).

        Returns:
            Dict with "message", optional "context_edit", "emote", and "speak"
        """
        prompt = self._build_prompt(mode="conversational")

        # Get conversational response
        response_text = await self.llm.complete(
            prompt,
            temperature=self.conversational_temperature
        )

        # Strip <think>...</think> blocks from deepseek-r1
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        if cleaned:
            response_text = cleaned

        # Check if reasoning is needed
        if self.reasoning_enabled and self._needs_reasoning(response_text):
            logger.info("[Executive] Escalating to reasoning mode")
            print("[Executive:Reasoning] ESCALATED TO REASONING")

            # Build reasoning prompt
            reasoning_prompt = self._build_prompt(mode="reasoning", conversation=response_text)

            # Get deep analysis
            analysis = await self.llm.complete(
                reasoning_prompt,
                temperature=self.reasoning_temperature
            )

            print(f"[Executive:Reasoning] {analysis}")

            # Merge insights
            response_text = self._merge_reasoning(response_text, analysis)

        # Parse response
        return self._parse_response(response_text)

    def _build_prompt(self, mode: str = "conversational", conversation: str = None) -> str:
        """
        Build prompt from dialogue history + side context.

        Args:
            mode: "conversational" or "reasoning"
            conversation: Previous conversation (for reasoning mode)

        Returns:
            Formatted prompt with system instructions
        """
        context_str = json.dumps(self.context, indent=2)
        dialogue_str = self._format_dialogue()

        if mode == "conversational":
            # Build system prompt with available actions
            if self.stage_enabled and self.available_reactions:
                reactions_str = ", ".join(self.available_reactions)
                system_prompt = f"""You are an AI character observing the user's environment. You can:

- React with expressions: [emote:name]
  Available: {reactions_str}
- Speak aloud: [speak] followed by what to say
- Think silently: plain text (no tags)

Combine: "[emote:excited] [speak] Oh wow, that's amazing!"
React only: "[emote:hmm]"

Most of the time, observe silently. Speak when you have something meaningful to say.
Emote freely to show you're paying attention."""
            else:
                system_prompt = self.conversational_prompt

            task = """Based on recent observations, provide a brief response.
Respond naturally and concisely."""

            return f"""{system_prompt}

Current Context:
{context_str}

Recent Dialogue:
{dialogue_str}

{task}
"""
        else:  # reasoning mode
            system_prompt = self.reasoning_prompt
            return f"""{system_prompt}

Current Context:
{context_str}

Recent Conversation:
{conversation}

Provide:
1. Strategic analysis of what's happening
2. Patterns or insights you notice
3. Recommended focus or actions

Keep your analysis concise but insightful.
"""

    def _format_dialogue(self) -> str:
        """Format dialogue history for prompt."""
        if not self.dialogue_history:
            return "(No dialogue yet)"

        lines = []
        for msg in list(self.dialogue_history)[-5:]:  # Last 5 messages
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _needs_reasoning(self, response: str) -> bool:
        """Check if response contains reasoning triggers."""
        response_lower = response.lower()
        return any(trigger in response_lower for trigger in self.reasoning_triggers)

    def _merge_reasoning(self, conversation: str, analysis: str) -> str:
        """Merge conversational response with reasoning analysis."""
        return f"{conversation}\n\n[Analysis: {analysis}]"

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response into an ordered list of actions + context edits.

        Handles interleaved emotes and speech like:
            [emote:sparkle] [speak] Hello! [emote:hmm] [speak] What's up?

        Returns:
            Dict with "message", "actions" (list of {type, value}), and optional "context_edit"
        """
        result = {"message": response_text.strip(), "actions": [], "context_edit": None}

        # Tokenize into ordered action sequence
        actions = []
        last_end = 0

        for match in ACTION_RE.finditer(response_text):
            tag_start = match.start()

            # Close pending speak with text up to this tag
            if actions and actions[-1]["type"] == "speak" and actions[-1]["value"] is None:
                speech_text = response_text[last_end:tag_start].strip()
                if speech_text:
                    actions[-1]["value"] = speech_text
                else:
                    actions.pop()

            if match.group(1):
                actions.append({"type": "emote", "value": match.group(1)})
            else:
                actions.append({"type": "speak", "value": None})

            last_end = match.end()

        # Close final speak with remaining text
        if actions and actions[-1]["type"] == "speak" and actions[-1]["value"] is None:
            speech_text = response_text[last_end:].strip()
            if speech_text:
                actions[-1]["value"] = speech_text
            else:
                actions.pop()

        result["actions"] = actions

        # Try to extract JSON context edits only if no action tags found
        if not actions:
            try:
                if "{" in response_text and "}" in response_text:
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    json_str = response_text[start:end]
                    parsed = json.loads(json_str)

                    if "message" in parsed:
                        result["message"] = parsed["message"]
                    if "context_edit" in parsed:
                        result["context_edit"] = parsed["context_edit"]
            except (json.JSONDecodeError, ValueError):
                pass

        return result

    def _apply_context_edit(self, edit: Dict[str, Any]):
        """Executive edits its own context."""
        for key, value in edit.items():
            if key in self.context:
                old_value = self.context[key]
                self.context[key] = value
                print(f"[Executive] Updated context.{key}: {old_value} -> {value}")
            elif key == "notes" and isinstance(value, str):
                # Append to notes
                if isinstance(self.context["notes"], list):
                    self.context["notes"].append(value)
                    print(f"[Executive] Added note: {value}")

    def get_context(self) -> Dict[str, Any]:
        """Get current context (for subconscious feedback loop)."""
        return self.context.copy()
