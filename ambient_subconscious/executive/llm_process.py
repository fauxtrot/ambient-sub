"""
Executive LLM Process - Generates observations when context updates.

Subscribes to ExecutiveContext updates via Svelte bridge and generates
observations using Ollama LLM with conversational and reasoning prompts.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
import websockets

from ..llm import OllamaClient
from .attention_dynamics import AttentionDynamics

logger = logging.getLogger(__name__)


class ExecutiveLLM:
    """
    Subscribes to ExecutiveContext updates and generates observations.
    Uses dual-prompt architecture: conversational for quick responses,
    reasoning for deeper analysis when triggered.
    """

    def __init__(
        self,
        process_id: str = "executive-llm-001",
        bridge_url: str = "ws://localhost:8175",
        llm_config: Optional[Dict[str, Any]] = None,
        conversational_config: Optional[Dict[str, Any]] = None,
        reasoning_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Executive LLM process.

        Args:
            process_id: Unique identifier for this process
            bridge_url: WebSocket URL for Svelte bridge
            llm_config: LLM configuration (host, model, etc.)
            conversational_config: Conversational prompt config
            reasoning_config: Reasoning prompt config
        """
        self.process_id = process_id
        self.bridge_url = bridge_url

        # Default configs
        llm_config = llm_config or {}
        conversational_config = conversational_config or {}
        reasoning_config = reasoning_config or {}

        # Initialize LLM
        self.llm = OllamaClient(
            host=llm_config.get('host', 'http://localhost:11434'),
            model=llm_config.get('model', 'deepseek-r1:8b'),
            temperature=conversational_config.get('temperature', 0.7)
        )

        self.conversational_prompt = conversational_config.get('system_prompt', """
You are an ambient AI assistant observing the user's environment.
Provide brief, helpful observations about what you notice.
Keep responses to 1-2 sentences unless more detail is warranted.
""")

        self.reasoning_prompt = reasoning_config.get('system_prompt', """
You are an AI assistant performing deeper analysis of the user's context.
Provide thoughtful insights, identify patterns, and suggest helpful actions.
""")

        self.reasoning_enabled = reasoning_config.get('enabled', True)
        self.reasoning_triggers = set(reasoning_config.get('triggers', [
            'question', 'why', 'how', 'what if', 'should i', 'can you'
        ]))

        # Respond mode config
        self.respond_enabled = conversational_config.get('respond_enabled', True)
        self.respond_triggers = set(conversational_config.get('respond_triggers', [
            'hey', 'agent', 'hello', '?'
        ]))

        # Initialize Attention Dynamics
        attention_config = conversational_config.get('attention', {})
        self.attention = AttentionDynamics(attention_config)
        logger.info(f"[Executive] Attention dynamics: {self.attention}")

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self):
        """Connect to Svelte bridge and start listening."""
        self.running = True
        logger.info(f"Executive LLM {self.process_id} starting...")
        logger.info(f"Connecting to bridge at {self.bridge_url}")

        # Check LLM availability
        if not await self.llm.is_available():
            logger.error("LLM not available - cannot start")
            logger.error(f"Make sure Ollama is running at {self.llm.host}")
            return

        logger.info(f"LLM available: {self.llm.model}")

        try:
            async with websockets.connect(self.bridge_url) as ws:
                self.websocket = ws

                # Register
                await ws.send(json.dumps({
                    "type": "register",
                    "process_id": self.process_id,
                    "process_type": "executive",
                    "subscriptions": ["executive_context"]
                }))

                # Wait for registration
                msg = await ws.recv()
                response = json.loads(msg)
                if response["type"] == "registered":
                    logger.info(f"Executive LLM registered: {response['process_id']}")

                # Start heartbeat task
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                # Main event loop
                try:
                    while self.running:
                        msg = await ws.recv()
                        event = json.loads(msg)

                        if event["type"] == "event":
                            await self._handle_event(event)
                        elif event["type"] == "heartbeat_ack":
                            pass  # Acknowledged
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Connection to bridge closed")
                finally:
                    if self._heartbeat_task:
                        self._heartbeat_task.cancel()

        except Exception as e:
            logger.error(f"Executive LLM error: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        """Gracefully shutdown."""
        self.running = False
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({
                    "type": "shutdown",
                    "process_id": self.process_id,
                    "reason": "Graceful shutdown"
                }))
            except:
                pass
        logger.info(f"Executive LLM {self.process_id} stopped")

    async def _handle_event(self, event: Dict[str, Any]):
        """Handle ExecutiveContext update event."""
        if event["table"] != "executive_context":
            return

        event_type = event["event_type"]
        if event_type not in ["insert", "update"]:
            return

        context = event["row"]
        logger.info(f"Received context update: {event_type}")

        # Generate observation
        await self._generate_observation(context)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to bridge to keep connection alive."""
        while self.running:
            try:
                await asyncio.sleep(15.0)  # Send heartbeat every 15 seconds

                if self.websocket:
                    await self.websocket.send(json.dumps({
                        "type": "heartbeat",
                        "process_id": self.process_id
                    }))
                    logger.debug("Sent heartbeat to bridge")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")

    def _should_respond(self, context: Dict[str, Any]) -> bool:
        """Determine if agent should respond (not just observe)."""
        if not self.respond_enabled:
            return False

        # Check if user directly addressed agent
        try:
            recent_audio = context.get('recentAudio', '[]')
            audio_entries = json.loads(recent_audio) if isinstance(recent_audio, str) else recent_audio
        except json.JSONDecodeError:
            return False

        if not audio_entries:
            return False

        latest_entry = audio_entries[-1]
        speaker = latest_entry.get('speaker', '')
        text = latest_entry.get('text', '').lower()

        # User speaking + contains trigger keyword
        if speaker.lower() in ["user", "unknown"]:
            return any(trigger in text for trigger in self.respond_triggers)

        return False

    async def _create_response_entry(self, response_text: str):
        """Create TranscriptEntry for agent response."""
        import time

        entry_id = f"agent_{int(time.time() * 1000)}"

        try:
            await self.websocket.send(json.dumps({
                "type": "reducer_call",
                "process_id": self.process_id,
                "reducer": "CreateEntry",
                "args": {
                    "sessionId": 1,
                    "entryId": entry_id,
                    "durationMs": 0,
                    "transcript": response_text,
                    "confidence": 1.0,
                    "audioClipPath": None,
                    "recordingStartMs": 0,
                    "recordingEndMs": 0
                }
            }))

            # Set speaker to "Agent" using the new reducer
            await self.websocket.send(json.dumps({
                "type": "reducer_call",
                "process_id": self.process_id,
                "reducer": "UpdateEntrySpeakerByEntryId",
                "args": {
                    "entryId": entry_id,
                    "speakerName": "Agent"
                }
            }))

            logger.info(f"Created agent response entry: {entry_id}")
        except Exception as e:
            logger.error(f"Failed to create response entry: {e}")

    def _calculate_next_checkin(self, context: Dict[str, Any]) -> int:
        """Calculate next check-in time based on context activity."""
        import time

        # Parse audio activity
        try:
            recent_audio = context.get('recentAudio', '[]')
            audio_entries = json.loads(recent_audio) if isinstance(recent_audio, str) else recent_audio
            baseline_audio = context.get('baselineAudio', '{}')
            baseline_data = json.loads(baseline_audio) if isinstance(baseline_audio, str) else {}
            avg_rate = baseline_data.get('avg_entries_per_min', 0.0)
        except json.JSONDecodeError:
            audio_entries = []
            avg_rate = 0.0

        # Decision logic
        if len(audio_entries) > 2:
            # Active conversation - check in 30 seconds
            delay = 30
        elif len(audio_entries) > 0:
            # Some activity - check in 1 minute
            delay = 60
        else:
            # Quiet - check in 1 hour
            delay = 3600

        next_time = int(time.time()) + delay
        logger.info(f"Scheduling next check-in in {delay}s (activity: {len(audio_entries)} entries)")

        return next_time

    async def _schedule_checkin(self, next_check_in: int):
        """Update ExecutiveContext with scheduled check-in time."""
        try:
            await self.websocket.send(json.dumps({
                "type": "reducer_call",
                "process_id": self.process_id,
                "reducer": "UpdateExecutiveContext",
                "args": {
                    "recentVisual": None,
                    "recentAudio": None,
                    "baselineVisual": None,
                    "baselineAudio": None,
                    "userState": None,
                    "agentState": None,
                    "nextCheckIn": next_check_in,
                    "notes": None
                }
            }))
            logger.debug(f"Scheduled next check-in at {next_check_in}")
        except Exception as e:
            logger.error(f"Failed to schedule check-in: {e}")

    def _calculate_delta_magnitude(self, context: Dict[str, Any]) -> float:
        """
        Calculate delta magnitude from context for attention boost.

        Uses visual and audio deltas from baseline to determine significance.
        Returns value between 0.0 (no change) and 1.0 (major change).
        """
        try:
            # Parse visual delta
            visual_objects = []
            baseline_visual = {}

            recent_visual_raw = context.get('recentVisual', '[]')
            if isinstance(recent_visual_raw, str):
                visual_objects = json.loads(recent_visual_raw)
            elif isinstance(recent_visual_raw, list):
                visual_objects = recent_visual_raw

            baseline_visual_raw = context.get('baselineVisual', '{}')
            if isinstance(baseline_visual_raw, str):
                baseline_visual = json.loads(baseline_visual_raw)
            elif isinstance(baseline_visual_raw, dict):
                baseline_visual = baseline_visual_raw

            # Calculate visual delta
            visual_delta = 0.0
            if baseline_visual:
                avg_objects = baseline_visual.get('avg_objects', 0)
                if avg_objects > 0:
                    current_count = len(visual_objects)
                    visual_delta = abs(current_count - avg_objects) / (avg_objects + 1)
                    visual_delta = min(1.0, visual_delta)  # Cap at 1.0

            # Parse audio delta
            audio_entries = []
            baseline_audio = {}

            recent_audio_raw = context.get('recentAudio', '[]')
            if isinstance(recent_audio_raw, str):
                audio_entries = json.loads(recent_audio_raw)
            elif isinstance(recent_audio_raw, list):
                audio_entries = recent_audio_raw

            baseline_audio_raw = context.get('baselineAudio', '{}')
            if isinstance(baseline_audio_raw, str):
                baseline_audio = json.loads(baseline_audio_raw)
            elif isinstance(baseline_audio_raw, dict):
                baseline_audio = baseline_audio_raw

            # Calculate audio delta (new entries = significant)
            audio_delta = 0.0
            if len(audio_entries) > 0:
                audio_delta = min(1.0, len(audio_entries) / 5.0)  # 5+ entries = max delta

            # Combine deltas (take max of visual or audio)
            delta_magnitude = max(visual_delta, audio_delta)

            logger.debug(f"[Executive] Delta magnitude: {delta_magnitude:.2f} "
                        f"(visual={visual_delta:.2f}, audio={audio_delta:.2f})")

            return delta_magnitude

        except Exception as e:
            logger.error(f"Error calculating delta magnitude: {e}")
            return 0.0  # Default to no delta on error

    async def _generate_observation(self, context: Dict[str, Any]):
        """
        Generate observation using Conscious Agent architecture.

        Uses structured JSON output with attention dynamics, context enrichment,
        and capability routing.
        """
        try:
            # 1. Update attention dynamics
            self.attention.update_on_cycle()  # Apply decay
            delta_magnitude = self._calculate_delta_magnitude(context)
            self.attention.boost_from_delta(delta_magnitude)

            logger.info(f"[Conscious Agent] Attention: {self.attention.activation:.2f}, "
                       f"Next check-in: {self.attention.get_check_in_seconds()}s")

            # 2. Parse context fields
            recent_visual = context.get('recentVisual', '[]')
            recent_audio = context.get('recentAudio', '[]')
            user_state = context.get('userState', 'unknown')
            agent_state = context.get('agentState', 'active')

            # Parse JSON fields
            try:
                visual_objects = json.loads(recent_visual) if isinstance(recent_visual, str) else recent_visual
                audio_entries = json.loads(recent_audio) if isinstance(recent_audio, str) else recent_audio
            except json.JSONDecodeError:
                visual_objects = []
                audio_entries = []

            # 3. Check if response mode needed
            respond_mode = self._should_respond(context)

            # 4. Build Conscious Agent prompt (structured JSON output)
            prompt = self._build_conscious_prompt(
                visual_objects,
                audio_entries,
                user_state,
                agent_state,
                respond_mode
            )

            # 5. Get structured response from LLM
            logger.info(f"[Conscious Agent] Generating {'response' if respond_mode else 'observation'}...")
            response_text = await self.llm.complete(prompt)

            # 6. Parse structured JSON (try to extract JSON from response)
            structured_response = self._parse_structured_response(response_text)

            if not structured_response:
                # Fallback: treat as simple text response
                logger.warning("[Conscious Agent] Failed to parse structured response, using fallback")
                structured_response = {
                    "attention_activation": self.attention.activation,
                    "needs_reasoning": False,
                    "capability": "text_response" if not respond_mode else "text_response",
                    "response_content": response_text,
                    "log_for_review": False
                }

            # 7. Log enriched context
            enrichment = structured_response.get('context_enrichment', {})
            if enrichment:
                logger.info(f"[Context Enrichment] Action: {enrichment.get('action_context', 'N/A')}")
                logger.info(f"[Context Enrichment] Tags: {enrichment.get('tags', [])}")
                logger.info(f"[Context Enrichment] Significance: {enrichment.get('significance', 0):.2f}")

            # 8. Handle reasoning if needed
            if structured_response.get('needs_reasoning', False) and self.reasoning_enabled:
                logger.info("[Conscious Agent] Invoking reasoner...")
                reasoning_query = structured_response.get('reasoning_query', 'Analyze this situation')

                # Build reasoning prompt
                reasoning_prompt = f"""{self.reasoning_prompt}

Context:
{json.dumps({
    'visual_objects': visual_objects,
    'audio_entries': audio_entries,
    'user_state': user_state,
    'agent_state': agent_state
}, indent=2)}

Question: {reasoning_query}

Provide deep analysis and recommended action.
"""

                analysis = await self.llm.complete(reasoning_prompt)
                logger.info(f"[Reasoning] {analysis}")

                # Merge reasoning into response
                original_content = structured_response.get('response_content', '')
                structured_response['response_content'] = f"{original_content}\n\n[Reasoning: {analysis}]"

            # 9. Execute capability (route response)
            response_content = structured_response.get('response_content', '')
            capability = structured_response.get('capability', 'text_response')

            logger.info(f"[Conscious Agent] Output: {response_content}")

            if capability == 'text_response' or not respond_mode:
                # Just log (already done above)
                pass
            elif respond_mode and capability in ['text_response', 'audio_response', 'respond_via_channel']:
                # Create transcript entry for response
                await self._create_response_entry(response_content)

            # 10. Log decision if needed
            if structured_response.get('log_for_review', False):
                decision_ctx = structured_response.get('decision_context', {})
                logger.info(f"[Decision Log] Uncertainty: {decision_ctx.get('uncertainty', 0):.2f}")
                logger.info(f"[Decision Log] Reasoning: {decision_ctx.get('reasoning', 'N/A')}")
                # TODO: Implement decision logging to DecisionLog table/file

            # 11. Schedule next check-in using attention dynamics
            check_in_seconds = structured_response.get(
                'next_check_in_seconds',
                self.attention.get_check_in_seconds()
            )
            next_check_in = int(time.time()) + check_in_seconds
            await self._schedule_checkin(next_check_in)

        except Exception as e:
            logger.error(f"[Conscious Agent] Error: {e}", exc_info=True)

    def _needs_reasoning(self, response: str) -> bool:
        """Check if reasoning is triggered based on response content."""
        response_lower = response.lower()
        return any(trigger in response_lower for trigger in self.reasoning_triggers)

    def _build_conscious_prompt(
        self,
        visual_objects: list,
        audio_entries: list,
        user_state: str,
        agent_state: str,
        respond_mode: bool
    ) -> str:
        """
        Build Conscious Agent prompt with structured JSON output instructions.

        Returns a prompt that instructs the model to output structured JSON
        with fields for attention, reasoning decisions, capability routing, etc.
        """
        # Build audio transcripts section
        audio_section = ""
        if audio_entries:
            audio_section = "\nRecent Audio Transcripts:\n"
            for entry in audio_entries:
                speaker = entry.get('speaker', 'Unknown')
                text = entry.get('text', '')
                audio_section += f"  - {speaker}: {text}\n"

        prompt = f"""You are a Conscious Agent managing an ambient awareness system.

ROLE:
- Executive decision-maker (not just conversational responder)
- You decide: what's happening, is it significant, should you reason deeper, should you respond
- You manage your own attention and check-in schedule

CURRENT STATE:
- Attention Activation: {self.attention.activation:.2f}
- Autonomy Level: {self.attention.autonomy_level:.2f}

CONTEXT:
- Visual Objects: {', '.join(visual_objects) if visual_objects else 'none'}
- Audio Entries: {len(audio_entries)} recent entries{audio_section}
- User State: {user_state}
- Agent State: {agent_state}

MODE:
- {'RESPONSE MODE: User has addressed you directly. Generate a conversational response.' if respond_mode else 'OBSERVATION MODE: Provide brief observation of context changes.'}

DECISION FRAMEWORK:
1. Enrich context: Add semantic meaning (action_context, tags, significance)
2. Assess significance: Is this context significant? (rate 0-1)
3. Decide reasoning: If uncertain/novel/consequential, set needs_reasoning=true
4. Choose capability: text_response, audio_response, or respond_via_channel
5. Decide logging: If uncertain>0.5 OR novel OR consequential, set log_for_review=true
6. Calculate next check-in: Based on activation and activity level

OUTPUT FORMAT (JSON):
{{
  "attention_activation": <float 0.0-1.0>,
  "context_enrichment": {{
    "action_context": "<what's happening in plain language>",
    "tags": ["<tag1>", "<tag2>"],
    "significance": <float 0.0-1.0>
  }},
  "needs_reasoning": <true/false>,
  "reasoning_query": "<question for reasoner if needs_reasoning=true>",
  "capability": "<text_response|audio_response|respond_via_channel>",
  "response_content": "<your response/observation>",
  "log_for_review": <true/false>,
  "decision_context": {{
    "uncertainty": <float 0.0-1.0>,
    "reasoning": "<why you made this decision>"
  }},
  "next_check_in_seconds": <integer seconds>
}}

TAGS to use:
- "routine", "novel", "significant", "concerning", "monitoring_required"
- "user_present", "user_away", "activity_detected", "quiet", "conversation"
- "uncertain", "confident", "needs_attention"

Remember:
- Local reasoning is FREE - use liberally when uncertain
- Token usage should be EMERGENT - speak naturally based on complexity
- You have AGENCY - make executive decisions
- Attention has DECAY BIAS - you want to relax, deltas wake you up

Generate your structured JSON response:"""

        return prompt

    def _parse_structured_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse structured JSON response from LLM.

        Attempts to extract JSON from the response, handling cases where
        the model may include extra text before/after the JSON.

        Returns:
            Parsed dictionary if successful, None otherwise
        """
        try:
            # Try direct parse first
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                pass

            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try to find JSON object anywhere in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            logger.warning("[Conscious Agent] Could not extract JSON from response")
            return None

        except Exception as e:
            logger.error(f"[Conscious Agent] Error parsing structured response: {e}")
            return None


def setup_logging(log_file: str, debug: bool = False):
    """Configure logging to both file and console."""
    import os

    # Create logs directory
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)

    # Console handler (for terminal viewing)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)


async def main():
    """Main entry point for Executive LLM process."""
    import sys
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Executive LLM process for ambient-subconscious")
    parser.add_argument('--bridge-url', default='ws://localhost:8175',
                        help='WebSocket URL for bridge server')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--log-file', default='logs/llm.log',
                        help='Log file path')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--respond', type=bool, default=None,
                        help='Enable response generation (overrides config)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file, args.debug)

    # Load config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        executive_config = config.get("executive", {})
    else:
        # Try to load from default location
        config_path = Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            executive_config = config.get("executive", {})
        else:
            logger.warning("No config file found, using defaults")
            executive_config = {}

    # Override conversational config with command-line args if provided
    conversational_config = executive_config.get("conversational", {})
    if args.respond is not None:
        conversational_config['respond_enabled'] = args.respond

    llm = ExecutiveLLM(
        bridge_url=args.bridge_url,
        llm_config=executive_config.get("llm", {}),
        conversational_config=conversational_config,
        reasoning_config=executive_config.get("reasoning", {})
    )

    try:
        await llm.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await llm.stop()


if __name__ == "__main__":
    asyncio.run(main())
