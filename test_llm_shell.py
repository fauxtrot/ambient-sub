"""Interactive LLM shell for testing stage emote + speak integration.

Usage:
    python test_llm_shell.py

Type messages, the LLM responds with [emote:X] and/or [speak] tags,
and actions are dispatched to the stage server in real time.

Commands:
    /emote <name>    - Send an emote directly (e.g. /emote excited)
    /say <text>      - TTS + speak directly (bypass LLM)
    /reset           - Reset expression to neutral
    /stop            - Stop current speech
    /reactions       - List available reactions
    /tts             - Check TTS availability
    /quit            - Exit
"""
import asyncio
import logging
import re
import sys

import numpy as np

# Add project to path
sys.path.insert(0, "ambient_subconscious")

from ambient_subconscious.executive.stage_client import StageClient
from ambient_subconscious.executive.tts import TTSEngine
from ambient_subconscious.llm.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shell")

# Available reactions (fetched from server on startup, fallback list here)
FALLBACK_REACTIONS = [
    "aaah", "blep", "bruh", "dead", "default", "derp", "disgust", "eerr",
    "ehh", "excited", "focused", "gasp", "heartpupils", "heh", "hehe",
    "hmm", "love", "neko", "ooh", "owo", "pfft", "pout", "rage", "shh",
    "shockbarest", "sleepy", "Sly", "smug", "sob", "sparkle", "tch",
    "uwu", "waa", "wink", "yikes",
]

SYSTEM_PROMPT = """You are Selestia, an AI VTuber character observing and chatting with the user. You can:

- React with expressions: [emote:name]
  Available: {reactions}
- Speak aloud: [speak] followed by what to say
- Think silently: plain text (no tags)

Combine: "[emote:excited] [speak] Oh wow, that's amazing!"
React only: "[emote:hmm]"
Speak only: "[speak] Let me think about that..."

Keep responses short and natural. You're chatty and expressive.
Use emotes frequently to show personality."""

# Regex for tokenizing the response into an ordered sequence of actions
ACTION_RE = re.compile(r"\[emote:(\w+)\]|\[speak\]", re.IGNORECASE)


def parse_response(text: str) -> list[dict]:
    """Parse LLM response into an ordered list of actions.

    Handles interleaved emotes and speech blocks like:
        [emote:sparkle] [speak] Hello! [emote:hmm] [speak] What's up?
    Returns:
        [{"type": "emote", "value": "sparkle"},
         {"type": "speak", "value": "Hello!"},
         {"type": "emote", "value": "hmm"},
         {"type": "speak", "value": "What's up?"}]
    """
    actions = []
    last_end = 0

    for match in ACTION_RE.finditer(text):
        tag_start = match.start()

        # If there's a pending speak action, close it with text up to this tag
        if actions and actions[-1]["type"] == "speak" and actions[-1]["value"] is None:
            speech_text = text[last_end:tag_start].strip()
            if speech_text:
                actions[-1]["value"] = speech_text
            else:
                actions.pop()  # empty speak, discard

        if match.group(1):
            # [emote:name]
            actions.append({"type": "emote", "value": match.group(1)})
        else:
            # [speak] — text follows until next tag or end
            actions.append({"type": "speak", "value": None})

        last_end = match.end()

    # Close final speak with remaining text
    if actions and actions[-1]["type"] == "speak" and actions[-1]["value"] is None:
        speech_text = text[last_end:].strip()
        if speech_text:
            actions[-1]["value"] = speech_text
        else:
            actions.pop()

    return actions


async def execute_actions(actions: list[dict], stage: StageClient, tts: TTSEngine):
    """Execute an ordered list of emote/speak actions sequentially."""
    for action in actions:
        if action["type"] == "emote":
            name = action["value"]
            print(f"  -> emote: {name}")
            await stage.react(name)

        elif action["type"] == "speak":
            text = action["value"]
            print(f"  -> speak: {text}")
            audio = await asyncio.to_thread(tts.synthesize, text)
            if audio is not None:
                print(f"     audio: {len(audio)} samples ({len(audio)/tts.sample_rate:.2f}s)")
                await stage.speak_chunked(audio, tts.sample_rate)
            else:
                silence = np.zeros(int(tts.sample_rate * 0.1), dtype=np.float32)
                await stage.speak_chunked(silence, tts.sample_rate)
                print("     (TTS unavailable, sent silence)")


async def main():
    stage = StageClient()
    tts = TTSEngine(voice="fantine")
    llm = OllamaClient(
        host="http://localhost:11434",
        model="deepseek-r1:8b",
        temperature=0.7,
    )

    # Check connectivity
    print("Checking connections...")

    stage_ok = await stage.health()
    print(f"  Stage server: {'OK' if stage_ok else 'OFFLINE'}")

    llm_ok = await llm.is_available()
    print(f"  Ollama LLM:   {'OK' if llm_ok else 'OFFLINE'}")

    # Probe TTS to see native vs target sample rates
    print(f"  TTS voice:    {tts.voice_name}")
    print(f"  TTS target:   {tts.sample_rate} Hz")
    tts._ensure_initialized()
    if tts._model is not None:
        print(f"  TTS native:   {tts._model_sample_rate} Hz")
        if tts._model_sample_rate != tts._target_sample_rate:
            print(f"  ** RESAMPLING {tts._model_sample_rate} -> {tts._target_sample_rate} **")
    else:
        print(f"  TTS model:    NOT LOADED")

    # Fetch available reactions
    reactions = FALLBACK_REACTIONS
    if stage_ok:
        server_reactions = await stage.get_reactions()
        if server_reactions:
            reactions = server_reactions
            print(f"  Reactions:    {len(reactions)} loaded from server")
        else:
            print(f"  Reactions:    using fallback list ({len(reactions)})")
    else:
        print(f"  Reactions:    using fallback list ({len(reactions)})")

    system_prompt = SYSTEM_PROMPT.format(reactions=", ".join(reactions))

    # Dialogue history for context
    history = []

    print()
    print("=" * 60)
    print("LLM Shell — type messages, /help for commands, /quit to exit")
    print("=" * 60)
    print()

    try:
        while True:
            try:
                user_input = await asyncio.to_thread(input, "you> ")
            except EOFError:
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "/quit" or cmd == "/exit":
                    break
                elif cmd == "/emote":
                    if arg:
                        await stage.react(arg)
                        print(f"  -> emote: {arg}")
                    else:
                        print("  Usage: /emote <name>")
                elif cmd == "/say":
                    if arg:
                        await execute_actions({"speak": arg}, stage, tts)
                    else:
                        print("  Usage: /say <text>")
                elif cmd == "/reset":
                    await stage.reset_reaction()
                    print("  -> reset to neutral")
                elif cmd == "/stop":
                    await stage.stop_speaking()
                    print("  -> speech stopped")
                elif cmd == "/reactions":
                    print(f"  Available ({len(reactions)}): {', '.join(reactions)}")
                elif cmd == "/tts":
                    avail = tts.available
                    print(f"  TTS available: {avail}")
                    if avail:
                        print(f"  Target sample rate: {tts.sample_rate}")
                        print(f"  Native sample rate: {tts._model_sample_rate}")
                        print(f"  Voice: {tts.voice_name}")
                elif cmd == "/testwav":
                    # Synthesize and save to WAV for local listening comparison
                    test_text = arg or "Hello, this is a sample rate test."
                    print(f"  Synthesizing: {test_text}")
                    audio = await asyncio.to_thread(tts.synthesize, test_text)
                    if audio is not None:
                        import wave
                        import struct
                        wav_path = "test_tts_output.wav"
                        with wave.open(wav_path, 'w') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(tts.sample_rate)
                            # Convert float32 [-1,1] to int16
                            int16_audio = (audio * 32767).astype(np.int16)
                            wf.writeframes(int16_audio.tobytes())
                        print(f"  Saved: {wav_path} ({len(audio)} samples, {len(audio)/tts.sample_rate:.2f}s)")
                        print(f"  Listen locally to check if it sounds normal vs chipmunk")
                    else:
                        print("  TTS unavailable")
                elif cmd == "/voice":
                    if arg:
                        # Switch voice (requires re-init)
                        print(f"  Switching voice to: {arg}")
                        tts = TTSEngine(voice=arg)
                        tts._ensure_initialized()
                        if tts.available:
                            print(f"  Voice loaded. Native: {tts._model_sample_rate} Hz")
                        else:
                            print(f"  Failed to load voice '{arg}'")
                    else:
                        print(f"  Current: {tts.voice_name}")
                        print(f"  Available: {', '.join(TTSEngine.PREDEFINED_VOICES)}")
                elif cmd == "/help":
                    print("  /emote <name>  - Send emote directly")
                    print("  /say <text>    - TTS + speak directly")
                    print("  /reset         - Reset to neutral expression")
                    print("  /stop          - Stop current speech")
                    print("  /reactions     - List available reactions")
                    print("  /tts           - Check TTS status")
                    print("  /testwav [txt] - Synth + save WAV locally")
                    print("  /voice [name]  - Show/switch voice")
                    print("  /quit          - Exit")
                else:
                    print(f"  Unknown command: {cmd}")
                continue

            # Add user message to history
            history.append(f"User: {user_input}")

            # Build prompt
            history_str = "\n".join(history[-10:])  # last 10 exchanges
            prompt = f"""{system_prompt}

Conversation:
{history_str}

Respond as Selestia:"""

            # Call LLM
            print("  (thinking...)")
            response = await llm.complete(prompt)

            # Strip <think>...</think> blocks from deepseek-r1
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            if not cleaned:
                cleaned = response.strip()

            print(f"  LLM: {cleaned}")

            # Parse and execute
            actions = parse_response(cleaned)
            await execute_actions(actions, stage, tts)

            # Add to history
            history.append(f"Selestia: {cleaned}")

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        await stage.close()
        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
