# Executive Agent Architecture - Full Vision

## Document Purpose
This document describes the complete vision for the Executive Agent, including current Phase 1 implementation and future phases for streaming inference, interruption handling, and multi-party conversation support.

---

## Core Philosophy

The Executive Agent is the **conscious interface** of the ambient subconscious system. It:
- **Observes** the environment through provider streams
- **Reasons** about context using dual-mode LLM processing
- **Speaks** naturally in multi-party conversations
- **Learns** from user corrections via context editing
- **Adapts** by feeding context edits back to the subconscious layer

---

## Architecture Evolution

### Phase 1: Sequential Dual-Prompt (CURRENT - IMPLEMENTED)
```
┌─────────────────────────────────────────────┐
│          EXECUTIVE AGENT (Phase 1)          │
├─────────────────────────────────────────────┤
│                                             │
│  1. Poll provider data (frames, entries)   │
│  2. Update side context                    │
│  3. Generate conversational response        │
│     ↓ (check keywords)                     │
│  4. [Optional] Escalate to reasoning       │
│  5. Apply context edits                    │
│  6. Output to console                       │
│                                             │
└─────────────────────────────────────────────┘
         │
         ↓ (single LLM, two prompts)
    deepseek-r1:8b via Ollama
```

**Characteristics**:
- ✓ Working and tested
- ✓ Sequential: conversational → reasoning (if needed)
- ✓ Context editing functional
- ✓ Dialogue history maintained
- ✗ No interruption handling
- ✗ No streaming
- ✗ No parallel execution

---

### Phase 2: Streaming with Interruption (FUTURE)
```
┌─────────────────────────────────────────────┐
│          EXECUTIVE AGENT (Phase 2)          │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────────────────┐          │
│  │  Conversational Agent        │          │
│  │  (streaming inference)       │          │
│  │                              │          │
│  │  • Generates response        │          │
│  │  • Can be interrupted ◄──────┼─ User   │
│  │  • Saves partial output      │   Input  │
│  └──────────────────────────────┘          │
│         │                                   │
│         ↓ (on keywords or decision)        │
│  ┌──────────────────────────────┐          │
│  │  Reasoning Agent             │          │
│  │  (background task)           │          │
│  │                              │          │
│  │  • Spawned asynchronously   │          │
│  │  • Enriches context          │          │
│  │  • Interruptible             │          │
│  └──────────────────────────────┘          │
│                                             │
└─────────────────────────────────────────────┘
```

**New Capabilities**:
- ✓ Streaming inference (Ollama streaming API)
- ✓ Interruption detection (user input monitoring)
- ✓ Partial output capture
- ✓ Sequential still (conversational first, then reasoning)
- ✗ Not yet parallel

**Implementation Requirements**:
- Ollama streaming: `"stream": true` in API calls
- Async task spawning for reasoning
- Interruption buffer for partial responses
- Prompt awareness: "You may be interrupted mid-thought"

---

### Phase 3: Parallel Agents (FUTURE)
```
┌─────────────────────────────────────────────────────┐
│            EXECUTIVE AGENT (Phase 3)                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌───────────────────────┐  ┌──────────────────┐  │
│  │ Conversational Agent  │  │ Reasoning Agent  │  │
│  │ (main loop)           │  │ (background)     │  │
│  │                       │  │                  │  │
│  │ • Maintains dialogue  │  │ • Continuous     │  │
│  │ • Reads enriched      │──┤   enrichment     │  │
│  │   context from ───────┘  │ • Longform mode  │  │
│  │   reasoning buffer       │ • Follow-on mode │  │
│  │ • Delegates thinking     │                  │  │
│  │   tasks ─────────────────┤                  │  │
│  │ • Interruptible          │ • Interruptible  │  │
│  └───────────────────────┘  └──────────────────┘  │
│         ▲                           │              │
│         │                           ↓              │
│         └───── USER INTERRUPT ──────┘              │
│         (both agents can be stopped)               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**New Capabilities**:
- ✓ True parallelism: both agents run simultaneously
- ✓ Conversational pipes from reasoning_buffer
- ✓ Reasoning runs continuously in background
- ✓ Both interruptible independently
- ✓ Asynchronous enrichment

**Implementation Requirements**:
- Shared context with reasoning_buffer field
- Background task management (asyncio)
- Inter-agent communication channel
- Interruption priority handling

---

### Phase 4: Multi-Party Conversation (FUTURE)
```
┌─────────────────────────────────────────────────────┐
│            EXECUTIVE AGENT (Phase 4)                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌───────────────────────┐  ┌──────────────────┐  │
│  │ Conversational Agent  │  │ Reasoning Agent  │  │
│  │ (dialogue manager)    │  │ (context         │  │
│  │                       │  │  enricher)       │  │
│  │ • Handles turn-taking │  │                  │  │
│  │ • Agent speaks        │  │ • Longform mode  │  │
│  │ • User speaks    ◄────┼──┤ • Follow-on mode │  │
│  │ • Discord speaks      │  │ • Captures       │  │
│  │ • Maintains           │  │   partial on     │  │
│  │   conversational flow │  │   interrupt      │  │
│  │ • Reads enriched      │  │ • Resumes with   │  │
│  │   context        ─────┼──┤   added context  │  │
│  └───────────────────────┘  └──────────────────┘  │
│         ▲                           │              │
│         │                           ↓              │
│     Multi-party Interruptions                      │
│     • User: "Hey, what time is it?"                │
│     • Discord: "We should meet tomorrow"           │
│     • Audio: New speaker detected                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**New Capabilities**:
- ✓ Multi-party conversation (user, agent, Discord users)
- ✓ Turn-taking management
- ✓ Speaker attribution
- ✓ Follow-on reasoning with interruption context
- ✓ Conversational appropriateness maintained

**Interruption Flow Example**:
```
[Reasoning: Longform]
  "The user appears focused because..."

[USER INTERRUPTS: "Hey, what time is it?"]

[Reasoning: Save partial] → context.reasoning_buffer
  "The user appears focused because..."

[Conversational: Responds]
  "It's 3:45 PM"

[Reasoning: Follow-on Mode]
  Previous thought: "The user appears focused because..."
  New context: User asked about time, might be time-pressured
  Continue: "...their focus may be deadline-driven"
```

---

## Dual-Mode LLM System

### Conversational Mode
**Purpose**: Natural dialogue, quick responses, conversational appropriateness

**System Prompt**:
```
You are an observant AI assistant monitoring the user's environment.
Respond naturally and concisely in 1-2 sentences. Focus on what you observe
and notice. You may be interrupted mid-thought by the user - this is normal.
```

**Characteristics**:
- Fast (512 max tokens)
- Concise (1-2 sentences)
- Conversationally aware
- Can read from reasoning_buffer
- Temperature: 0.7

**Example Output**:
```
"I notice you're at your desk with your laptop. The coffee cup suggests
you've been working for a while."
```

### Reasoning Mode
**Purpose**: Deep analysis, strategic thinking, context enrichment

**System Prompt**:
```
You are a strategic reasoning agent. Provide deep analysis, identify patterns,
and recommend actions. Think step-by-step about complex situations. Your
thoughts may be interrupted - save your progress and continue when resumed.
```

**Characteristics**:
- Deeper (1024 max tokens)
- Analytical
- Pattern recognition
- Two modes: Longform and Follow-on
- Temperature: 0.7

**Longform Mode** (uninterrupted):
```
### Strategic Analysis
The user has been at their desk for approximately 2 hours based on...

### Patterns Identified
1. Consistent laptop usage suggests focused work
2. Coffee cup placement indicates routine...

### Recommendations
Monitor for break signals. User may benefit from...
```

**Follow-on Mode** (after interruption):
```
### Resuming Analysis
Previous thought: "User appears focused"
Interruption: User asked about time (3:45 PM)
New insight: Time awareness + focus = deadline pressure

### Updated Analysis
The user's focus combined with time-checking behavior suggests...
```

---

## Context Architecture

### Side Context Structure
```python
self.context = {
    # Provider-enriched fields
    "current_environment": "home office, afternoon",
    "recent_audio": "User: 'I need to finish this by 4 PM'",
    "recent_visual": "Detected: laptop, monitor, coffee cup, person",
    "user_state": "focused, time-pressured",

    # Agent-managed fields
    "agent_state": "active, monitoring",
    "notes": ["User has deadline at 4 PM", "Coffee suggests long session"],

    # Reasoning enrichment (Phase 3+)
    "reasoning_buffer": "User's focus appears deadline-driven...",
}
```

### Dialogue History
```python
self.dialogue_history = deque(maxlen=50)  # Bounded memory

# Example entries:
{
    "role": "user",
    "content": "What time is it?",
    "timestamp": 1738772345.2
}
{
    "role": "assistant",
    "content": "It's 3:45 PM",
    "timestamp": 1738772345.8
}
```

---

## Streaming Implementation (Phase 2)

### Ollama Streaming API
```python
async def stream_complete(self, prompt: str, **kwargs):
    """Stream completion with interruption support."""
    payload = {
        "model": self.model,
        "prompt": prompt,
        "temperature": kwargs.get('temperature', 0.7),
        "stream": True  # Enable streaming
    }

    partial_response = ""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{self.host}/api/generate",
            json=payload,
        ) as response:
            async for line in response.content:
                if self.interrupted:
                    # Save partial response
                    self.partial_buffer = partial_response
                    break

                chunk = json.loads(line)
                token = chunk.get("response", "")
                partial_response += token

                # Yield token for real-time display
                yield token

    return partial_response
```

### Interruption Detection
```python
class InterruptionMonitor:
    """Monitors for user input while LLM generates."""

    def __init__(self):
        self.interrupted = False
        self.interrupt_source = None

    async def monitor(self):
        """Watch for user input (text, audio, Discord)."""
        while True:
            # Check for text input
            if stdin_has_input():
                self.interrupted = True
                self.interrupt_source = "text"
                break

            # Check for audio input
            if new_audio_entry():
                self.interrupted = True
                self.interrupt_source = "audio"
                break

            await asyncio.sleep(0.1)
```

---

## Parallel Execution (Phase 3)

### Architecture
```python
class ExecutiveAgent:
    def __init__(self):
        self.conversational = ConversationalAgent()
        self.reasoning = ReasoningAgent()

        # Shared context
        self.context = SharedContext()

        # Background tasks
        self.reasoning_task = None

    async def start(self):
        """Start both agents."""
        # Start conversational in main loop
        conversational_task = asyncio.create_task(
            self.conversational.run(self.context)
        )

        # Start reasoning in background
        reasoning_task = asyncio.create_task(
            self.reasoning.run(self.context)
        )

        # Both run independently
        await asyncio.gather(conversational_task, reasoning_task)
```

### Conversational Agent (Main Loop)
```python
class ConversationalAgent:
    async def run(self, context):
        """Main dialogue loop."""
        while True:
            # 1. Check for user input
            user_input = await self.wait_for_input()

            if user_input:
                # Interrupt reasoning if running
                await context.interrupt_reasoning()

            # 2. Read enriched context from reasoning
            enriched = context.get_reasoning_buffer()

            # 3. Generate response (with enrichment)
            response = await self.generate_response(
                user_input, enriched
            )

            # 4. Output
            print(f"[Agent] {response}")

            await asyncio.sleep(0.1)
```

### Reasoning Agent (Background)
```python
class ReasoningAgent:
    async def run(self, context):
        """Continuous reasoning loop."""
        while True:
            # 1. Check if interrupted
            if context.is_interrupted():
                # Save partial thinking
                context.save_partial_reasoning(self.current_thought)

                # Wait for conversational to finish
                await context.wait_for_resume()

                # Resume in follow-on mode
                mode = "follow-on"
            else:
                mode = "longform"

            # 2. Generate reasoning
            analysis = await self.reason(context, mode)

            # 3. Write to reasoning buffer
            context.update_reasoning_buffer(analysis)

            await asyncio.sleep(5)  # Periodic updates
```

---

## Follow-on Reasoning (Phase 4)

### Prompt Construction
```python
def build_followon_prompt(self, partial_thought, interruption):
    """Build prompt for follow-on reasoning."""
    return f"""
You were thinking:
{partial_thought}

You were interrupted by:
{interruption}

Continue your analysis, incorporating this new information.
"""
```

### Example Flow
```
Time: 3:00 PM
[Reasoning: Longform]
"The user has been at their desk for 2 hours. Their posture
suggests focused work. The laptop placement and monitor
configuration indicate..."

Time: 3:45 PM
[User]: "Hey, what time is it?"
[Conversational]: "It's 3:45 PM"

[Reasoning: Saved Partial]
"...laptop placement and monitor configuration indicate professional
work setup. The user appears to be..."

[Reasoning: Follow-on with new context]
Previous: "The user appears to be..."
Interruption: User asked for time at 3:45 PM
New insight: Time awareness + previous focus = deadline pressure

"...engaged in time-sensitive work. The time check at 3:45 PM
suggests awareness of an upcoming deadline. Combined with the
2-hour focused session, this indicates the user is working under
time pressure but maintaining productivity."
```

---

## Multi-Party Conversation Support (Phase 4)

### Turn-Taking Protocol
```python
class ConversationManager:
    """Manages multi-party conversation flow."""

    def __init__(self):
        self.current_speaker = None
        self.turn_queue = []

    async def process_turn(self, speaker, content):
        """Process a turn in the conversation."""
        # 1. Interrupt current generation if speaking
        if self.current_speaker == "agent":
            await self.interrupt_agent()

        # 2. Add to dialogue history
        self.dialogue_history.append({
            "speaker": speaker,
            "content": content,
            "timestamp": time.time()
        })

        # 3. Update reasoning context
        await self.reasoning_agent.add_context(
            f"{speaker} said: {content}"
        )

        # 4. Generate agent response (if appropriate)
        if self.should_respond(speaker, content):
            response = await self.conversational_agent.respond()
            self.current_speaker = "agent"

        self.current_speaker = None
```

### Speaker Attribution
```python
class SpeakerManager:
    """Tracks speakers in multi-party conversation."""

    speakers = {
        "user": "Primary user at desk",
        "discord_user1": "Discord: Alice",
        "discord_user2": "Discord: Bob",
        "agent": "Ambient Assistant"
    }

    async def attribute_speaker(self, audio_event):
        """Determine who is speaking."""
        # Use Diart + trained speaker embeddings
        embedding = await self.get_speaker_embedding(audio_event)
        speaker_id = await self.match_speaker(embedding)
        return self.speakers.get(speaker_id, "Unknown")
```

---

## Implementation Phases

### Phase 1: Sequential Dual-Prompt ✓ DONE
**Status**: Implemented and tested
**Files**:
- `ambient_subconscious/executive/executive_agent.py`
- `ambient_subconscious/llm/ollama_client.py`
- `test_executive.py`

**Capabilities**:
- Sequential conversational → reasoning
- Context editing
- Dialogue history
- Keyword-based escalation

---

### Phase 2: Streaming with Interruption
**Status**: Designed, not implemented
**Estimated**: 1-2 weeks

**Steps**:
1. Implement streaming Ollama API wrapper
2. Add interruption monitoring (stdin, audio events)
3. Update prompts with interruption awareness
4. Test with live interruptions

**Files to modify**:
- `ambient_subconscious/llm/ollama_client.py` - add streaming
- `ambient_subconscious/executive/executive_agent.py` - add interruption handling
- New: `ambient_subconscious/executive/interruption_monitor.py`

---

### Phase 3: Parallel Agents
**Status**: Designed, not implemented
**Estimated**: 2-3 weeks

**Steps**:
1. Implement SharedContext with reasoning_buffer
2. Split conversational and reasoning into separate async tasks
3. Add inter-agent communication
4. Implement follow-on reasoning mode
5. Test parallel execution

**Files to modify**:
- `ambient_subconscious/executive/executive_agent.py` - split into two agents
- New: `ambient_subconscious/executive/shared_context.py`
- New: `ambient_subconscious/executive/conversational_agent.py`
- New: `ambient_subconscious/executive/reasoning_agent.py`

---

### Phase 4: Multi-Party Conversation
**Status**: Designed, not implemented
**Estimated**: 3-4 weeks

**Steps**:
1. Implement ConversationManager
2. Add speaker attribution system
3. Integrate Discord voice/text
4. Add turn-taking protocol
5. Test with multiple speakers

**Files to create**:
- `ambient_subconscious/executive/conversation_manager.py`
- `ambient_subconscious/executive/speaker_manager.py`
- `ambient_subconscious/agents/discord_agent.py`

---

## Testing Strategy

### Phase 1 Tests (Current)
- ✓ Standalone executive test
- ✓ Context editing
- ✓ Reasoning escalation
- [ ] Integration with main.py
- [ ] Live webcam/audio data

### Phase 2 Tests
- [ ] Streaming inference
- [ ] Interruption capture
- [ ] Partial output preservation
- [ ] Prompt awareness

### Phase 3 Tests
- [ ] Parallel execution
- [ ] Inter-agent communication
- [ ] Follow-on reasoning
- [ ] Context synchronization

### Phase 4 Tests
- [ ] Multi-speaker attribution
- [ ] Turn-taking correctness
- [ ] Discord integration
- [ ] Conversational flow quality

---

## Configuration

### Current (Phase 1)
```yaml
executive:
  enabled: true
  update_interval_seconds: 5
  context_window_seconds: 30

  llm:
    provider: "ollama"
    model: "deepseek-r1:8b"
    host: "http://localhost:11434"

  conversational:
    temperature: 0.7
    max_tokens: 512
    system_prompt: "..."

  reasoning:
    temperature: 0.7
    max_tokens: 1024
    enabled: true
    system_prompt: "..."
    triggers: ["think", "analyze", "reason", "decide", "complex"]
```

### Future (Phase 2+)
```yaml
executive:
  streaming:
    enabled: true
    buffer_size: 128
    display_partial: true

  interruption:
    enabled: true
    sources: ["text", "audio", "discord"]
    save_partial: true

  parallel:
    enabled: true
    reasoning_continuous: true
    reasoning_interval: 5

  multi_party:
    enabled: true
    speaker_attribution: true
    turn_taking_protocol: "polite"  # polite, aggressive, passive
```

---

## Notes for Future Phases

### Performance Considerations
- Streaming adds ~10-20ms latency vs. batch
- Parallel agents increase GPU/CPU usage
- Context synchronization needs careful locking
- Dialogue history should be bounded (deque)

### Memory Management
- Reasoning_buffer should have max size
- Partial responses need cleanup after merge
- Speaker embeddings cached, not recomputed

### Error Handling
- Interruption mid-token: save partial + token
- Ollama timeout: fallback to non-streaming
- Context desync: reset and resync from source
- Multiple simultaneous interruptions: priority queue

---

## References

- **Current Implementation**: `test_executive.py`, `executive_agent.py`
- **Ollama Streaming API**: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
- **Asyncio Patterns**: https://docs.python.org/3/library/asyncio-task.html
- **Multi-Party Conversation**: See `.docs/continuous_latent_continuous_thought.md`
