"""
Standalone test script for Executive Agent.

Tests the single-LLM dual-prompt architecture with deepseek-r1:8b.
"""

import asyncio
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from ambient_subconscious.executive import ExecutiveAgent


async def test_executive():
    """Test executive agent with mock data."""

    print("=" * 60)
    print("Executive Agent Test")
    print("=" * 60)
    print()

    # Configuration
    llm_config = {
        "host": "http://localhost:11434",
        "model": "deepseek-r1:8b"
    }

    conversational_config = {
        "temperature": 0.7,
        "max_tokens": 512,
        "system_prompt": "You are an observant AI assistant monitoring the user's environment. Respond naturally and concisely in 1-2 sentences. Focus on what you observe and notice."
    }

    reasoning_config = {
        "temperature": 0.7,
        "max_tokens": 1024,
        "enabled": True,
        "system_prompt": "You are a strategic reasoning agent. Provide deep analysis, identify patterns, and recommend actions. Think step-by-step about complex situations.",
        "triggers": ["think", "analyze", "reason", "decide", "complex"]
    }

    # Initialize executive agent
    print("Initializing Executive Agent...")
    print(f"LLM: {llm_config['model']}")
    print(f"Host: {llm_config['host']}")
    print()

    agent = ExecutiveAgent(
        llm_config=llm_config,
        conversational_config=conversational_config,
        reasoning_config=reasoning_config,
        svelte_api_url="http://localhost:5174",
        update_interval=5,
        context_window_seconds=30
    )

    # Check LLM availability
    print("Checking LLM availability...")
    available = await agent.llm.is_available()
    print(f"[OK] LLM available: {available}")
    print()

    if not available:
        print("ERROR: Ollama not available! Make sure ollama is running:")
        print("  ollama serve")
        print("  ollama pull deepseek-r1:8b")
        return

    # Test 1: Simple observation (conversational)
    print("Test 1: Simple Observation")
    print("-" * 60)
    agent.context["recent_visual"] = "Detected: laptop, person, coffee cup"
    agent.context["recent_audio"] = "No recent audio"
    agent.context["user_state"] = "present at desk"

    response = await agent._think()
    print(f"Response: {response['message']}")
    if response.get("context_edit"):
        print(f"Context edits: {response['context_edit']}")
    print()
    print("=" * 60)
    print()

    # Test 2: Observation with reasoning trigger
    print("Test 2: Observation with Reasoning Trigger")
    print("-" * 60)
    agent.context["recent_audio"] = "User: I need to think about this problem carefully"
    agent.dialogue_history.append({
        "role": "user",
        "content": "I need to think about this problem carefully",
        "timestamp": asyncio.get_event_loop().time()
    })

    response = await agent._think()
    print(f"Response: {response['message']}")
    if response.get("context_edit"):
        print(f"Context edits: {response['context_edit']}")
    print()
    print("=" * 60)
    print()

    # Test 3: Context editing
    print("Test 3: Context Editing")
    print("-" * 60)
    print(f"Current context: {agent.context}")
    print()

    # Simulate a context edit
    edit = {"user_state": "focused on problem-solving"}
    agent._apply_context_edit(edit)
    print(f"Updated context: {agent.context}")
    print()
    print("=" * 60)
    print()

    print("[OK] Test complete!")


if __name__ == "__main__":
    asyncio.run(test_executive())
