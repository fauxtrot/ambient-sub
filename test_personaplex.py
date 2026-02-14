"""
Standalone test script for Personaplex LLM client.

Tests nvidia/personaplex-7b-v1 model loading and text generation.
"""

import asyncio
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from ambient_subconscious.llm import PersonaplexClient


async def test_personaplex():
    """Test Personaplex client with various prompts."""

    print("=" * 60)
    print("Personaplex LLM Test")
    print("=" * 60)
    print()

    # Initialize client
    print("Initializing Personaplex client...")
    print("Model: nvidia/personaplex-7b-v1")
    print()

    client = PersonaplexClient(
        model_name="nvidia/personaplex-7b-v1",
        device=None,  # Auto-detect (cuda or cpu)
        max_length=256,
        temperature=0.7
    )

    # Check availability
    print("Checking model availability...")
    available = await client.is_available()
    print(f"[OK] Model available: {available}")
    print()

    if not available:
        print("ERROR: Personaplex model not available!")
        return

    # Test prompts
    test_prompts = [
        {
            "name": "Simple greeting",
            "prompt": "Hello, how are you today?"
        },
        {
            "name": "Observation task",
            "prompt": "You are observing a user working on their laptop. What do you notice?"
        },
        {
            "name": "Context-aware response",
            "prompt": """Current context:
- Visual: laptop, person, coffee cup
- Audio: typing sounds
- Time: afternoon

What is the user likely doing?"""
        }
    ]

    for i, test in enumerate(test_prompts, 1):
        print(f"Test {i}/{len(test_prompts)}: {test['name']}")
        print("-" * 60)
        print(f"Prompt: {test['prompt']}")
        print()

        # Generate response
        print("Generating response...")
        response = await client.complete(test['prompt'])

        print(f"Response:")
        print(response)
        print()
        print("=" * 60)
        print()

    # Cleanup
    print("Cleaning up...")
    await client.cleanup()
    print("[OK] Test complete!")


if __name__ == "__main__":
    asyncio.run(test_personaplex())
