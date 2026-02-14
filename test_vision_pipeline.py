"""
Test vision pipeline end-to-end.

This script tests the webcam and screen capture agents with YOLO detection
and SpacetimeDB storage.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ambient_subconscious.agents.vision.webcam_agent import WebcamAgent
from ambient_subconscious.agents.vision.screen_capture_agent import ScreenCaptureAgent
from ambient_subconscious.spacetime.client import SpacetimeClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_webcam_agent():
    """Test webcam agent."""
    logger.info("=" * 60)
    logger.info("Testing Webcam Agent")
    logger.info("=" * 60)

    # Create SpacetimeDB client
    client = SpacetimeClient(
        host="http://127.0.0.1:3000",
        module_name="ambient-listener"
    )

    try:
        # Connect to SpacetimeDB
        logger.info("Connecting to SpacetimeDB...")
        await client.connect()
        logger.info("[OK] Connected to SpacetimeDB")

        # Create webcam agent
        agent = WebcamAgent(
            agent_id="test-webcam-agent",
            spacetime_client=client,
            camera_index=0,
            fps=1.0,  # 1 frame per second for testing
            resolution=(640, 480),  # Lower resolution for faster testing
            yolo_model="yolo26n",
            yolo_confidence=0.5,
            output_dir="data/frames",
            warmup_frames=3
        )

        # Start agent
        logger.info("Starting webcam agent...")
        await agent.start()

        # Capture for 5 seconds
        logger.info("Capturing frames for 5 seconds...")
        await asyncio.sleep(5)

        # Stop agent
        logger.info("Stopping webcam agent...")
        await agent.stop()

        # Get agent info
        info = agent.get_agent_info()
        logger.info(f"Agent info: {info}")

        logger.info("[OK] Webcam agent test completed")

    except Exception as e:
        logger.error(f"Error testing webcam agent: {e}", exc_info=True)
        raise
    finally:
        await client.disconnect()


async def test_screen_capture_agent():
    """Test screen capture agent."""
    logger.info("=" * 60)
    logger.info("Testing Screen Capture Agent")
    logger.info("=" * 60)

    # Create SpacetimeDB client
    client = SpacetimeClient(
        host="http://127.0.0.1:3000",
        module_name="ambient-listener"
    )

    try:
        # Connect to SpacetimeDB
        logger.info("Connecting to SpacetimeDB...")
        await client.connect()
        logger.info("[OK] Connected to SpacetimeDB")

        # Create screen capture agent
        agent = ScreenCaptureAgent(
            agent_id="test-screen-agent",
            spacetime_client=client,
            fps=0.5,  # 1 frame every 2 seconds for testing
            yolo_model="yolo26n",
            yolo_confidence=0.5,
            output_dir="data/frames",
            capture_active_window_only=False,
            privacy_mode_apps=[]
        )

        # Start agent
        logger.info("Starting screen capture agent...")
        await agent.start()

        # Capture for 5 seconds
        logger.info("Capturing frames for 5 seconds...")
        await asyncio.sleep(5)

        # Stop agent
        logger.info("Stopping screen capture agent...")
        await agent.stop()

        # Get agent info
        info = agent.get_agent_info()
        logger.info(f"Agent info: {info}")

        logger.info("[OK] Screen capture agent test completed")

    except Exception as e:
        logger.error(f"Error testing screen capture agent: {e}", exc_info=True)
        raise
    finally:
        await client.disconnect()


async def test_both_agents():
    """Test both agents running simultaneously."""
    logger.info("=" * 60)
    logger.info("Testing Both Agents Simultaneously")
    logger.info("=" * 60)

    # Create SpacetimeDB client
    client = SpacetimeClient(
        host="http://127.0.0.1:3000",
        module_name="ambient-listener"
    )

    try:
        # Connect to SpacetimeDB
        logger.info("Connecting to SpacetimeDB...")
        await client.connect()
        logger.info("[OK] Connected to SpacetimeDB")

        # Create agents
        webcam_agent = WebcamAgent(
            agent_id="test-webcam-agent",
            spacetime_client=client,
            camera_index=0,
            fps=1.0,
            resolution=(640, 480),
            yolo_model="yolo26n",
            yolo_confidence=0.5,
            output_dir="data/frames",
            warmup_frames=3
        )

        screen_agent = ScreenCaptureAgent(
            agent_id="test-screen-agent",
            spacetime_client=client,
            fps=0.5,
            yolo_model="yolo26n",
            yolo_confidence=0.5,
            output_dir="data/frames",
            capture_active_window_only=False,
            privacy_mode_apps=[]
        )

        # Start both agents
        logger.info("Starting both agents...")
        await webcam_agent.start()
        await screen_agent.start()

        # Capture for 10 seconds
        logger.info("Capturing frames for 10 seconds...")
        await asyncio.sleep(10)

        # Stop both agents
        logger.info("Stopping both agents...")
        await webcam_agent.stop()
        await screen_agent.stop()

        # Get agent info
        webcam_info = webcam_agent.get_agent_info()
        screen_info = screen_agent.get_agent_info()

        logger.info(f"Webcam agent info: {webcam_info}")
        logger.info(f"Screen agent info: {screen_info}")

        logger.info("[OK] Both agents test completed")

    except Exception as e:
        logger.error(f"Error testing both agents: {e}", exc_info=True)
        raise
    finally:
        await client.disconnect()


async def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test vision pipeline")
    parser.add_argument(
        'mode',
        choices=['webcam', 'screen', 'both'],
        help='Which agent(s) to test'
    )
    args = parser.parse_args()

    try:
        if args.mode == 'webcam':
            await test_webcam_agent()
        elif args.mode == 'screen':
            await test_screen_capture_agent()
        elif args.mode == 'both':
            await test_both_agents()

        logger.info("=" * 60)
        logger.info("All tests completed successfully!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
