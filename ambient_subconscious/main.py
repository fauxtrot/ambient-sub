"""
Main entry point for ambient-subconscious agent swarm.

This provides a CLI interface to start/stop/manage the agent system.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from .agents.coordinator import AgentCoordinator
from .config import Config, get_config, setup_logging
from .conversation import ConversationEngine
from .executive import ExecutiveAgent
from .pipeline.audio_pipeline import AudioPipeline
from .pipeline.video_pipeline import VideoPipeline
from .spacetime.client import SpacetimeClient

logger = logging.getLogger(__name__)


class AmbientSubconscious:
    """
    Main application class.

    Manages the agent swarm lifecycle and coordinates all components.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize application.

        Args:
            config_path: Path to config file (optional)
        """
        self.config = get_config(config_path)
        setup_logging(self.config)

        self.config.ensure_directories()

        self.spacetime_client: Optional[SpacetimeClient] = None
        self.coordinator: Optional[AgentCoordinator] = None
        self.executive_agent: Optional[ExecutiveAgent] = None
        self.conversation_engine: Optional[ConversationEngine] = None
        self.audio_pipeline: Optional[AudioPipeline] = None
        self.video_pipeline: Optional[VideoPipeline] = None
        self._shutdown_event = asyncio.Event()

    async def start(self, agent_names: Optional[list] = None) -> None:
        """
        Start the agent swarm.

        Args:
            agent_names: Optional list of specific agents to start.
                        If None, starts all enabled agents from config.
        """
        logger.info("=" * 60)
        logger.info("Starting Ambient Subconscious Agent Swarm")
        logger.info("=" * 60)

        try:
            # Connect to SpacetimeDB
            logger.info("Connecting to SpacetimeDB...")
            self.spacetime_client = SpacetimeClient(
                host=self.config.spacetimedb_host,
                module_name=self.config.spacetimedb_module,
                auth_token=self.config.spacetimedb_auth_token,
                svelte_api_url=self.config.spacetimedb_svelte_api_url
            )
            await self.spacetime_client.connect()
            logger.info("[OK] Connected to SpacetimeDB")

            # Create coordinator
            self.coordinator = AgentCoordinator(
                spacetime_client=self.spacetime_client,
                config=self.config.to_dict()
            )

            # Register agents based on config
            await self._register_agents(agent_names)

            # Start coordinator
            logger.info("Starting agent coordinator...")
            await self.coordinator.start_all()
            logger.info("[OK] Agent coordinator started")

            # Start ZMQ capture pipelines (audio + video from Godot client)
            zmq_config = self.config.to_dict().get('zmq', {})
            bind_ip = zmq_config.get('bind_address', '*')
            audio_port = zmq_config.get('audio_port', 5555)
            video_port = zmq_config.get('video_port', 5556)

            audio_cfg = self.config.to_dict().get('agents', {}).get('audio', {}).get('config', {})

            self.audio_pipeline = AudioPipeline(
                spacetime_client=self.spacetime_client,
                bind_address=f"tcp://{bind_ip}:{audio_port}",
                whisper_model=audio_cfg.get('whisper_model', 'base'),
                sample_rate=audio_cfg.get('sample_rate', 16000),
                device=audio_cfg.get('diart_device', 'cuda'),
            )
            self.audio_pipeline.start()
            logger.info(f"[OK] Audio pipeline started (ZMQ :{audio_port})")

            self.video_pipeline = VideoPipeline(
                spacetime_client=self.spacetime_client,
                bind_address=f"tcp://{bind_ip}:{video_port}",
                yolo_model=self.config.to_dict().get('agents', {}).get('webcam', {}).get('yolo_model', 'yolo11n'),
                yolo_confidence=self.config.to_dict().get('agents', {}).get('webcam', {}).get('yolo_confidence', 0.5),
                device='cuda',
            )
            self.video_pipeline.start()
            logger.info(f"[OK] Video pipeline started (ZMQ :{video_port})")

            # Start executive agent if enabled
            if self.config.executive_enabled:
                logger.info("Starting executive agent...")
                executive_config = self.config.executive_config

                self.executive_agent = ExecutiveAgent(
                    llm_config=executive_config.get('llm', {}),
                    conversational_config=executive_config.get('conversational', {}),
                    reasoning_config=executive_config.get('reasoning', {}),
                    svelte_api_url=self.config.spacetimedb_svelte_api_url,
                    update_interval=executive_config.get('update_interval_seconds', 5),
                    context_window_seconds=executive_config.get('context_window_seconds', 30),
                    stage_config=executive_config.get('stage', {}),
                    tts_config=executive_config.get('tts', {}),
                )

                # Start in background
                asyncio.create_task(self.executive_agent.start())
                logger.info("[OK] Executive agent started")

            # Start conversation engine if enabled
            conv_config = self.config.to_dict().get('conversation', {})
            if conv_config.get('enabled', False):
                logger.info("Starting conversation engine...")
                executive_config = self.config.executive_config

                florence2_config = self.config.to_dict().get('florence2', {})

                self.conversation_engine = ConversationEngine(
                    audio_receiver=self.audio_pipeline.receiver,
                    config=conv_config,
                    llm_config=executive_config.get('llm', {}),
                    tts_config=executive_config.get('tts', {}),
                    stage_config=executive_config.get('stage', {}),
                    florence2_config=florence2_config,
                )

                asyncio.create_task(self.conversation_engine.start())
                logger.info("[OK] Conversation engine started")

            # Setup signal handlers
            self._setup_signal_handlers()

            logger.info("=" * 60)
            logger.info("Ambient Subconscious is running")
            logger.info("Press Ctrl+C to stop")
            logger.info("=" * 60)

            # Log agent status
            status = self.coordinator.get_status()
            logger.info(f"Active agents: {status['agent_count']}")
            for agent_info in status['agents']:
                logger.info(f"  - {agent_info['agent_id']} ({agent_info['agent_type']}): {agent_info['status']}")

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error starting application: {e}", exc_info=True)
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the agent swarm gracefully."""
        logger.info("=" * 60)
        logger.info("Stopping Ambient Subconscious")
        logger.info("=" * 60)

        try:
            # Stop capture pipelines
            if self.video_pipeline:
                logger.info("Stopping video pipeline...")
                self.video_pipeline.stop()
                logger.info("[OK] Video pipeline stopped")

            if self.audio_pipeline:
                logger.info("Stopping audio pipeline...")
                self.audio_pipeline.stop()
                logger.info("[OK] Audio pipeline stopped")

            # Stop conversation engine
            if self.conversation_engine:
                logger.info("Stopping conversation engine...")
                await self.conversation_engine.stop()
                logger.info("[OK] Conversation engine stopped")

            # Stop executive agent
            if self.executive_agent:
                logger.info("Stopping executive agent...")
                await self.executive_agent.stop()
                logger.info("[OK] Executive agent stopped")

            # Stop coordinator
            if self.coordinator:
                logger.info("Stopping agent coordinator...")
                await self.coordinator.stop_all()
                logger.info("[OK] Agent coordinator stopped")

            # Disconnect from SpacetimeDB
            if self.spacetime_client:
                logger.info("Disconnecting from SpacetimeDB...")
                await self.spacetime_client.disconnect()
                logger.info("[OK] Disconnected from SpacetimeDB")

            logger.info("=" * 60)
            logger.info("Ambient Subconscious stopped cleanly")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def _register_agents(self, agent_names: Optional[list] = None) -> None:
        """
        Register agents with coordinator.

        Args:
            agent_names: Optional list of specific agents to register
        """
        # Get enabled agents from config
        enabled_agents = self.config.enabled_agents

        # Filter by requested agents if specified
        if agent_names:
            enabled_agents = {
                name: config
                for name, config in enabled_agents.items()
                if name in agent_names
            }

        logger.info(f"Registering {len(enabled_agents)} agents...")

        # NOTE: webcam, screen_capture, and audio agents are superseded by
        # ZMQ-based AudioPipeline and VideoPipeline (started above in start()).
        # The old agents used local device capture; the new pipelines receive
        # frames/audio from the Godot client over ZMQ.

        if not enabled_agents:
            logger.info("Note: No agents enabled in configuration")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def status(self) -> None:
        """Display current system status."""
        if not self.coordinator:
            print("Ambient Subconscious is not running")
            return

        status = self.coordinator.get_status()

        print("=" * 60)
        print("Ambient Subconscious Status")
        print("=" * 60)
        print(f"Running: {status['running']}")
        print(f"Active agents: {status['agent_count']}")
        print(f"Event queue size: {status['event_queue_size']}")
        print("\nAgents:")
        for agent_info in status['agents']:
            print(f"  - {agent_info['agent_id']}")
            print(f"    Type: {agent_info['agent_type']}")
            print(f"    Status: {agent_info['status']}")
            print(f"    Events processed: {agent_info['events_processed']}")
            print(f"    Errors: {agent_info['errors']}")
            if agent_info['last_error']:
                print(f"    Last error: {agent_info['last_error']}")
            print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ambient Subconscious Agent Swarm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start all enabled agents
  python -m ambient_subconscious start

  # Start specific agents
  python -m ambient_subconscious start --agents audio,webcam

  # Use custom config file
  python -m ambient_subconscious start --config my_config.yaml

  # Check status
  python -m ambient_subconscious status
        """
    )

    parser.add_argument(
        'command',
        choices=['start', 'stop', 'status'],
        help='Command to execute'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--agents',
        type=str,
        default=None,
        help='Comma-separated list of agents to start (e.g., audio,webcam)'
    )

    args = parser.parse_args()

    # Create application instance
    app = AmbientSubconscious(config_path=args.config)

    # Parse agent list if provided
    agent_names = None
    if args.agents:
        agent_names = [name.strip() for name in args.agents.split(',')]

    # Execute command
    try:
        if args.command == 'start':
            asyncio.run(app.start(agent_names=agent_names))
        elif args.command == 'stop':
            asyncio.run(app.stop())
        elif args.command == 'status':
            asyncio.run(app.status())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
