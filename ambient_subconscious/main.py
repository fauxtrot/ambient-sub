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
from .executive import ExecutiveAgent
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
                    context_window_seconds=executive_config.get('context_window_seconds', 30)
                )

                # Start in background
                asyncio.create_task(self.executive_agent.start())
                logger.info("[OK] Executive agent started")

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

        # Register webcam agent
        if 'webcam' in enabled_agents:
            from .agents.vision.webcam_agent import WebcamAgent
            webcam_config = enabled_agents['webcam']

            webcam_agent = WebcamAgent(
                agent_id='webcam-agent-1',
                spacetime_client=self.spacetime_client,
                camera_index=webcam_config.get('camera_index', 0),
                fps=webcam_config.get('fps', 2.0),
                resolution=tuple(webcam_config.get('resolution', [1280, 720])),
                yolo_model=webcam_config.get('yolo_model', 'yolo26n'),
                yolo_confidence=webcam_config.get('yolo_confidence', 0.5),
                output_dir=self.config.data_dir / "frames",
                warmup_frames=webcam_config.get('warmup_frames', 5)
            )
            self.coordinator.register_agent(webcam_agent)
            logger.info("[OK] Registered webcam agent")

        # Register screen capture agent
        if 'screen_capture' in enabled_agents:
            from .agents.vision.screen_capture_agent import ScreenCaptureAgent
            screen_config = enabled_agents['screen_capture']

            screen_agent = ScreenCaptureAgent(
                agent_id='screen-capture-agent-1',
                spacetime_client=self.spacetime_client,
                fps=screen_config.get('fps', 1.0),
                yolo_model=screen_config.get('yolo_model', 'yolo26n'),
                yolo_confidence=screen_config.get('yolo_confidence', 0.5),
                output_dir=self.config.data_dir / "frames",
                capture_active_window_only=screen_config.get('capture_active_window_only', False),
                privacy_mode_apps=screen_config.get('privacy_mode_apps', [])
            )
            self.coordinator.register_agent(screen_agent)
            logger.info("[OK] Registered screen capture agent")

        # Register audio agents
        if 'audio' in enabled_agents:
            from .agents.audio.audio_agent import AudioAgent
            audio_config = enabled_agents['audio']

            # Support multiple audio sources
            audio_sources = audio_config.get('sources', [
                {'device_id': None, 'label': 'default'}
            ])

            for idx, source in enumerate(audio_sources):
                audio_agent = AudioAgent(
                    agent_id=f"audio-agent-{source['label']}-{idx}",
                    spacetime_client=self.spacetime_client,
                    device_id=source.get('device_id'),
                    device_label=source.get('label', f'mic{idx}'),
                    sample_rate=audio_config.get('sample_rate', 16000),
                    whisper_model=audio_config.get('whisper_model', 'base'),
                    output_dir=self.config.data_dir / "audio",
                    session_id=audio_config.get('session_id', 1),
                    # VAD settings
                    vad_energy_threshold=audio_config.get('vad_energy_threshold', 0.01),
                    vad_ema_alpha=audio_config.get('vad_ema_alpha', 0.3),
                    vad_min_utterance_duration=audio_config.get('vad_min_utterance_duration', 0.5),
                    vad_max_utterance_duration=audio_config.get('vad_max_utterance_duration', 30.0),
                    vad_silence_duration=audio_config.get('vad_silence_duration', 0.8),
                    # Filter settings
                    min_confidence=audio_config.get('min_confidence', -0.5),
                    reject_fillers=audio_config.get('reject_fillers', True)
                )
                self.coordinator.register_agent(audio_agent)
                logger.info(f"[OK] Registered audio agent: {source['label']} (device {source.get('device_id', 'default')})")

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
