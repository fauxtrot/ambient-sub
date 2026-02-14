"""
Agent Coordinator - manages agent lifecycle and coordination.

The coordinator is responsible for:
- Spawning and stopping agents
- Routing events between agents
- Health monitoring and heartbeats
- Graceful shutdown
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from .base import Agent, AgentStatus
from .events import Event, EventType
from ..spacetime.client import SpacetimeClient

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Coordinates multiple agents in the swarm.

    The coordinator manages agent lifecycle, routes events,
    and provides a centralized control point for the agent system.
    """

    def __init__(
        self,
        spacetime_client: Optional[SpacetimeClient] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agent coordinator.

        Args:
            spacetime_client: Shared SpacetimeDB client for all agents
            config: Configuration dictionary
        """
        self.spacetime_client = spacetime_client
        self.config = config or {}

        self._agents: Dict[str, Agent] = {}
        self._event_bus: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._tasks: Set[asyncio.Task] = set()

    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the coordinator.

        Args:
            agent: Agent to register
        """
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")

        self._agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} ({agent.__class__.__name__})")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent.

        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent instance or None
        """
        return self._agents.get(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents.

        Returns:
            List of agent info dictionaries
        """
        return [agent.get_agent_info() for agent in self._agents.values()]

    async def start_all(self) -> None:
        """Start all registered agents."""
        if self._running:
            logger.warning("Coordinator already running")
            return

        logger.info("Starting agent coordinator")
        self._running = True

        # Start event bus
        task = asyncio.create_task(self._event_bus_loop())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        # Start health monitor
        task = asyncio.create_task(self._health_monitor_loop())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        # Start all agents
        for agent_id, agent in self._agents.items():
            try:
                logger.info(f"Starting agent: {agent_id}")
                await agent.start()
            except Exception as e:
                logger.error(f"Failed to start agent {agent_id}: {e}", exc_info=True)

        logger.info(f"Agent coordinator started with {len(self._agents)} agents")

    async def stop_all(self) -> None:
        """Stop all agents gracefully."""
        if not self._running:
            logger.warning("Coordinator not running")
            return

        logger.info("Stopping agent coordinator")
        self._running = False

        # Stop all agents
        for agent_id, agent in self._agents.items():
            try:
                logger.info(f"Stopping agent: {agent_id}")
                await agent.stop()
            except Exception as e:
                logger.error(f"Error stopping agent {agent_id}: {e}", exc_info=True)

        # Cancel coordinator tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("Agent coordinator stopped")

    async def pause_agent(self, agent_id: str) -> None:
        """
        Pause an agent.

        Args:
            agent_id: Agent to pause
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        await agent.pause()
        logger.info(f"Paused agent: {agent_id}")

    async def resume_agent(self, agent_id: str) -> None:
        """
        Resume a paused agent.

        Args:
            agent_id: Agent to resume
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        await agent.resume()
        logger.info(f"Resumed agent: {agent_id}")

    async def publish_event(self, event: Event) -> None:
        """
        Publish event to all subscribed agents.

        Args:
            event: Event to publish
        """
        if not self._running:
            logger.warning("Cannot publish event, coordinator not running")
            return

        await self._event_bus.put(event)

    async def _event_bus_loop(self) -> None:
        """Event bus processing loop."""
        logger.info("Event bus started")

        while self._running:
            try:
                # Get event from bus
                try:
                    event = await asyncio.wait_for(
                        self._event_bus.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                logger.debug(f"Event bus processing: {event.event_type.value}")

                # Route event to appropriate agents
                await self._route_event(event)

            except asyncio.CancelledError:
                logger.info("Event bus loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event bus loop: {e}", exc_info=True)

        logger.info("Event bus stopped")

    async def _route_event(self, event: Event) -> None:
        """
        Route event to appropriate agents.

        Args:
            event: Event to route
        """
        # Route to specific agents based on event type
        # This is a simple broadcast for now, can be made more sophisticated

        for agent_id, agent in self._agents.items():
            if agent.status != AgentStatus.RUNNING:
                continue

            try:
                await agent.enqueue_event(event)
            except Exception as e:
                logger.error(
                    f"Error routing event to agent {agent_id}: {e}",
                    exc_info=True
                )

    async def _health_monitor_loop(self) -> None:
        """Monitor agent health and restart if needed."""
        logger.info("Health monitor started")

        while self._running:
            try:
                await asyncio.sleep(30.0)  # Check every 30 seconds

                for agent_id, agent in self._agents.items():
                    if agent.status == AgentStatus.ERROR:
                        logger.warning(
                            f"Agent {agent_id} in error state: {agent.info.last_error}"
                        )
                        # TODO: Implement auto-restart logic if configured

                    elif agent.status == AgentStatus.STOPPED and self._running:
                        logger.warning(f"Agent {agent_id} unexpectedly stopped")
                        # TODO: Implement auto-restart logic if configured

            except asyncio.CancelledError:
                logger.info("Health monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}", exc_info=True)

        logger.info("Health monitor stopped")

    def get_status(self) -> Dict[str, Any]:
        """
        Get coordinator status.

        Returns:
            Status dictionary with agent information
        """
        return {
            'running': self._running,
            'agent_count': len(self._agents),
            'agents': self.list_agents(),
            'event_queue_size': self._event_bus.qsize(),
        }

    async def __aenter__(self):
        """Context manager enter."""
        await self.start_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop_all()
