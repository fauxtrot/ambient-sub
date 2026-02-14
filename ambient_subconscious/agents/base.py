"""
Base agent classes for the agent swarm system.

All agents in the system inherit from these base classes and follow
the same lifecycle and communication patterns.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent lifecycle status."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentInfo:
    """Information about an agent."""
    agent_id: str
    agent_type: str
    status: AgentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    events_processed: int = 0
    errors: int = 0
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'stopped_at': self.stopped_at.isoformat() if self.stopped_at else None,
            'events_processed': self.events_processed,
            'errors': self.errors,
            'last_error': self.last_error,
            'metadata': self.metadata,
        }


class Agent(ABC):
    """
    Base class for all agents in the system.

    Agents are autonomous entities that:
    - Subscribe to SpacetimeDB tables/events
    - Process events asynchronously
    - Maintain their own state
    - Report health and metrics

    All agents follow a consistent lifecycle:
    1. Created (initialized with config)
    2. Starting (connecting to SpacetimeDB, subscribing)
    3. Running (processing events)
    4. Stopping (cleanup, unsubscribe)
    5. Stopped (fully shut down)
    """

    def __init__(
        self,
        agent_id: str,
        spacetime_conn: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize agent.

        Args:
            agent_id: Unique identifier for this agent instance
            spacetime_conn: SpacetimeDB connection (optional for testing)
            config: Agent-specific configuration
        """
        self.agent_id = agent_id
        self.spacetime_conn = spacetime_conn
        self.config = config or {}

        self._info = AgentInfo(
            agent_id=agent_id,
            agent_type=self.__class__.__name__,
            status=AgentStatus.CREATED,
            created_at=datetime.now()
        )

        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._tasks: Set[asyncio.Task] = set()
        self._logger = logging.getLogger(f"{__name__}.{agent_id}")

    @property
    def status(self) -> AgentStatus:
        """Get current agent status."""
        return self._info.status

    @property
    def info(self) -> AgentInfo:
        """Get agent info."""
        return self._info

    async def start(self) -> None:
        """
        Start the agent.

        This performs:
        1. Status transition to STARTING
        2. Setup (connect to SpacetimeDB, subscribe to tables)
        3. Start event processing loop
        4. Status transition to RUNNING
        """
        if self._running:
            self._logger.warning(f"Agent {self.agent_id} already running")
            return

        try:
            self._info.status = AgentStatus.STARTING
            self._logger.info(f"Starting agent {self.agent_id}")

            # Perform agent-specific setup
            await self.setup()

            # Start event processing
            self._running = True
            self._info.status = AgentStatus.RUNNING
            self._info.started_at = datetime.now()

            # Start event loop task
            task = asyncio.create_task(self._event_loop())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

            self._logger.info(f"Agent {self.agent_id} started successfully")

        except Exception as e:
            self._info.status = AgentStatus.ERROR
            self._info.errors += 1
            self._info.last_error = str(e)
            self._logger.error(f"Failed to start agent {self.agent_id}: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """
        Stop the agent gracefully.

        This performs:
        1. Status transition to STOPPING
        2. Cancel event processing
        3. Cleanup (unsubscribe, disconnect)
        4. Status transition to STOPPED
        """
        if not self._running:
            self._logger.warning(f"Agent {self.agent_id} not running")
            return

        try:
            self._info.status = AgentStatus.STOPPING
            self._logger.info(f"Stopping agent {self.agent_id}")

            # Stop accepting new events
            self._running = False

            # Cancel all tasks
            for task in self._tasks:
                task.cancel()

            # Wait for tasks to complete
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

            # Perform agent-specific cleanup
            await self.cleanup()

            self._info.status = AgentStatus.STOPPED
            self._info.stopped_at = datetime.now()

            self._logger.info(f"Agent {self.agent_id} stopped successfully")

        except Exception as e:
            self._info.status = AgentStatus.ERROR
            self._info.errors += 1
            self._info.last_error = str(e)
            self._logger.error(f"Error stopping agent {self.agent_id}: {e}", exc_info=True)
            raise

    async def pause(self) -> None:
        """Pause event processing without stopping."""
        if self._info.status != AgentStatus.RUNNING:
            self._logger.warning(f"Cannot pause agent {self.agent_id} (not running)")
            return

        self._info.status = AgentStatus.PAUSED
        self._logger.info(f"Agent {self.agent_id} paused")

    async def resume(self) -> None:
        """Resume event processing."""
        if self._info.status != AgentStatus.PAUSED:
            self._logger.warning(f"Cannot resume agent {self.agent_id} (not paused)")
            return

        self._info.status = AgentStatus.RUNNING
        self._logger.info(f"Agent {self.agent_id} resumed")

    async def _event_loop(self) -> None:
        """Internal event processing loop."""
        while self._running:
            try:
                # Wait for event with timeout to check _running flag
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Skip processing if paused
                if self._info.status == AgentStatus.PAUSED:
                    continue

                # Process event
                try:
                    await self.process(event)
                    self._info.events_processed += 1
                except Exception as e:
                    self._info.errors += 1
                    self._info.last_error = str(e)
                    self._logger.error(
                        f"Error processing event in agent {self.agent_id}: {e}",
                        exc_info=True
                    )

            except asyncio.CancelledError:
                self._logger.info(f"Event loop for agent {self.agent_id} cancelled")
                break
            except Exception as e:
                self._logger.error(
                    f"Unexpected error in event loop for agent {self.agent_id}: {e}",
                    exc_info=True
                )

    async def enqueue_event(self, event: Any) -> None:
        """
        Add event to processing queue.

        Args:
            event: Event to process
        """
        if not self._running:
            self._logger.warning(f"Cannot enqueue event, agent {self.agent_id} not running")
            return

        await self._event_queue.put(event)

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information for monitoring.

        Returns:
            Dictionary with agent status, metrics, etc.
        """
        return self._info.to_dict()

    @abstractmethod
    async def setup(self) -> None:
        """
        Perform agent-specific setup.

        This is called during start() and should:
        - Connect to external services (SpacetimeDB, etc.)
        - Subscribe to relevant tables/events
        - Initialize any required resources
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Perform agent-specific cleanup.

        This is called during stop() and should:
        - Unsubscribe from tables/events
        - Close connections
        - Release resources
        """
        pass

    @abstractmethod
    async def process(self, event: Any) -> None:
        """
        Process an event.

        This is the core agent logic - what happens when an event arrives.

        Args:
            event: Event to process (type depends on agent)
        """
        pass


class ProviderAgent(Agent):
    """
    Base class for provider agents (input layer).

    Provider agents:
    - Capture data from external sources (mic, webcam, screen, Discord)
    - Run AI models (Whisper, YOLO, Diart)
    - Publish results to SpacetimeDB via reducers
    - Generate M4 tokens for the provider system
    """

    def __init__(
        self,
        agent_id: str,
        provider_capabilities: List[tuple[str, str]],
        spacetime_conn: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize provider agent.

        Args:
            agent_id: Unique identifier
            provider_capabilities: List of (modality, capability) pairs
            spacetime_conn: SpacetimeDB connection
            config: Configuration
        """
        super().__init__(agent_id, spacetime_conn, config)
        self.provider_capabilities = provider_capabilities
        self._info.metadata['capabilities'] = provider_capabilities

    @abstractmethod
    async def capture(self) -> Any:
        """
        Capture data from source.

        Returns:
            Captured data (format depends on provider)
        """
        pass

    @abstractmethod
    async def publish_to_spacetime(self, data: Any) -> None:
        """
        Publish captured data to SpacetimeDB.

        Args:
            data: Data to publish
        """
        pass


class EnrichmentAgent(Agent):
    """
    Base class for enrichment agents (processing layer).

    Enrichment agents:
    - Subscribe to SpacetimeDB table inserts (TranscriptEntry, Frame)
    - Run enrichment models (sentiment, intent, context)
    - Update entries via reducers (UpdateEntrySentiment, etc.)
    """

    def __init__(
        self,
        agent_id: str,
        enrichment_type: str,
        spacetime_conn: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enrichment agent.

        Args:
            agent_id: Unique identifier
            enrichment_type: Type of enrichment (sentiment, intent, etc.)
            spacetime_conn: SpacetimeDB connection
            config: Configuration
        """
        super().__init__(agent_id, spacetime_conn, config)
        self.enrichment_type = enrichment_type
        self._info.metadata['enrichment_type'] = enrichment_type

    @abstractmethod
    async def enrich(self, data: Any) -> Any:
        """
        Perform enrichment on data.

        Args:
            data: Data to enrich (TranscriptEntry, Frame, etc.)

        Returns:
            Enrichment result
        """
        pass

    @abstractmethod
    async def update_spacetime(self, entry_id: str, enrichment: Any) -> None:
        """
        Update SpacetimeDB with enrichment.

        Args:
            entry_id: Entry to update
            enrichment: Enrichment data
        """
        pass


class TrainingAgent(Agent):
    """
    Base class for training agents.

    Training agents:
    - Monitor for user corrections
    - Collect labeled training data
    - Trigger model retraining
    - Perform A/B testing
    """

    def __init__(
        self,
        agent_id: str,
        model_type: str,
        spacetime_conn: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize training agent.

        Args:
            agent_id: Unique identifier
            model_type: Type of model to train (snn, hybrid, yolo, etc.)
            spacetime_conn: SpacetimeDB connection
            config: Configuration
        """
        super().__init__(agent_id, spacetime_conn, config)
        self.model_type = model_type
        self._info.metadata['model_type'] = model_type
        self._corrections: List[Any] = []

    @abstractmethod
    async def collect_correction(self, correction: Any) -> None:
        """
        Collect a user correction for training.

        Args:
            correction: Correction data
        """
        pass

    @abstractmethod
    async def should_retrain(self) -> bool:
        """
        Determine if model should be retrained.

        Returns:
            True if retraining should be triggered
        """
        pass

    @abstractmethod
    async def trigger_retraining(self) -> None:
        """Trigger model retraining with collected corrections."""
        pass
