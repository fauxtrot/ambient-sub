"""
SpacetimeDB Python client wrapper.

This provides a high-level interface for connecting to SpacetimeDB,
invoking reducers, and subscribing to table updates.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urljoin

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ReducerCall:
    """Represents a reducer invocation."""
    reducer_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubscriptionConfig:
    """Configuration for table subscriptions."""
    queries: List[str] = field(default_factory=list)
    on_insert: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_update: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_delete: Optional[Callable[[str, Any], None]] = None


class SpacetimeClient:
    """
    Python client for SpacetimeDB.

    This client provides:
    - Reducer invocation via HTTP API
    - Table subscriptions via WebSocket (future)
    - Connection pooling
    - Automatic retries
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:3000",
        module_name: str = "ambient-listener",
        auth_token: Optional[str] = None,
        svelte_api_url: str = "http://127.0.0.1:5173"
    ):
        """
        Initialize SpacetimeDB client.

        Args:
            host: SpacetimeDB host URL (e.g., "http://127.0.0.1:3000")
            module_name: Module name (e.g., "ambient-listener")
            auth_token: Optional authentication token
            svelte_api_url: Svelte API URL for reducer calls (e.g., "http://127.0.0.1:5173")
        """
        self.host = host.rstrip('/')
        self.module_name = module_name
        self.auth_token = auth_token
        self.svelte_api_url = svelte_api_url.rstrip('/')

        self._session: Optional[aiohttp.ClientSession] = None
        self._subscriptions: Dict[str, SubscriptionConfig] = {}
        self._connected = False

        # API endpoints
        self.reducer_url = f"{self.host}/database/call/{self.module_name}"
        self.query_url = f"{self.host}/database/sql/{self.module_name}"

    async def connect(self) -> None:
        """Initialize HTTP client for Svelte API communication."""
        if self._connected:
            logger.warning("Already connected")
            return

        try:
            # Create aiohttp session for API calls
            headers = {}
            if self.auth_token:
                headers['Authorization'] = f"Bearer {self.auth_token}"

            self._session = aiohttp.ClientSession(headers=headers)
            self._connected = True

            logger.info(f"HTTP client initialized (Svelte API: {self.svelte_api_url})")

        except Exception as e:
            logger.error(f"Failed to initialize HTTP client: {e}", exc_info=True)
            raise

    async def disconnect(self) -> None:
        """Close connection to SpacetimeDB."""
        if not self._connected:
            return

        try:
            if self._session:
                await self._session.close()
                self._session = None

            self._connected = False
            logger.info("Disconnected from SpacetimeDB")

        except Exception as e:
            logger.error(f"Error disconnecting from SpacetimeDB: {e}", exc_info=True)

    async def call_reducer(
        self,
        reducer_name: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke a reducer.

        Args:
            reducer_name: Name of the reducer to call
            *args: Positional arguments to pass to reducer
            **kwargs: Keyword arguments to pass to reducer

        Returns:
            Response from SpacetimeDB

        Example:
            await client.call_reducer(
                "CreateEntry",
                session_id=123,
                entry_id="abc",
                transcript="Hello world"
            )
        """
        if not self._connected or not self._session:
            raise RuntimeError("Not connected to SpacetimeDB. Call connect() first.")

        # Route certain reducers through Svelte API
        if reducer_name == "CreateFrame":
            return await self._call_svelte_api_frame_create(kwargs)

        try:
            # Prepare reducer call payload
            # SpacetimeDB expects reducer args as a JSON array
            # We convert kwargs to positional args based on C# reducer signatures
            payload = {
                "reducer": reducer_name,
                "args": self._prepare_reducer_args(reducer_name, args, kwargs)
            }

            logger.debug(f"Calling reducer {reducer_name} with payload: {payload}")

            async with self._session.post(
                self.reducer_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                result = await response.json()

                logger.debug(f"Reducer {reducer_name} returned: {result}")
                return result

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling reducer {reducer_name}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error calling reducer {reducer_name}: {e}", exc_info=True)
            raise

    async def _call_svelte_api_frame_create(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the Svelte API to create a frame.

        Args:
            kwargs: Frame creation parameters

        Returns:
            API response
        """
        try:
            # Convert snake_case to camelCase for API
            payload = {
                "sessionId": kwargs.get("session_id"),
                "frameType": kwargs.get("frame_type"),
                "imagePath": kwargs.get("image_path"),
                "detections": kwargs.get("detections", "[]"),
                "reviewed": kwargs.get("reviewed", False),
                "notes": kwargs.get("notes")
            }

            logger.debug(f"Calling Svelte API /api/frame/create with payload: {payload}")

            async with self._session.post(
                f"{self.svelte_api_url}/api/frame/create",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                result = await response.json()

                logger.debug(f"Svelte API returned: {result}")
                return result

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling Svelte API /api/frame/create: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error calling Svelte API /api/frame/create: {e}", exc_info=True)
            raise

    def _prepare_reducer_args(
        self,
        reducer_name: str,
        args: tuple,
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """
        Prepare reducer arguments for SpacetimeDB.

        SpacetimeDB expects positional arguments in order matching the
        C# reducer signature. This method converts Python kwargs to
        positional args based on known reducer signatures.

        Args:
            reducer_name: Reducer name
            args: Positional args
            kwargs: Keyword args

        Returns:
            List of positional arguments
        """
        # If positional args provided, use them directly
        if args:
            return list(args)

        # Otherwise, convert kwargs to positional args based on reducer signature
        # This mapping should match the C# reducer signatures in Lib.cs
        reducer_signatures = {
            "CreateEntry": [
                "sessionId", "entryId", "durationMs", "transcript",
                "confidence", "audioClipPath", "recordingStartMs", "recordingEndMs"
            ],
            "UpdateEntry": [
                "entryId", "transcript", "speakerName", "sentiment", "notes"
            ],
            "UpdateEntryTranscript": ["entryId", "transcript"],
            "UpdateEntrySpeaker": ["entryId", "speakerName"],
            "UpdateEntrySentiment": ["entryId", "sentiment"],
            "UpdateEntryNotes": ["entryId", "notes"],
            "MarkEntryReviewed": ["entryId", "reviewed"],
            "MarkEntryEnriched": ["entryId"],
            "DeleteEntry": ["entryId"],
            "CreateFrame": [
                "sessionId", "timestamp", "frameType", "imagePath",
                "detections", "reviewed", "notes"
            ],
            "UpdateFrameDetections": ["frameId", "detections"],
            "UpdateFrameNotes": ["frameId", "notes"],
            "CreateDiarizationSegment": [
                "entryId", "startMs", "endMs", "pyannoteLabel",
                "matchedSpeaker", "confidence", "transcriptSlice", "embedding"
            ],
            "UpdateDiarizationSegmentSpeaker": [
                "segmentId", "matchedSpeaker", "confidence"
            ],
            "CreateAssistantResponse": [
                "entryId", "quickReply", "fullReply", "reaction", "thinkingLog"
            ],
            "CreateOrUpdateSpeaker": ["name", "embedding"],
            "UpdateSpeakerThreshold": ["name", "threshold"],
            "DeleteSpeaker": ["name"],
            "StartSession": ["deviceId", "date", "mode", "audioPath"],
            "UpdateSessionStatus": ["sessionId", "status"],
            "UpdateSessionDuration": ["sessionId", "durationMs"],
            "RegisterDevice": ["deviceIdentity", "name", "platform"],
            "DeviceHeartbeat": ["deviceIdentity"],
            "UpdateDeviceStatus": ["deviceIdentity", "status"],
        }

        signature = reducer_signatures.get(reducer_name)
        if not signature:
            logger.warning(
                f"No signature mapping for reducer {reducer_name}, "
                f"using kwargs as-is. This may fail."
            )
            return list(kwargs.values())

        # Build positional args list from kwargs using signature order
        positional_args = []
        for param_name in signature:
            if param_name in kwargs:
                positional_args.append(kwargs[param_name])
            else:
                # Optional parameters can be None
                positional_args.append(None)

        return positional_args

    async def query(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute SQL query against SpacetimeDB.

        Args:
            sql: SQL query string

        Returns:
            List of rows as dictionaries

        Example:
            rows = await client.query("SELECT * FROM transcript_entry WHERE reviewed = false")
        """
        if not self._connected or not self._session:
            raise RuntimeError("Not connected to SpacetimeDB. Call connect() first.")

        try:
            payload = {"sql": sql}

            logger.debug(f"Executing query: {sql}")

            async with self._session.post(
                self.query_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                result = await response.json()

                # Extract rows from response
                # Format depends on SpacetimeDB HTTP API response structure
                rows = result.get('rows', [])
                logger.debug(f"Query returned {len(rows)} rows")
                return rows

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error executing query: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error executing query: {e}", exc_info=True)
            raise

    async def subscribe(
        self,
        subscription_config: SubscriptionConfig
    ) -> None:
        """
        Subscribe to table updates.

        Note: This is a placeholder for WebSocket-based subscriptions.
        Full implementation requires SpacetimeDB WebSocket protocol.

        Args:
            subscription_config: Subscription configuration
        """
        logger.warning("WebSocket subscriptions not yet implemented. Use polling instead.")
        # TODO: Implement WebSocket subscriptions
        # This requires implementing the SpacetimeDB WebSocket protocol
        # for now, agents can poll using query()

    async def get_entries_for_enrichment(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get transcript entries that need enrichment.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of entries needing enrichment
        """
        sql = f"""
        SELECT * FROM transcript_entry
        WHERE enriched_at IS NULL
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        return await self.query(sql)

    async def get_recent_entries(
        self,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent transcript entries.

        Args:
            session_id: Optional session ID to filter by
            limit: Maximum number of entries to return

        Returns:
            List of recent entries
        """
        if session_id:
            sql = f"""
            SELECT * FROM transcript_entry
            WHERE session_id = '{session_id}'
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
        else:
            sql = f"""
            SELECT * FROM transcript_entry
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
        return await self.query(sql)

    async def get_recent_frames(
        self,
        frame_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent frames (vision data).

        Args:
            frame_type: Optional frame type filter ("webcam", "screen")
            limit: Maximum number of frames to return

        Returns:
            List of recent frames
        """
        if frame_type:
            sql = f"""
            SELECT * FROM frame
            WHERE frame_type = '{frame_type}'
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
        else:
            sql = f"""
            SELECT * FROM frame
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
        return await self.query(sql)

    async def __aenter__(self):
        """Context manager enter."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()


# Convenience functions for common operations

async def create_entry(
    client: SpacetimeClient,
    session_id: int,
    entry_id: str,
    duration_ms: float,
    transcript: str,
    confidence: float,
    audio_clip_path: str,
    recording_start_ms: float,
    recording_end_ms: float
) -> Dict[str, Any]:
    """
    Create a new transcript entry.

    Args:
        client: SpacetimeDB client
        session_id: Session ID
        entry_id: Entry ID (timestamp_random format)
        duration_ms: Duration in milliseconds
        transcript: Transcribed text
        confidence: Confidence score (0-1)
        audio_clip_path: Path to audio clip
        recording_start_ms: Recording start time
        recording_end_ms: Recording end time

    Returns:
        Reducer response
    """
    return await client.call_reducer(
        "CreateEntry",
        sessionId=session_id,
        entryId=entry_id,
        durationMs=duration_ms,
        transcript=transcript,
        confidence=confidence,
        audioClipPath=audio_clip_path,
        recordingStartMs=recording_start_ms,
        recordingEndMs=recording_end_ms
    )


async def update_entry_speaker(
    client: SpacetimeClient,
    entry_id: str,
    speaker_name: str
) -> Dict[str, Any]:
    """
    Update the speaker for an entry.

    Args:
        client: SpacetimeDB client
        entry_id: Entry ID
        speaker_name: Speaker name

    Returns:
        Reducer response
    """
    return await client.call_reducer(
        "UpdateEntrySpeaker",
        entryId=entry_id,
        speakerName=speaker_name
    )


async def update_entry_sentiment(
    client: SpacetimeClient,
    entry_id: str,
    sentiment: str
) -> Dict[str, Any]:
    """
    Update the sentiment for an entry.

    Args:
        client: SpacetimeDB client
        entry_id: Entry ID
        sentiment: Sentiment value

    Returns:
        Reducer response
    """
    return await client.call_reducer(
        "UpdateEntrySentiment",
        entryId=entry_id,
        sentiment=sentiment
    )


async def mark_entry_enriched(
    client: SpacetimeClient,
    entry_id: str
) -> Dict[str, Any]:
    """
    Mark an entry as enriched.

    Args:
        client: SpacetimeDB client
        entry_id: Entry ID

    Returns:
        Reducer response
    """
    return await client.call_reducer(
        "MarkEntryEnriched",
        entryId=entry_id
    )


async def create_frame(
    client: SpacetimeClient,
    session_id: int,
    timestamp: float,
    frame_type: str,
    image_path: str,
    detections: str = "",
    reviewed: bool = False,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Create a new frame (vision data).

    Args:
        client: SpacetimeDB client
        session_id: Session ID
        timestamp: Frame timestamp
        frame_type: Type ("webcam", "screen")
        image_path: Path to saved frame image
        detections: JSON string of YOLO detections
        reviewed: Whether frame has been reviewed
        notes: Optional notes

    Returns:
        Reducer response
    """
    return await client.call_reducer(
        "CreateFrame",
        sessionId=session_id,
        timestamp=timestamp,
        frameType=frame_type,
        imagePath=image_path,
        detections=detections,
        reviewed=reviewed,
        notes=notes
    )
