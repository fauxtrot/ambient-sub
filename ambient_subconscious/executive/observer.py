"""
Observer Process - Aggregates recent frames and entries into executive context.

Subscribes to Frame and TranscriptEntry events from SpacetimeDB via Svelte bridge,
maintains sliding window of recent data, and aggregates into ExecutiveContext.
"""

import asyncio
import json
import logging
import time
from collections import deque, Counter
from typing import Dict, List, Any, Optional
import websockets

logger = logging.getLogger(__name__)


def timestamp_to_unix(ts: Any) -> float:
    """
    Convert SpacetimeDB timestamp to Unix time (seconds).

    SpacetimeDB timestamps come as either:
    - A number (already Unix time)
    - A string (Unix time as string)
    - A dict with 'seconds' and 'nanos' fields
    - A dict with '__timestamp_micros_since_unix_epoch__' field (microseconds)
    """
    if isinstance(ts, (int, float)):
        return float(ts)
    elif isinstance(ts, str):
        try:
            return float(ts)
        except ValueError:
            return time.time()
    elif isinstance(ts, dict):
        # Check for microseconds format first
        if '__timestamp_micros_since_unix_epoch__' in ts:
            micros = ts['__timestamp_micros_since_unix_epoch__']
            if isinstance(micros, str):
                micros = int(micros)
            return float(micros) / 1_000_000  # Convert microseconds to seconds

        # SpacetimeDB timestamp format: {seconds: X, nanos: Y}
        seconds = ts.get('seconds', ts.get('secs', 0))
        nanos = ts.get('nanos', ts.get('nsecs', 0))
        if isinstance(seconds, str):
            seconds = int(seconds)
        if isinstance(nanos, str):
            nanos = int(nanos)
        return float(seconds) + float(nanos) / 1_000_000_000
    else:
        # Fallback to current time
        return time.time()


class Observer:
    """
    Observes Frame and TranscriptEntry events from SpacetimeDB via Svelte bridge.
    Maintains sliding window of recent data and aggregates into ExecutiveContext.
    Uses baseline tracking and delta detection to send updates only when significant changes occur.
    """

    def __init__(
        self,
        process_id: str = "observer-001",
        bridge_url: str = "ws://localhost:8175",
        window_seconds: int = 30,
        update_interval: float = 5.0,
        visual_delta_threshold: float = 0.2,
        sound_delta_threshold: float = 0.0,
        ema_alpha: float = 0.1
    ):
        """
        Initialize Observer process.

        Args:
            process_id: Unique identifier for this process
            bridge_url: WebSocket URL for Svelte bridge
            window_seconds: Time window for aggregation (seconds)
            update_interval: How often to aggregate and update (seconds)
            visual_delta_threshold: Visual delta threshold (0.0-1.0, default 0.2 = 20%)
            sound_delta_threshold: Sound delta threshold (default 0.0 = any new entry)
            ema_alpha: EMA smoothing factor (default 0.1)
        """
        self.process_id = process_id
        self.bridge_url = bridge_url
        self.window_seconds = window_seconds
        self.update_interval = update_interval

        # Sliding windows (deque with maxlen based on expected rate)
        self.recent_frames = deque(maxlen=100)
        self.recent_entries = deque(maxlen=50)

        # Baseline tracking (exponential moving average)
        self.baseline_object_count = 0.0  # EMA of object count
        self.baseline_objects = Counter()  # EMA of object types
        self.baseline_audio_rate = 0.0     # EMA of entries per minute

        # Delta thresholds (configurable)
        self.visual_delta_threshold = visual_delta_threshold
        self.sound_delta_threshold = sound_delta_threshold
        self.ema_alpha = ema_alpha

        # Manual trigger
        self.manual_trigger_requested = False

        # Last update tracking
        self.last_update_time = 0.0
        self.last_context: Optional[Dict[str, Any]] = None
        self.last_audio_count = 0

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self._aggregation_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self):
        """Connect to Svelte bridge and start observing."""
        self.running = True
        logger.info(f"Observer {self.process_id} starting...")
        logger.info(f"Connecting to bridge at {self.bridge_url}")

        try:
            async with websockets.connect(self.bridge_url) as ws:
                self.websocket = ws

                # Register with bridge
                await ws.send(json.dumps({
                    "type": "register",
                    "process_id": self.process_id,
                    "process_type": "observer",
                    "subscriptions": ["frame", "transcript_entry", "executive_context"]
                }))

                # Wait for registration confirmation
                msg = await ws.recv()
                response = json.loads(msg)
                if response["type"] == "registered":
                    logger.info(f"Observer registered: {response['process_id']}")

                # Start background tasks
                self._aggregation_task = asyncio.create_task(self._aggregation_loop())
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                # Main event loop
                try:
                    while self.running:
                        msg = await ws.recv()
                        event = json.loads(msg)

                        logger.debug(f"Received message type: {event.get('type')}")

                        if event["type"] == "event":
                            logger.info(f"Received event: {event.get('table')}.{event.get('event_type')}")
                            await self._handle_event(event)
                        elif event["type"] == "heartbeat_ack":
                            pass  # Acknowledged
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Connection to bridge closed")
                finally:
                    if self._aggregation_task:
                        self._aggregation_task.cancel()
                    if self._heartbeat_task:
                        self._heartbeat_task.cancel()

        except Exception as e:
            logger.error(f"Observer error: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        """Gracefully shutdown and notify bridge."""
        self.running = False
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({
                    "type": "shutdown",
                    "process_id": self.process_id,
                    "reason": "Graceful shutdown"
                }))
            except:
                pass
        logger.info(f"Observer {self.process_id} stopped")

    async def _handle_event(self, event: Dict[str, Any]):
        """Handle incoming SpacetimeDB event."""
        table = event["table"]
        event_type = event["event_type"]
        row = event["row"]

        if table == "frame" and event_type == "insert":
            # Normalize timestamp to Unix time (float)
            if "timestamp" in row:
                row["timestamp"] = timestamp_to_unix(row["timestamp"])
            else:
                row["timestamp"] = time.time()

            self.recent_frames.append(row)
            logger.debug(f"Frame added: {row.get('id', 'unknown')}")

        elif table == "transcript_entry" and event_type == "insert":
            # Normalize timestamp to Unix time (float)
            if "timestamp" in row:
                orig_ts = row["timestamp"]
                row["timestamp"] = timestamp_to_unix(row["timestamp"])
                logger.info(f"Entry timestamp: orig={orig_ts}, normalized={row['timestamp']}, current={time.time()}")
            else:
                row["timestamp"] = time.time()
                logger.info(f"Entry has no timestamp, using current: {row['timestamp']}")

            self.recent_entries.append(row)
            logger.info(f"Entry added to deque: id={row.get('id', 'unknown')}, entryId={row.get('entryId', 'unknown')}, deque_size={len(self.recent_entries)}")
            logger.info(f"Entry keys: {list(row.keys())}")

        elif table == "executive_context" and event_type == "update":
            # Receive manual triggers and scheduled check-ins from Executive
            if row.get("agentState") == "wake":
                logger.info("Manual wake trigger received")
                self.manual_trigger_requested = True

            # Store next check-in time
            if "nextCheckIn" in row and row["nextCheckIn"] is not None:
                self.last_context = self.last_context or {}
                self.last_context["next_check_in"] = row["nextCheckIn"]
                logger.info(f"Scheduled check-in at {row['nextCheckIn']} (in {row['nextCheckIn'] - time.time():.0f}s)")

    def _should_send_update(self, context: Dict[str, Any]) -> bool:
        """Determine if update should be sent based on deltas."""
        current_time = time.time()

        # Parse current context
        visual_objects = json.loads(context["recent_visual"])
        audio_entries = json.loads(context["recent_audio"])

        # Check manual trigger
        if self.manual_trigger_requested:
            logger.info("Manual trigger - sending update")
            self.manual_trigger_requested = False
            return True

        # Check scheduled check-in (from ExecutiveContext.nextCheckIn)
        if self.last_context and "next_check_in" in self.last_context:
            next_check_in = self.last_context.get("next_check_in")
            if next_check_in and current_time >= next_check_in:
                logger.info(f"Scheduled check-in reached - sending update")
                return True

        # Visual delta: significant change in object count or new object type
        current_object_count = len(visual_objects)

        if self.baseline_object_count == 0:
            # First aggregation - always send
            visual_delta = True
        else:
            count_delta = abs(current_object_count - self.baseline_object_count) / (self.baseline_object_count + 1e-6)
            new_objects = set(visual_objects) - set(self.baseline_objects.keys())
            visual_delta = (count_delta > self.visual_delta_threshold) or (len(new_objects) > 0)

        if visual_delta:
            count_delta_str = f"{count_delta:.2f}" if self.baseline_object_count > 0 else "N/A"
            new_objects_str = str(new_objects) if self.baseline_object_count > 0 else "initial"
            logger.info(f"Visual delta detected: count_delta={count_delta_str}, new_objects={new_objects_str}")

        # Sound delta: new audio entry
        current_audio_count = len(audio_entries)
        sound_delta = current_audio_count > self.last_audio_count

        if sound_delta:
            logger.info(f"Sound delta detected: {current_audio_count} entries (was {self.last_audio_count})")

        # Send update if any delta exceeded threshold
        return visual_delta or sound_delta

    def _update_baseline(self, context: Dict[str, Any]):
        """Update baseline using exponential moving average."""
        visual_objects = json.loads(context["recent_visual"])
        audio_entries = json.loads(context["recent_audio"])

        # Update baseline object count (EMA)
        current_count = len(visual_objects)
        self.baseline_object_count = (1 - self.ema_alpha) * self.baseline_object_count + self.ema_alpha * current_count

        # Update baseline objects (EMA of frequencies)
        for obj in visual_objects:
            self.baseline_objects[obj] = (1 - self.ema_alpha) * self.baseline_objects.get(obj, 0) + self.ema_alpha

        # Decay objects not seen
        for obj in list(self.baseline_objects.keys()):
            if obj not in visual_objects:
                self.baseline_objects[obj] *= (1 - self.ema_alpha)
                if self.baseline_objects[obj] < 0.1:
                    del self.baseline_objects[obj]

        # Update last audio count
        self.last_audio_count = len(audio_entries)

        # Update baseline audio rate (entries per minute)
        # (simplified: just track recent rate)
        time_delta = time.time() - self.last_update_time if self.last_update_time > 0 else 60.0
        current_rate = len(audio_entries) / (time_delta / 60.0) if time_delta > 0 else 0.0
        self.baseline_audio_rate = (1 - self.ema_alpha) * self.baseline_audio_rate + self.ema_alpha * current_rate

        logger.debug(f"Updated baseline: objects={self.baseline_object_count:.1f}, rate={self.baseline_audio_rate:.2f}/min")

    async def _aggregation_loop(self):
        """Periodically aggregate recent data and update ExecutiveContext."""
        while self.running:
            try:
                await asyncio.sleep(self.update_interval)

                # Aggregate context
                context = self._aggregate_context()

                # Check if update should be sent (delta detection)
                if self._should_send_update(context):
                    # Build baseline data
                    baseline_visual = json.dumps({
                        "avg_objects": self.baseline_object_count,
                        "common_objects": [obj for obj, _ in self.baseline_objects.most_common(5)]
                    })
                    baseline_audio = json.dumps({
                        "avg_entries_per_min": self.baseline_audio_rate
                    })

                    # Update ExecutiveContext via reducer
                    await self._update_executive_context(context, baseline_visual, baseline_audio)

                    self.last_update_time = time.time()
                    self.last_context = context
                else:
                    logger.debug("No significant delta - skipping update")

                # Always update baseline (even if we didn't send an update)
                self._update_baseline(context)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation error: {e}", exc_info=True)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to bridge to keep connection alive."""
        while self.running:
            try:
                await asyncio.sleep(15.0)  # Send heartbeat every 15 seconds (half of 30s timeout)

                if self.websocket:
                    await self.websocket.send(json.dumps({
                        "type": "heartbeat",
                        "process_id": self.process_id
                    }))
                    logger.debug("Sent heartbeat to bridge")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")

    def _aggregate_context(self) -> Dict[str, str]:
        """Aggregate recent frames and entries into context."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        logger.debug(f"Aggregation starting: deque has {len(self.recent_entries)} entries, cutoff_time={cutoff_time}, current_time={current_time}")

        # Filter by time window
        recent_frames = [f for f in self.recent_frames if f.get("timestamp", 0) >= cutoff_time]
        recent_entries = [e for e in self.recent_entries if e.get("timestamp", 0) >= cutoff_time]

        logger.debug(f"After timestamp filter: {len(recent_entries)} entries (from {len(self.recent_entries)} total)")

        # Aggregate visual: top detected objects
        all_objects = []
        for frame in recent_frames:
            try:
                detections_str = frame.get("detections", "[]")
                if isinstance(detections_str, str):
                    detections = json.loads(detections_str)
                else:
                    detections = detections_str

                all_objects.extend([d.get("class", d.get("label", "unknown")) for d in detections])
            except Exception as e:
                logger.debug(f"Error parsing detections: {e}")
                pass

        object_counts = Counter(all_objects)
        top_objects = [obj for obj, _ in object_counts.most_common(5)]
        recent_visual = json.dumps(top_objects)

        # Aggregate audio: recent transcripts
        recent_audio_list = []
        logger.debug(f"Processing last 3 entries from {len(recent_entries)} filtered entries")

        for idx, entry in enumerate(list(recent_entries)[-3:]):  # Last 3 entries
            logger.debug(f"Entry {idx}: keys={list(entry.keys())}, timestamp={entry.get('timestamp')}")

            # Try both capitalized and lowercase field names
            speaker_raw = entry.get("speaker") or entry.get("Speaker")
            transcript_raw = entry.get("transcript") or entry.get("Transcript")

            logger.debug(f"Entry {idx} raw values: speaker_raw={speaker_raw} (type={type(speaker_raw).__name__}), transcript_raw={transcript_raw} (type={type(transcript_raw).__name__})")

            # Handle SpacetimeDB Option types (could be {"some": value} or just value)
            speaker = "Unknown"
            if isinstance(speaker_raw, dict) and "some" in speaker_raw:
                speaker = speaker_raw["some"]
            elif isinstance(speaker_raw, str):
                speaker = speaker_raw

            transcript = ""
            if isinstance(transcript_raw, dict) and "some" in transcript_raw:
                transcript = transcript_raw["some"]
            elif isinstance(transcript_raw, str):
                transcript = transcript_raw

            logger.debug(f"Entry {idx} parsed: speaker='{speaker}', transcript='{transcript}'")

            if transcript:
                recent_audio_list.append({"speaker": speaker, "text": transcript})
            else:
                logger.debug(f"Entry {idx} SKIPPED: empty transcript")

        logger.debug(f"Final recent_audio_list: {len(recent_audio_list)} entries")
        recent_audio = json.dumps(recent_audio_list)

        # Infer user state (simple heuristic for now)
        user_state = "idle"
        if top_objects:
            if "person" in top_objects:
                user_state = "present"
            if "laptop" in top_objects or "monitor" in top_objects or "computer" in top_objects:
                user_state = "working"
        if recent_entries:
            user_state = "speaking"

        logger.info(f"Aggregated context: visual={len(top_objects)} objects, audio={len(recent_audio_list)} entries, state={user_state}")

        return {
            "recent_visual": recent_visual,
            "recent_audio": recent_audio,
            "user_state": user_state,
            "agent_state": "active",
            "notes": None
        }

    async def _update_executive_context(self, context: Dict[str, str], baseline_visual: str, baseline_audio: str):
        """Call UpdateExecutiveContext reducer via bridge."""
        if not self.websocket:
            logger.warning("WebSocket not connected, cannot update context")
            return

        try:
            await self.websocket.send(json.dumps({
                "type": "reducer_call",
                "process_id": self.process_id,
                "reducer": "UpdateExecutiveContext",
                "args": {
                    "recentVisual": context["recent_visual"],
                    "recentAudio": context["recent_audio"],
                    "baselineVisual": baseline_visual,
                    "baselineAudio": baseline_audio,
                    "userState": context["user_state"],
                    "agentState": context["agent_state"],
                    "nextCheckIn": None,  # Observer doesn't set this, Executive does
                    "notes": context.get("notes")
                }
            }))
            logger.info("Updated ExecutiveContext (delta triggered)")
        except Exception as e:
            logger.error(f"Failed to update context: {e}")


def setup_logging(log_file: str, debug: bool = False):
    """Configure logging to both file and console."""
    import os

    # Create logs directory
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)

    # Console handler (for terminal viewing)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)


async def main():
    """Main entry point for Observer process."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Observer process for ambient-subconscious")
    parser.add_argument('--bridge-url', default='ws://localhost:8175',
                        help='WebSocket URL for bridge server')
    parser.add_argument('--window', type=int, default=30,
                        help='Time window for aggregation (seconds)')
    parser.add_argument('--interval', type=float, default=5.0,
                        help='Update interval (seconds)')
    parser.add_argument('--log-file', default='logs/observer.log',
                        help='Log file path')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--visual-delta', type=float, default=0.2,
                        help='Visual delta threshold (0.0-1.0) - percentage change to trigger update')
    parser.add_argument('--sound-delta', type=float, default=0.0,
                        help='Sound delta threshold - minimum new entries to trigger update')
    parser.add_argument('--ema-alpha', type=float, default=0.1,
                        help='EMA smoothing factor for baseline tracking (0.0-1.0)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file, args.debug)

    observer = Observer(
        bridge_url=args.bridge_url,
        window_seconds=args.window,
        update_interval=args.interval,
        visual_delta_threshold=args.visual_delta,
        sound_delta_threshold=args.sound_delta,
        ema_alpha=args.ema_alpha
    )

    try:
        await observer.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await observer.stop()


if __name__ == "__main__":
    asyncio.run(main())
