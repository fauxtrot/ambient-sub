import { writable, derived, get } from 'svelte/store';
import { DbConnection } from '../../generated';
import type {
  TranscriptEntry,
  AudioSession,
  AudioInput,
  CaptureDevice,
  Speaker,
  CaptureConfig,
  DiarizationSegment,
  Frame
} from '$lib/types';

// Configuration
const SPACETIMEDB_HOST = import.meta.env.VITE_SPACETIMEDB_HOST || 'ws://127.0.0.1:3000';
const SPACETIMEDB_MODULE = import.meta.env.VITE_SPACETIMEDB_MODULE || 'ambient-listener';

// Connection singleton - managed outside Svelte lifecycle
let connectionInstance: DbConnection | null = null;

// Getter function to access the connection instance
export function getConnection(): DbConnection | null {
	return connectionInstance;
}

// Connection state stores
export const isConnected = writable(false);
export const isConnecting = writable(false);
export const connectionError = writable<string | null>(null);

// Data stores
export const entries = writable<TranscriptEntry[]>([]);
export const sessions = writable<AudioSession[]>([]);
export const audioInputs = writable<AudioInput[]>([]);
export const devices = writable<CaptureDevice[]>([]);
export const speakers = writable<Speaker[]>([]);
export const configs = writable<CaptureConfig[]>([]);
export const diarizationSegments = writable<DiarizationSegment[]>([]);
export const frames = writable<Frame[]>([]);

// UI state stores
export const selectedDate = writable<string>(new Date().toISOString().split('T')[0]);
export const selectedEntryId = writable<number | null>(null);
export const selectedFrameId = writable<number | null>(null);
export const drawerOpen = writable(false);

// Helper to convert SpacetimeDB Timestamp to Date
export function timestampToDate(timestamp: unknown): Date {
  if (typeof (timestamp as { toDate?: () => Date }).toDate === 'function') {
    return (timestamp as { toDate: () => Date }).toDate();
  }
  return new Date(Number(timestamp));
}

// Derived stores
export const selectedEntry = derived(
  [entries, selectedEntryId],
  ([$entries, $selectedEntryId]) => {
    if ($selectedEntryId === null) return null;
    return $entries.find(e => e.id === $selectedEntryId) ?? null;
  }
);

export const entriesForDate = derived(
  [entries, selectedDate],
  ([$entries, $selectedDate]) => {
    return $entries
      .filter(e => {
        const entryDate = timestampToDate(e.timestamp)
          .toISOString()
          .split('T')[0];
        return entryDate === $selectedDate;
      })
      .sort((a, b) => timestampToDate(b.timestamp).getTime() - timestampToDate(a.timestamp).getTime());
  }
);

export const currentDevice = derived(devices, ($devices) => {
  return $devices.length > 0 ? $devices[0] : null;
});

export const currentConfig = derived(
  [configs, currentDevice],
  ([$configs, $currentDevice]) => {
    if (!$currentDevice) return null;
    return $configs.find(c => c.deviceId === $currentDevice.id) ?? null;
  }
);

export const currentSession = derived(
  [sessions, currentDevice],
  ([$sessions, $currentDevice]) => {
    if (!$currentDevice) return null;
    return $sessions.find(s => s.deviceId === $currentDevice.id && s.status === 'recording') ?? null;
  }
);

export const inputsForDevice = derived(
  [audioInputs, currentDevice],
  ([$audioInputs, $currentDevice]) => {
    if (!$currentDevice) return [];
    return $audioInputs.filter(i => i.deviceId === $currentDevice.id);
  }
);

// Available dates based on entries
export const availableDates = derived(entries, ($entries) => {
  const dates = new Set<string>();
  $entries.forEach(e => {
    const date = timestampToDate(e.timestamp)
      .toISOString()
      .split('T')[0];
    dates.add(date);
  });
  return Array.from(dates).sort().reverse();
});

// Setup table handlers for real-time updates
function setupTableHandlers(conn: DbConnection) {
  // Transcript entries
  conn.db.transcriptEntry.onInsert((_ctx, row) => {
    entries.update(current => {
      if (current.some(e => e.id === row.id)) return current;
      return [...current, row];
    });
  });
  conn.db.transcriptEntry.onUpdate((_ctx, oldRow, newRow) => {
    entries.update(current =>
      current.map(e => (e.id === oldRow.id ? newRow : e))
    );
  });
  conn.db.transcriptEntry.onDelete((_ctx, row) => {
    entries.update(current => current.filter(e => e.id !== row.id));
    // Clear selection if deleted entry was selected
    selectedEntryId.update(id => (id === row.id ? null : id));
  });

  // Audio sessions
  conn.db.audioSession.onInsert((_ctx, row) => {
    sessions.update(current => {
      if (current.some(s => s.id === row.id)) return current;
      return [...current, row];
    });
  });
  conn.db.audioSession.onUpdate((_ctx, oldRow, newRow) => {
    sessions.update(current =>
      current.map(s => (s.id === oldRow.id ? newRow : s))
    );
  });

  // Audio inputs
  conn.db.audioInput.onInsert((_ctx, row) => {
    audioInputs.update(current => {
      if (current.some(a => a.id === row.id)) return current;
      return [...current, row];
    });
  });
  conn.db.audioInput.onUpdate((_ctx, oldRow, newRow) => {
    audioInputs.update(current =>
      current.map(a => (a.id === oldRow.id ? newRow : a))
    );
  });
  conn.db.audioInput.onDelete((_ctx, row) => {
    audioInputs.update(current => current.filter(a => a.id !== row.id));
  });

  // Capture devices
  conn.db.captureDevice.onInsert((_ctx, row) => {
    devices.update(current => {
      if (current.some(d => d.id === row.id)) return current;
      return [...current, row];
    });
  });
  conn.db.captureDevice.onUpdate((_ctx, oldRow, newRow) => {
    devices.update(current =>
      current.map(d => (d.id === oldRow.id ? newRow : d))
    );
  });

  // Speakers
  conn.db.speaker.onInsert((_ctx, row) => {
    speakers.update(current => {
      if (current.some(s => s.id === row.id)) return current;
      return [...current, row];
    });
  });
  conn.db.speaker.onUpdate((_ctx, oldRow, newRow) => {
    speakers.update(current =>
      current.map(s => (s.id === oldRow.id ? newRow : s))
    );
  });
  conn.db.speaker.onDelete((_ctx, row) => {
    speakers.update(current => current.filter(s => s.id !== row.id));
  });

  // Capture configs
  conn.db.captureConfig.onInsert((_ctx, row) => {
    configs.update(current => {
      if (current.some(c => c.deviceId === row.deviceId)) return current;
      return [...current, row];
    });
  });
  conn.db.captureConfig.onUpdate((_ctx, oldRow, newRow) => {
    configs.update(current =>
      current.map(c => (c.deviceId === oldRow.deviceId ? newRow : c))
    );
  });

  // Diarization segments
  conn.db.diarizationSegment.onInsert((_ctx, row) => {
    diarizationSegments.update(current => {
      if (current.some(s => s.id === row.id)) return current;
      return [...current, row];
    });
  });
  conn.db.diarizationSegment.onUpdate((_ctx, oldRow, newRow) => {
    diarizationSegments.update(current =>
      current.map(s => (s.id === oldRow.id ? newRow : s))
    );
  });
  conn.db.diarizationSegment.onDelete((_ctx, row) => {
    diarizationSegments.update(current => current.filter(s => s.id !== row.id));
  });

  // Frames
  conn.db.frame.onInsert((_ctx, row) => {
    frames.update(current => {
      if (current.some(f => f.id === row.id)) return current;
      return [...current, row];
    });
  });
  conn.db.frame.onUpdate((_ctx, oldRow, newRow) => {
    frames.update(current =>
      current.map(f => (f.id === oldRow.id ? newRow : f))
    );
  });
  conn.db.frame.onDelete((_ctx, row) => {
    frames.update(current => current.filter(f => f.id !== row.id));
    // Clear selection if deleted frame was selected
    selectedFrameId.update(id => (id === row.id ? null : id));
  });
}

// Connect to SpacetimeDB
export function connect() {
  if (connectionInstance || get(isConnecting)) {
    console.log('Already connected or connecting');
    return;
  }

  isConnecting.set(true);
  connectionError.set(null);

  const conn = DbConnection.builder()
    .withUri(SPACETIMEDB_HOST)
    .withModuleName(SPACETIMEDB_MODULE)
    .onConnect((_ctx, identity, _token) => {
      console.log('Connected to SpacetimeDB', identity.toHexString());
      isConnected.set(true);
      isConnecting.set(false);
      connectionError.set(null);

      // Subscribe to all tables
      conn
        .subscriptionBuilder()
        .onApplied(() => {
          console.log('Subscription applied');
          // Load initial data
          entries.set(Array.from(conn.db.transcriptEntry.iter()));
          sessions.set(Array.from(conn.db.audioSession.iter()));
          audioInputs.set(Array.from(conn.db.audioInput.iter()));
          devices.set(Array.from(conn.db.captureDevice.iter()));
          speakers.set(Array.from(conn.db.speaker.iter()));
          configs.set(Array.from(conn.db.captureConfig.iter()));
          diarizationSegments.set(Array.from(conn.db.diarizationSegment.iter()));
          frames.set(Array.from(conn.db.frame.iter()));
        })
        .subscribe([
          'SELECT * FROM transcript_entry',
          'SELECT * FROM audio_session',
          'SELECT * FROM audio_input',
          'SELECT * FROM capture_device',
          'SELECT * FROM speaker',
          'SELECT * FROM capture_config',
          'SELECT * FROM diarization_segment',
          'SELECT * FROM frame'
        ]);
    })
    .onConnectError((_ctx, err) => {
      console.error('Connection error', err);
      isConnected.set(false);
      isConnecting.set(false);
      connectionError.set(err instanceof Error ? err.message : String(err));
      connectionInstance = null;
    })
    .onDisconnect((_ctx, err) => {
      console.log('Disconnected from SpacetimeDB', err);
      if (connectionInstance === conn) {
        isConnected.set(false);
        connectionInstance = null;
      }
    })
    .build();

  connectionInstance = conn;
  setupTableHandlers(conn);
}

// Disconnect from SpacetimeDB
export function disconnect() {
  if (connectionInstance) {
    connectionInstance.disconnect();
    connectionInstance = null;
    isConnected.set(false);
  }
}

// Reducer actions
export function updateEntry(
  entryId: number,
  updates: { transcript?: string; speaker?: string; sentiment?: string; notes?: string }
) {
  if (!connectionInstance) {
    console.error('updateEntry: no connection');
    return;
  }
  connectionInstance.reducers.UpdateEntry({
    entryId,
    transcript: updates.transcript ?? undefined,
    speakerName: updates.speaker ?? undefined,
    sentiment: updates.sentiment ?? undefined,
    notes: updates.notes ?? undefined
  });
}

export function markReviewed(entryId: number, reviewed: boolean) {
  if (!connectionInstance) return;
  connectionInstance.reducers.MarkEntryReviewed({ entryId, reviewed });
}

export function deleteEntry(entryId: number) {
  if (!connectionInstance) return;
  connectionInstance.reducers.DeleteEntry({ entryId });
}

export function createSpeaker(name: string) {
  if (!connectionInstance) {
    console.error('createSpeaker: no connection');
    return;
  }
  connectionInstance.reducers.CreateOrUpdateSpeaker({ name, embedding: undefined });
}

export function assignSpeaker(entryId: number, speakerName: string) {
  if (!connectionInstance) return;
  connectionInstance.reducers.UpdateEntrySpeaker({ entryId, speakerName });
}

export function setSilenceThreshold(deviceId: number, threshold: number) {
  if (!connectionInstance) {
    console.error('setSilenceThreshold: no connection');
    return;
  }
  connectionInstance.reducers.SetSilenceThreshold({ deviceId, threshold });
}

export function setAudioInput(deviceId: number, inputIndex: number) {
  if (!connectionInstance) return;
  connectionInstance.reducers.SetAudioInput({ deviceId, inputIndex });
}

export function updateVadConfig(
  deviceId: number,
  config: {
    silenceThreshold: number;
    minSpeechDurationMs: number;
    maxSpeechDurationMs: number;
    silenceDurationMs: number;
  }
) {
  if (!connectionInstance) return;
  connectionInstance.reducers.UpdateVadConfig({
    deviceId,
    silenceThreshold: config.silenceThreshold,
    minSpeechDurationMs: config.minSpeechDurationMs,
    maxSpeechDurationMs: config.maxSpeechDurationMs,
    silenceDurationMs: config.silenceDurationMs
  });
}

export function setCapturePaused(deviceId: number, isPaused: boolean) {
  if (!connectionInstance) {
    console.error('setCapturePaused: no connection');
    return;
  }
  connectionInstance.reducers.SetCapturePaused({ deviceId, isPaused });
}

export function setCaptureMuted(deviceId: number, isMuted: boolean) {
  if (!connectionInstance) return;
  connectionInstance.reducers.SetCaptureMuted({ deviceId, isMuted });
}

export function updateSpeakerEmbeddingConfig(
  deviceId: number,
  enabled: boolean,
  model: string,
  matchThreshold: number,
  createUnknown: boolean
) {
  if (!connectionInstance) {
    console.error('updateSpeakerEmbeddingConfig: no connection');
    return;
  }
  connectionInstance.reducers.UpdateSpeakerEmbeddingConfig({
    deviceId,
    enabled,
    model,
    matchThreshold,
    createUnknown
  });
}

// Frame actions
export function updateFrameDetections(frameId: number, detections: string) {
  if (!connectionInstance) {
    console.error('updateFrameDetections: no connection');
    return;
  }
  connectionInstance.reducers.UpdateFrameDetections({ frameId, detections });
}

export function updateFrameNotes(frameId: number, notes: string | null) {
  if (!connectionInstance) {
    console.error('updateFrameNotes: no connection');
    return;
  }
  connectionInstance.reducers.UpdateFrameNotes({ frameId, notes: notes ?? undefined });
}

export function markFrameReviewed(frameId: number, reviewed: boolean) {
  if (!connectionInstance) {
    console.error('markFrameReviewed: no connection');
    return;
  }
  connectionInstance.reducers.MarkFrameReviewed({ frameId, reviewed });
}

export function deleteFrame(frameId: number) {
  if (!connectionInstance) {
    console.error('deleteFrame: no connection');
    return;
  }
  connectionInstance.reducers.DeleteFrame({ frameId });
}
