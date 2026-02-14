/**
 * WebSocket bridge for Python processes.
 *
 * Provides real-time pub/sub relay between SpacetimeDB and Python processes.
 * Each process (Observer, Executive LLM, User Input) connects via WebSocket,
 * registers itself, and receives relevant SpacetimeDB events.
 */

import type { RequestHandler } from './$types';
import { DbConnection } from '../../../generated';

// Track connected processes
const connections = new Map<string, {
    socket: WebSocket;
    processType: string;
    subscriptions: string[];
    lastHeartbeat: number;
}>();

// SpacetimeDB connection for event relaying
let spacetimeConnection: DbConnection | null = null;
const SPACETIMEDB_HOST = process.env.VITE_SPACETIMEDB_HOST || 'ws://127.0.0.1:3000';
const SPACETIMEDB_MODULE = process.env.VITE_SPACETIMEDB_MODULE || 'ambient-listener';

/**
 * Initialize SpacetimeDB connection and event handlers
 */
async function initializeSpacetimeConnection() {
    if (spacetimeConnection) return spacetimeConnection;

    console.log('[ProcessBridge] Connecting to SpacetimeDB...');

    try {
        spacetimeConnection = new DbConnection(
            SPACETIMEDB_HOST,
            SPACETIMEDB_MODULE,
            undefined // No auth token for now
        );

        await spacetimeConnection.connect();

        // Subscribe to Frame events
        spacetimeConnection.db.frame.onInsert((_ctx, row) => {
            relayEvent('frame', 'insert', row);
        });

        spacetimeConnection.db.frame.onUpdate((_ctx, oldRow, newRow) => {
            relayEvent('frame', 'update', newRow);
        });

        spacetimeConnection.db.frame.onDelete((_ctx, row) => {
            relayEvent('frame', 'delete', row);
        });

        // Subscribe to TranscriptEntry events
        spacetimeConnection.db.transcriptEntry.onInsert((_ctx, row) => {
            relayEvent('transcript_entry', 'insert', row);
        });

        spacetimeConnection.db.transcriptEntry.onUpdate((_ctx, oldRow, newRow) => {
            relayEvent('transcript_entry', 'update', newRow);
        });

        spacetimeConnection.db.transcriptEntry.onDelete((_ctx, row) => {
            relayEvent('transcript_entry', 'delete', row);
        });

        // Subscribe to ExecutiveContext events
        spacetimeConnection.db.executiveContext.onInsert((_ctx, row) => {
            relayEvent('executive_context', 'insert', row);
        });

        spacetimeConnection.db.executiveContext.onUpdate((_ctx, oldRow, newRow) => {
            relayEvent('executive_context', 'update', newRow);
        });

        console.log('[ProcessBridge] Connected to SpacetimeDB and subscribed to events');
        return spacetimeConnection;
    } catch (err) {
        console.error('[ProcessBridge] Failed to connect to SpacetimeDB:', err);
        spacetimeConnection = null;
        throw err;
    }
}

// Initialize connection immediately
initializeSpacetimeConnection().catch(err => {
    console.error('[ProcessBridge] Initialization failed:', err);
});

/**
 * WebSocket upgrade handler
 */
export const GET: RequestHandler = async ({ request }) => {
    const upgrade = request.headers.get('upgrade');

    if (upgrade !== 'websocket') {
        return new Response('Expected WebSocket', { status: 426 });
    }

    // Upgrade to WebSocket (SvelteKit/Vite dev server uses standard WebSocket API)
    // Note: This works in SvelteKit dev mode. For production, may need adapter-specific code.
    const { socket, response } = Deno.upgradeWebSocket(request);

    let processId: string | null = null;
    let processType: string | null = null;
    let subscriptions: string[] = [];

    socket.onopen = () => {
        console.log('[ProcessBridge] WebSocket connection opened');
    };

    socket.onmessage = async (event) => {
        try {
            const msg = JSON.parse(event.data);

            if (msg.type === 'register') {
                processId = msg.process_id;
                processType = msg.process_type;
                subscriptions = msg.subscriptions || [];

                connections.set(processId, {
                    socket,
                    processType,
                    subscriptions,
                    lastHeartbeat: Date.now()
                });

                socket.send(JSON.stringify({
                    type: 'registered',
                    process_id: processId,
                    status: 'ok'
                }));

                console.log(`[ProcessBridge] Registered: ${processId} (${processType}) - subscriptions: ${subscriptions.join(', ')}`);
            }
            else if (msg.type === 'heartbeat') {
                // Update last heartbeat timestamp
                if (processId && connections.has(processId)) {
                    const conn = connections.get(processId)!;
                    conn.lastHeartbeat = Date.now();
                }

                // Acknowledge heartbeat
                socket.send(JSON.stringify({ type: 'heartbeat_ack' }));
            }
            else if (msg.type === 'shutdown') {
                console.log(`[ProcessBridge] Shutdown: ${msg.process_id} - ${msg.reason}`);
                if (msg.process_id) {
                    connections.delete(msg.process_id);
                }
                socket.close();
            }
            else if (msg.type === 'reducer_call') {
                // Relay reducer call to SpacetimeDB
                if (!spacetimeConnection) {
                    socket.send(JSON.stringify({
                        type: 'error',
                        message: 'Not connected to SpacetimeDB'
                    }));
                    return;
                }

                const reducerName = msg.reducer;
                const args = msg.args;

                try {
                    // Call reducer through SpacetimeDB connection
                    const reducerFn = (spacetimeConnection as any).reducers[reducerName];
                    if (typeof reducerFn === 'function') {
                        await reducerFn(args);
                        socket.send(JSON.stringify({
                            type: 'reducer_ack',
                            reducer: reducerName
                        }));
                        console.log(`[ProcessBridge] Reducer called: ${reducerName}`);
                    } else {
                        socket.send(JSON.stringify({
                            type: 'error',
                            message: `Reducer not found: ${reducerName}`
                        }));
                    }
                } catch (err) {
                    console.error(`[ProcessBridge] Reducer error:`, err);
                    socket.send(JSON.stringify({
                        type: 'error',
                        message: `Reducer failed: ${err}`
                    }));
                }
            }
        } catch (err) {
            console.error('[ProcessBridge] Message handling error:', err);
            socket.send(JSON.stringify({
                type: 'error',
                message: `Invalid message: ${err}`
            }));
        }
    };

    socket.onclose = () => {
        if (processId) {
            console.log(`[ProcessBridge] Disconnected: ${processId}`);
            connections.delete(processId);
        }
    };

    socket.onerror = (err) => {
        console.error('[ProcessBridge] WebSocket error:', err);
    };

    return response;
};

/**
 * Relay SpacetimeDB events to connected processes.
 * Called by spacetime store when events occur.
 */
export function relayEvent(table: string, eventType: string, row: any) {
    const message = JSON.stringify({
        type: 'event',
        table,
        event_type: eventType,
        row
    });

    let relayedCount = 0;

    for (const [processId, conn] of connections) {
        // Check if process is subscribed to this table
        if (conn.subscriptions.length === 0 || conn.subscriptions.includes(table)) {
            if (conn.socket.readyState === WebSocket.OPEN) {
                try {
                    conn.socket.send(message);
                    relayedCount++;
                } catch (err) {
                    console.error(`[ProcessBridge] Failed to relay to ${processId}:`, err);
                }
            }
        }
    }

    if (relayedCount > 0) {
        console.log(`[ProcessBridge] Relayed ${table}.${eventType} to ${relayedCount} process(es)`);
    }
}

/**
 * Get connection stats (for debugging)
 */
export function getConnectionStats() {
    return {
        count: connections.size,
        processes: Array.from(connections.entries()).map(([id, conn]) => ({
            processId: id,
            processType: conn.processType,
            subscriptions: conn.subscriptions,
            lastHeartbeat: new Date(conn.lastHeartbeat).toISOString()
        }))
    };
}

// Periodic cleanup of stale connections (heartbeat timeout: 30 seconds)
setInterval(() => {
    const now = Date.now();
    const timeout = 30000; // 30 seconds

    for (const [processId, conn] of connections) {
        if (now - conn.lastHeartbeat > timeout) {
            console.log(`[ProcessBridge] Timeout: ${processId} (last heartbeat: ${Math.floor((now - conn.lastHeartbeat) / 1000)}s ago)`);
            conn.socket.close();
            connections.delete(processId);
        }
    }
}, 10000); // Check every 10 seconds
