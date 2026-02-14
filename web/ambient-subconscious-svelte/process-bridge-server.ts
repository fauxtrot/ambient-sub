/**
 * Standalone WebSocket bridge server for Python processes.
 *
 * Run this alongside the Svelte dev server:
 *   node process-bridge-server.js
 */

import { WebSocketServer } from 'ws';
import { DbConnection } from './src/generated/index.js';

const PORT = 8175; // Changed to avoid conflict with Vite dev server
const SPACETIMEDB_HOST = 'ws://127.0.0.1:3000'; // Direct connection to SpacetimeDB
const SPACETIMEDB_MODULE = 'ambient-listener';

// Track connected processes
const connections = new Map();

// SpacetimeDB connection for event relaying
let spacetimeConnection = null;

/**
 * Initialize SpacetimeDB connection and event handlers
 */
function initializeSpacetimeConnection() {
    if (spacetimeConnection) {
        console.log('[ProcessBridge] Already connected to SpacetimeDB');
        return Promise.resolve(spacetimeConnection);
    }

    console.log('[ProcessBridge] Connecting to SpacetimeDB...');

    return new Promise((resolve, reject) => {
        const conn = DbConnection.builder()
            .withUri(SPACETIMEDB_HOST)
            .withModuleName(SPACETIMEDB_MODULE)
            .onConnect((_ctx, identity, _token) => {
                console.log('[ProcessBridge] Connected to SpacetimeDB:', identity.toHexString());

                // Set up event handlers
                conn.db.frame.onInsert((_ctx, row) => {
                    relayEvent('frame', 'insert', row);
                });

                conn.db.frame.onUpdate((_ctx, oldRow, newRow) => {
                    relayEvent('frame', 'update', newRow);
                });

                conn.db.frame.onDelete((_ctx, row) => {
                    relayEvent('frame', 'delete', row);
                });

                conn.db.transcriptEntry.onInsert((_ctx, row) => {
                    console.log('[ProcessBridge] transcriptEntry.onInsert triggered:', row.entryId);
                    relayEvent('transcript_entry', 'insert', row);
                });

                conn.db.transcriptEntry.onUpdate((_ctx, oldRow, newRow) => {
                    console.log('[ProcessBridge] transcriptEntry.onUpdate triggered:', newRow.entryId);
                    relayEvent('transcript_entry', 'update', newRow);
                });

                conn.db.transcriptEntry.onDelete((_ctx, row) => {
                    console.log('[ProcessBridge] transcriptEntry.onDelete triggered:', row.entryId);
                    relayEvent('transcript_entry', 'delete', row);
                });

                conn.db.executiveContext.onInsert((_ctx, row) => {
                    relayEvent('executive_context', 'insert', row);
                });

                conn.db.executiveContext.onUpdate((_ctx, oldRow, newRow) => {
                    relayEvent('executive_context', 'update', newRow);
                });

                // Subscribe to all tables
                conn.subscriptionBuilder()
                    .onApplied(() => {
                        console.log('[ProcessBridge] Subscription applied - ready to relay events');
                        spacetimeConnection = conn;
                        resolve(conn);
                    })
                    .subscribe([
                        'SELECT * FROM frame',
                        'SELECT * FROM transcript_entry',
                        'SELECT * FROM executive_context'
                    ]);
            })
            .onConnectError((_ctx, err) => {
                console.error('[ProcessBridge] Failed to connect to SpacetimeDB:', err);
                spacetimeConnection = null;
                reject(err);
            })
            .onDisconnect((_ctx, err) => {
                console.log('[ProcessBridge] Disconnected from SpacetimeDB:', err);
                if (spacetimeConnection === conn) {
                    spacetimeConnection = null;
                }
            })
            .build();
    });
}

/**
 * Relay SpacetimeDB events to connected processes
 */
function relayEvent(table, eventType, row) {
    console.log(`[ProcessBridge] relayEvent called: ${table}.${eventType}, connections: ${connections.size}`);

    // Convert BigInt values to strings for JSON serialization
    const message = JSON.stringify({
        type: 'event',
        table,
        event_type: eventType,
        row
    }, (key, value) => {
        // Convert BigInt to string
        if (typeof value === 'bigint') {
            return value.toString();
        }
        return value;
    });

    let relayedCount = 0;

    for (const [processId, conn] of connections) {
        console.log(`[ProcessBridge] Checking process ${processId}: subscriptions=${conn.subscriptions.join(',')}, socketReady=${conn.socket.readyState === 1}`);

        // Check if process is subscribed to this table
        if (conn.subscriptions.length === 0 || conn.subscriptions.includes(table)) {
            if (conn.socket.readyState === 1) { // WebSocket.OPEN
                try {
                    conn.socket.send(message);
                    relayedCount++;
                    console.log(`[ProcessBridge] Sent to ${processId}`);
                } catch (err) {
                    console.error(`[ProcessBridge] Failed to relay to ${processId}:`, err);
                }
            } else {
                console.log(`[ProcessBridge] Socket not ready for ${processId}`);
            }
        } else {
            console.log(`[ProcessBridge] ${processId} not subscribed to ${table}`);
        }
    }

    if (relayedCount > 0) {
        console.log(`[ProcessBridge] Relayed ${table}.${eventType} to ${relayedCount} process(es)`);
    } else {
        console.log(`[ProcessBridge] No processes to relay to for ${table}.${eventType}`);
    }
}

/**
 * Start WebSocket server
 */
async function startServer() {
    // Initialize SpacetimeDB connection
    await initializeSpacetimeConnection();

    // Create WebSocket server
    const wss = new WebSocketServer({ port: PORT });

    console.log(`[ProcessBridge] WebSocket server listening on ws://localhost:${PORT}`);

    wss.on('connection', (socket) => {
        let processId = null;
        let processType = null;
        let subscriptions = [];

        console.log('[ProcessBridge] New connection');

        socket.on('message', async (data) => {
            try {
                const msg = JSON.parse(data.toString());

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
                        const conn = connections.get(processId);
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
                        const reducerFn = spacetimeConnection.reducers[reducerName];
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
        });

        socket.on('close', () => {
            if (processId) {
                console.log(`[ProcessBridge] Disconnected: ${processId}`);
                connections.delete(processId);
            }
        });

        socket.on('error', (err) => {
            console.error('[ProcessBridge] WebSocket error:', err);
        });
    });

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
}

// Start server
startServer().catch(err => {
    console.error('[ProcessBridge] Failed to start:', err);
    process.exit(1);
});
