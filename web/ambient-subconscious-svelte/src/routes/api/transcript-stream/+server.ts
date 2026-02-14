/**
 * Server-Sent Events endpoint for streaming transcript entries.
 *
 * Python (or any client) can connect and receive real-time transcript updates:
 *
 *   curl -N http://localhost:5173/api/transcript-stream
 *
 * Events are sent in SSE format:
 *   event: transcript
 *   data: {"id": 1, "transcript": "hello", ...}
 */
import { DbConnection } from '../../../generated';
import type { RequestHandler } from './$types';

const SPACETIMEDB_HOST = process.env.SPACETIMEDB_HOST || 'ws://127.0.0.1:3000';
const SPACETIMEDB_MODULE = process.env.SPACETIMEDB_MODULE || 'ambient-listener';

export const GET: RequestHandler = async ({ request }) => {
	// Track seen IDs to only send new entries
	const seenIds = new Set<number>();
	let conn: DbConnection | null = null;
	let closed = false;

	const stream = new ReadableStream({
		async start(controller) {
			const encoder = new TextEncoder();

			const send = (event: string, data: object) => {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
				} catch {
					// Stream closed
					closed = true;
				}
			};

			const sendComment = (comment: string) => {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`: ${comment}\n\n`));
				} catch {
					closed = true;
				}
			};

			// Send initial comment
			sendComment('Connected to transcript stream');

			try {
				conn = DbConnection.builder()
					.withUri(SPACETIMEDB_HOST)
					.withModuleName(SPACETIMEDB_MODULE)
					.onConnect((_ctx, identity, _token) => {
						console.log('[SSE] Connected to SpacetimeDB:', identity.toHexString().slice(0, 16));
						send('connected', { identity: identity.toHexString() });

						// Subscribe to transcript_entry
						conn!
							.subscriptionBuilder()
							.onApplied(() => {
								console.log('[SSE] Subscription applied');
								send('subscribed', { table: 'transcript_entry' });

								// Mark all existing entries as seen (don't send historical data)
								for (const entry of conn!.db.transcriptEntry.iter()) {
									seenIds.add(entry.id);
								}
								console.log(`[SSE] Marked ${seenIds.size} existing entries as seen`);
							})
							.subscribe(['SELECT * FROM transcript_entry']);
					})
					.onConnectError((_ctx, err) => {
						console.error('[SSE] Connection error:', err);
						send('error', { message: String(err) });
					})
					.onDisconnect((_ctx, err) => {
						console.log('[SSE] Disconnected:', err);
						if (!closed) {
							send('disconnected', { reason: err ? String(err) : 'unknown' });
						}
					})
					.build();

				// Register for new transcript entries
				conn.db.transcriptEntry.onInsert((_ctx, row) => {
					if (seenIds.has(row.id)) return;
					seenIds.add(row.id);

					console.log(`[SSE] New entry: ${row.id} - ${row.transcript?.slice(0, 50)}...`);

					// Convert to serializable format
					send('transcript', {
						id: row.id,
						sessionId: row.sessionId,
						entryId: row.entryId,
						timestamp: row.timestamp?.toString() ?? null,
						durationMs: row.durationMs,
						transcript: row.transcript ?? null,
						confidence: row.confidence ?? null,
						speaker: row.speaker ?? null,
						sentiment: row.sentiment ?? null,
						intent: row.intent ?? null,
						audioClipPath: row.audioClipPath ?? null,
						recordingStartMs: row.recordingStartMs,
						recordingEndMs: row.recordingEndMs,
						reviewed: row.reviewed,
						notes: row.notes ?? null,
						createdAt: row.createdAt?.toString() ?? null,
						updatedAt: row.updatedAt?.toString() ?? null,
						enrichedAt: row.enrichedAt?.toString() ?? null,
					});
				});

				// Keep-alive: send comment every 30 seconds
				const keepAlive = setInterval(() => {
					if (closed) {
						clearInterval(keepAlive);
						return;
					}
					sendComment('keep-alive');
				}, 30000);

				// Handle client disconnect
				request.signal.addEventListener('abort', () => {
					console.log('[SSE] Client disconnected');
					closed = true;
					clearInterval(keepAlive);
					if (conn) {
						conn.disconnect();
						conn = null;
					}
					controller.close();
				});
			} catch (err) {
				console.error('[SSE] Error setting up stream:', err);
				send('error', { message: String(err) });
				controller.close();
			}
		},

		cancel() {
			closed = true;
			if (conn) {
				conn.disconnect();
				conn = null;
			}
		}
	});

	return new Response(stream, {
		headers: {
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			'Connection': 'keep-alive',
			'X-Accel-Buffering': 'no', // Disable nginx buffering
		}
	});
};
