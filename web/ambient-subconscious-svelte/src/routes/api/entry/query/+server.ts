/**
 * Entry Query API endpoint.
 *
 * GET /api/entry/query?since=<timestamp>&limit=<n>
 *
 * Returns recent transcript entries for executive agent consumption.
 */
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { DbConnection } from '../../../../generated';

const SPACETIMEDB_HOST = process.env.SPACETIMEDB_HOST || 'ws://127.0.0.1:3000';
const SPACETIMEDB_MODULE = process.env.SPACETIMEDB_MODULE || 'ambient-listener';

export const GET: RequestHandler = async ({ url }) => {
	try {
		const sinceTimestamp = url.searchParams.get('since');
		const limit = parseInt(url.searchParams.get('limit') || '10', 10);

		if (limit < 1 || limit > 100) {
			return json({ error: 'limit must be between 1 and 100' }, { status: 400 });
		}

		// Connect to SpacetimeDB and query entries
		const result = await new Promise<{ success: boolean; entries?: any[]; error?: string }>(
			(resolve) => {
				let conn: DbConnection | null = null;
				const timeout = setTimeout(() => {
					if (conn) conn.disconnect();
					resolve({ success: false, error: 'Connection timeout' });
				}, 5000);

				conn = DbConnection.builder()
					.withUri(SPACETIMEDB_HOST)
					.withModuleName(SPACETIMEDB_MODULE)
					.onConnect(() => {
						// Query all entries
						const allEntries = Array.from(conn!.db.transcriptEntry.iter());

						// Filter by timestamp if provided
						let filteredEntries = allEntries;
						if (sinceTimestamp) {
							const sinceMs = parseFloat(sinceTimestamp);
							filteredEntries = allEntries.filter((e) => {
								const entryTime = e.timestamp.toDate().getTime();
								return entryTime >= sinceMs;
							});
						}

						// Sort by timestamp desc and limit
						const recentEntries = filteredEntries
							.sort((a, b) => b.timestamp.toDate().getTime() - a.timestamp.toDate().getTime())
							.slice(0, limit)
							.map((e) => ({
								id: e.id,
								sessionId: e.sessionId,
								timestamp: e.timestamp.toDate().getTime(),
								transcript: e.transcript,
								speaker: e.speaker,
								sentiment: e.sentiment,
								confidence: e.confidence,
								durationMs: e.durationMs,
								reviewed: e.reviewed,
								notes: e.notes
							}));

						clearTimeout(timeout);
						if (conn) conn.disconnect();
						resolve({ success: true, entries: recentEntries });
					})
					.onConnectError((_ctx, err) => {
						clearTimeout(timeout);
						if (conn) conn.disconnect();
						resolve({ success: false, error: String(err) });
					})
					.build();
			}
		);

		if (!result.success) {
			return json({ error: result.error || 'Failed to query entries' }, { status: 500 });
		}

		return json({ entries: result.entries || [] });
	} catch (err) {
		console.error('[EntryQuery] Error:', err);
		return json({ error: 'Invalid request' }, { status: 400 });
	}
};
