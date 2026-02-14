/**
 * Frame Query API endpoint.
 *
 * GET /api/frame/query?since=<timestamp>&limit=<n>
 *
 * Returns recent frames for executive agent consumption.
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

		// Connect to SpacetimeDB and query frames
		const result = await new Promise<{ success: boolean; frames?: any[]; error?: string }>(
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
						// Query all frames
						const allFrames = Array.from(conn!.db.frame.iter());

						// Filter by timestamp if provided
						let filteredFrames = allFrames;
						if (sinceTimestamp) {
							const sinceMs = parseFloat(sinceTimestamp);
							filteredFrames = allFrames.filter((f) => {
								const frameTime = f.timestamp.toDate().getTime();
								return frameTime >= sinceMs;
							});
						}

						// Sort by timestamp desc and limit
						const recentFrames = filteredFrames
							.sort((a, b) => b.timestamp.toDate().getTime() - a.timestamp.toDate().getTime())
							.slice(0, limit)
							.map((f) => ({
								id: f.id,
								sessionId: f.sessionId,
								timestamp: f.timestamp.toDate().getTime(),
								frameType: f.frameType,
								imagePath: f.imagePath,
								detections: f.detections,
								reviewed: f.reviewed,
								notes: f.notes
							}));

						clearTimeout(timeout);
						if (conn) conn.disconnect();
						resolve({ success: true, frames: recentFrames });
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
			return json({ error: result.error || 'Failed to query frames' }, { status: 500 });
		}

		return json({ frames: result.frames || [] });
	} catch (err) {
		console.error('[FrameQuery] Error:', err);
		return json({ error: 'Invalid request' }, { status: 400 });
	}
};
