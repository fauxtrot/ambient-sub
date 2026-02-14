/**
 * Frame Create API endpoint.
 *
 * POST /api/frame/create - Create a new frame record from webcam or screen capture
 *
 * Body: {
 *   sessionId: number,
 *   frameType: string,     // "webcam" or "screen"
 *   imagePath: string,
 *   detections: string,    // JSON array of YOLO detections
 *   reviewed: boolean,
 *   notes?: string
 * }
 */
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { DbConnection } from '../../../../generated';

const SPACETIMEDB_HOST = process.env.SPACETIMEDB_HOST || 'ws://127.0.0.1:3000';
const SPACETIMEDB_MODULE = process.env.SPACETIMEDB_MODULE || 'ambient-listener';

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();
		const { sessionId, frameType, imagePath, detections, reviewed, notes } = body;

		// Validate required fields
		if (typeof sessionId !== 'number') {
			return json({ error: 'sessionId is required and must be a number' }, { status: 400 });
		}

		if (typeof frameType !== 'string' || !['webcam', 'screen'].includes(frameType)) {
			return json({ error: 'frameType must be "webcam" or "screen"' }, { status: 400 });
		}

		if (typeof imagePath !== 'string' || imagePath.length === 0) {
			return json({ error: 'imagePath is required' }, { status: 400 });
		}

		if (typeof detections !== 'string') {
			return json({ error: 'detections must be a JSON string' }, { status: 400 });
		}

		if (typeof reviewed !== 'boolean') {
			return json({ error: 'reviewed must be a boolean' }, { status: 400 });
		}

		// Connect to SpacetimeDB and call reducer
		const result = await new Promise<{ success: boolean; error?: string }>((resolve) => {
			let conn: DbConnection | null = null;
			const timeout = setTimeout(() => {
				if (conn) conn.disconnect();
				resolve({ success: false, error: 'Connection timeout' });
			}, 10000);

			conn = DbConnection.builder()
				.withUri(SPACETIMEDB_HOST)
				.withModuleName(SPACETIMEDB_MODULE)
				.onConnect(() => {
					// Call the CreateFrame reducer
					conn!.reducers.CreateFrame({
						sessionId,
						frameType,
						imagePath,
						detections,
						reviewed,
						notes: notes ?? undefined
					});

					// Give it a moment to process, then disconnect
					setTimeout(() => {
						clearTimeout(timeout);
						if (conn) conn.disconnect();
						resolve({ success: true });
					}, 100);
				})
				.onConnectError((_ctx, err) => {
					clearTimeout(timeout);
					if (conn) conn.disconnect();
					resolve({ success: false, error: String(err) });
				})
				.build();
		});

		if (!result.success) {
			return json({ error: result.error || 'Failed to create frame' }, { status: 500 });
		}

		console.log(`[FrameCreate] Created ${frameType} frame for session ${sessionId}: ${imagePath}`);
		return json({ success: true, sessionId, frameType, imagePath });
	} catch (err) {
		console.error('[FrameCreate] Error:', err);
		return json({ error: 'Invalid request' }, { status: 400 });
	}
};
