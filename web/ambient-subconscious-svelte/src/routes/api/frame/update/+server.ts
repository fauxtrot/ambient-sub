/**
 * Frame Update API endpoint.
 *
 * POST /api/frame/update - Update a frame's detections or notes
 *
 * Body: {
 *   frameId: number,
 *   detections?: string,    // JSON array of updated detections
 *   notes?: string,
 *   reviewed?: boolean
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
		const { frameId, detections, notes, reviewed } = body;

		// Validate required fields
		if (typeof frameId !== 'number') {
			return json({ error: 'frameId is required and must be a number' }, { status: 400 });
		}

		// At least one field should be provided
		if (detections === undefined && notes === undefined && reviewed === undefined) {
			return json({ error: 'At least one field to update is required' }, { status: 400 });
		}

		// Connect to SpacetimeDB and call appropriate reducers
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
					// Call appropriate reducers based on what's being updated
					if (detections !== undefined) {
						conn!.reducers.UpdateFrameDetections({
							frameId,
							detections
						});
					}

					if (notes !== undefined) {
						conn!.reducers.UpdateFrameNotes({
							frameId,
							notes: notes ?? undefined
						});
					}

					if (reviewed !== undefined) {
						conn!.reducers.MarkFrameReviewed({
							frameId,
							reviewed
						});
					}

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
			return json({ error: result.error || 'Failed to update frame' }, { status: 500 });
		}

		console.log(`[FrameUpdate] Updated frame ${frameId}`);
		return json({ success: true, frameId });
	} catch (err) {
		console.error('[FrameUpdate] Error:', err);
		return json({ error: 'Invalid request' }, { status: 400 });
	}
};
