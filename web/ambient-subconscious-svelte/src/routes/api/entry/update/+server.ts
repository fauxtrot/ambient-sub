/**
 * Entry Update API endpoint.
 *
 * POST /api/entry/update - Update a transcript entry's fields
 *
 * Body: {
 *   entryId: number,
 *   speaker?: string,
 *   sentiment?: string,
 *   transcript?: string,
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
		const { entryId, speaker, sentiment, transcript, notes } = body;

		if (typeof entryId !== 'number') {
			return json({ error: 'entryId is required and must be a number' }, { status: 400 });
		}

		// At least one field should be provided
		if (
			speaker === undefined &&
			sentiment === undefined &&
			transcript === undefined &&
			notes === undefined
		) {
			return json({ error: 'At least one field to update is required' }, { status: 400 });
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
					// Call the UpdateEntry reducer using PascalCase and object syntax
					conn!.reducers.UpdateEntry({
						entryId,
						transcript: transcript ?? undefined,
						speakerName: speaker ?? undefined,
						sentiment: sentiment ?? undefined,
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
			return json({ error: result.error || 'Failed to update entry' }, { status: 500 });
		}

		console.log(`[EntryUpdate] Updated entry ${entryId}`);
		return json({ success: true, entryId });
	} catch (err) {
		console.error('[EntryUpdate] Error:', err);
		return json({ error: 'Invalid request' }, { status: 400 });
	}
};
