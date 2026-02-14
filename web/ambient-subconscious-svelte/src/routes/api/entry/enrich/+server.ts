/**
 * Entry Enrichment API endpoint.
 *
 * POST /api/entry/enrich - Enrich a transcript entry with speaker/sentiment/diarization
 *
 * Body: {
 *   entryId: number,
 *   speaker?: string,
 *   sentiment?: string,
 *   markEnriched?: boolean
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
		const { entryId, speaker, sentiment, markEnriched } = body;

		if (typeof entryId !== 'number') {
			return json({ error: 'entryId is required and must be a number' }, { status: 400 });
		}

		// Connect to SpacetimeDB and call reducers
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
					let delay = 0;

					// Update speaker if provided
					if (speaker) {
						setTimeout(() => {
							conn!.reducers.UpdateEntrySpeaker({ entryId, speakerName: speaker });
						}, delay);
						delay += 50;
					}

					// Update sentiment if provided
					if (sentiment) {
						setTimeout(() => {
							conn!.reducers.UpdateEntrySentiment({ entryId, sentiment });
						}, delay);
						delay += 50;
					}

					// Mark enriched if requested
					if (markEnriched) {
						setTimeout(() => {
							conn!.reducers.MarkEntryEnriched({ entryId });
						}, delay);
						delay += 50;
					}

					// Give it a moment to process, then disconnect
					setTimeout(() => {
						clearTimeout(timeout);
						if (conn) conn.disconnect();
						resolve({ success: true });
					}, delay + 100);
				})
				.onConnectError((_ctx, err) => {
					clearTimeout(timeout);
					if (conn) conn.disconnect();
					resolve({ success: false, error: String(err) });
				})
				.build();
		});

		if (!result.success) {
			return json({ error: result.error || 'Failed to enrich entry' }, { status: 500 });
		}

		console.log(`[EntryEnrich] Enriched entry ${entryId}`);
		return json({ success: true, entryId });
	} catch (err) {
		console.error('[EntryEnrich] Error:', err);
		return json({ error: 'Invalid request' }, { status: 400 });
	}
};
