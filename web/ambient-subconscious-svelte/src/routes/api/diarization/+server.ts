/**
 * Diarization API endpoint.
 *
 * POST /api/diarization - Store a diarization segment for a transcript entry
 *
 * Body: {
 *   entryId: number,
 *   startMs: number,
 *   endMs: number,
 *   pyannoteLabel: string,
 *   matchedSpeaker?: string,
 *   confidence?: number,
 *   transcriptSlice?: string
 * }
 */
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { DbConnection } from '../../../generated';

const SPACETIMEDB_HOST = process.env.SPACETIMEDB_HOST || 'ws://127.0.0.1:3000';
const SPACETIMEDB_MODULE = process.env.SPACETIMEDB_MODULE || 'ambient-listener';

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();
		const { entryId, startMs, endMs, pyannoteLabel, matchedSpeaker, confidence, transcriptSlice } =
			body;

		// Validate required fields
		if (typeof entryId !== 'number') {
			return json({ error: 'entryId is required and must be a number' }, { status: 400 });
		}
		if (typeof startMs !== 'number') {
			return json({ error: 'startMs is required and must be a number' }, { status: 400 });
		}
		if (typeof endMs !== 'number') {
			return json({ error: 'endMs is required and must be a number' }, { status: 400 });
		}
		if (typeof pyannoteLabel !== 'string') {
			return json({ error: 'pyannoteLabel is required and must be a string' }, { status: 400 });
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
					// Call the CreateDiarizationSegment reducer using PascalCase and object syntax
					conn!.reducers.CreateDiarizationSegment({
						entryId,
						startMs,
						endMs,
						pyannoteLabel,
						matchedSpeaker: matchedSpeaker ?? undefined,
						confidence: confidence ?? undefined,
						transcriptSlice: transcriptSlice ?? undefined
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
			return json({ error: result.error || 'Failed to create diarization segment' }, { status: 500 });
		}

		console.log(
			`[Diarization] Created segment for entry ${entryId}: ${pyannoteLabel} (${startMs}ms - ${endMs}ms)`
		);
		return json({ success: true, entryId });
	} catch (err) {
		console.error('[Diarization] Error:', err);
		return json({ error: 'Invalid request' }, { status: 400 });
	}
};
