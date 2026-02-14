/**
 * Assistant Response API endpoint.
 *
 * POST /api/assistant-response - Store an AI assistant response
 *
 * Body: {
 *   entryId: number,
 *   quickReply?: string,
 *   fullReply?: string,
 *   reaction?: string,
 *   thinkingLog?: string
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
		const { entryId, quickReply, fullReply, reaction, thinkingLog } = body;

		if (typeof entryId !== 'number') {
			return json({ error: 'entryId is required and must be a number' }, { status: 400 });
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
					// Call the reducer using PascalCase and object syntax
					conn!.reducers.CreateAssistantResponse({
						entryId,
						quickReply: quickReply ?? undefined,
						fullReply: fullReply ?? undefined,
						reaction: reaction ?? undefined,
						thinkingLog: thinkingLog ?? undefined
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
			return json({ error: result.error || 'Failed to store response' }, { status: 500 });
		}

		console.log(`[AssistantResponse] Stored response for entry ${entryId}`);
		return json({ success: true, entryId });
	} catch (err) {
		console.error('[AssistantResponse] Error:', err);
		return json({ error: 'Invalid request' }, { status: 400 });
	}
};
