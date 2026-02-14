import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const SESSIONS_DIR = join(process.cwd(), '..', '..', 'data', 'sessions');

export const GET: RequestHandler = async () => {
	try {
		const entries = await readdir(SESSIONS_DIR, { withFileTypes: true });
		const sessions = [];

		for (const entry of entries) {
			if (!entry.isDirectory()) continue;

			try {
				const metaPath = join(SESSIONS_DIR, entry.name, 'metadata.json');
				const meta = JSON.parse(await readFile(metaPath, 'utf-8'));

				let segmentCount = 0;
				try {
					const enrichedPath = join(SESSIONS_DIR, entry.name, 'enriched_training_data.json');
					const enriched = JSON.parse(await readFile(enrichedPath, 'utf-8'));
					segmentCount = Array.isArray(enriched) ? enriched.length : 0;
				} catch {
					/* no enriched data */
				}

				sessions.push({
					session_id: entry.name,
					started_at: meta.started_at,
					device_name: meta.device_name,
					has_enriched_data: segmentCount > 0,
					segment_count: segmentCount
				});
			} catch {
				/* skip dirs without metadata */
			}
		}

		sessions.sort((a, b) => b.started_at.localeCompare(a.started_at));
		return json({ sessions });
	} catch (err) {
		return json({ error: 'Failed to read sessions directory' }, { status: 500 });
	}
};
