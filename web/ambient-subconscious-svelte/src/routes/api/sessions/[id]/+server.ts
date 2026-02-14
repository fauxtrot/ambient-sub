import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const SESSIONS_DIR = join(process.cwd(), '..', '..', 'data', 'sessions');

function sessionPath(id: string) {
	// Sanitize: only allow alphanumeric, underscore, hyphen
	if (!/^[\w-]+$/.test(id)) throw new Error('Invalid session ID');
	return join(SESSIONS_DIR, id);
}

export const GET: RequestHandler = async ({ params }) => {
	try {
		const dir = sessionPath(params.id);
		const enrichedPath = join(dir, 'enriched_training_data.json');
		const segments = JSON.parse(await readFile(enrichedPath, 'utf-8'));

		let meta = {};
		try {
			meta = JSON.parse(await readFile(join(dir, 'metadata.json'), 'utf-8'));
		} catch {
			/* optional */
		}

		let speakerMap = {};
		try {
			speakerMap = JSON.parse(await readFile(join(dir, 'speaker_map.json'), 'utf-8'));
		} catch {
			/* optional */
		}

		return json({ segments, metadata: meta, speaker_map: speakerMap });
	} catch (err) {
		return json({ error: 'Session not found' }, { status: 404 });
	}
};

export const POST: RequestHandler = async ({ params, request }) => {
	try {
		const dir = sessionPath(params.id);
		const body = await request.json();

		if (!Array.isArray(body.segments)) {
			return json({ error: 'segments array required' }, { status: 400 });
		}

		const enrichedPath = join(dir, 'enriched_training_data.json');
		await writeFile(enrichedPath, JSON.stringify(body.segments, null, 2), 'utf-8');

		// Save speaker map if provided
		if (body.speaker_map && typeof body.speaker_map === 'object') {
			const mapPath = join(dir, 'speaker_map.json');
			await writeFile(mapPath, JSON.stringify(body.speaker_map, null, 2), 'utf-8');
		}

		return json({ success: true, count: body.segments.length });
	} catch (err) {
		return json({ error: 'Failed to save' }, { status: 500 });
	}
};
