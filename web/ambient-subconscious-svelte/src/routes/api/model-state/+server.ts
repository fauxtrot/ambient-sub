/**
 * Model state API endpoint.
 *
 * GET  /api/model-state          - Get current state
 * POST /api/model-state           - Update state or control listener
 *   { is_sound: boolean }         - Update sound detection state (from Python listener)
 *   { action: "start", device?: number } - Start listener process
 *   { action: "stop" }            - Stop listener process
 */
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getState, setSound, startListener, stopListener } from '$lib/stores/model-state';

export const GET: RequestHandler = async () => {
	return json(getState());
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();

		// Action-based requests (start/stop)
		if (body.action) {
			if (body.action === 'start') {
				const result = startListener(body.device);
				if (!result.success) {
					return json({ error: result.error }, { status: 500 });
				}
				return json({ ...getState(), message: 'Listener started' });
			}

			if (body.action === 'stop') {
				stopListener();
				return json({ ...getState(), message: 'Listener stopped' });
			}

			return json({ error: 'Unknown action' }, { status: 400 });
		}

		// State update from Python listener
		if (typeof body.is_sound === 'boolean') {
			setSound(body.is_sound);
			return json({ ok: true });
		}

		return json({ error: 'Invalid request body' }, { status: 400 });
	} catch {
		return json({ error: 'Invalid JSON body' }, { status: 400 });
	}
};
