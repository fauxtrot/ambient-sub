import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import fs from 'fs';
import path from 'path';

const LOG_FILES: Record<string, string> = {
	observer: 'logs/observer.log',
	llm: 'logs/llm.log'
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const { processName } = await request.json();

		const logFile = LOG_FILES[processName];
		if (!logFile) {
			return json({ error: 'Unknown process' }, { status: 400 });
		}

		// Go up two directories from Svelte project to find logs directory
		const projectRoot = path.resolve(process.cwd(), '../..');
		const logPath = path.join(projectRoot, logFile);

		if (fs.existsSync(logPath)) {
			fs.writeFileSync(logPath, '', 'utf-8');
		}

		return json({ success: true });
	} catch (err) {
		return json({ error: String(err) }, { status: 500 });
	}
};
