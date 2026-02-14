import { writable } from 'svelte/store';

export interface ProcessStatus {
	running: boolean;
	lastActive: string | null;
}

export interface SwarmState {
	observer: ProcessStatus | null;
	llm: ProcessStatus | null;
}

export const swarmStatus = writable<SwarmState>({
	observer: null,
	llm: null
});

export const selectedProcess = writable<string>('observer');

// Check process status via log file age
async function checkProcessStatus(processName: string): Promise<ProcessStatus> {
	const res = await fetch(`/api/swarm/logs?process=${processName}&lines=1`);
	const { lastModified } = await res.json();

	if (!lastModified) return { running: false, lastActive: null };

	// Consider running if log updated in last 30 seconds
	const ageSeconds = (Date.now() - new Date(lastModified).getTime()) / 1000;
	return {
		running: ageSeconds < 30,
		lastActive: lastModified
	};
}

// Refresh status for all processes
export async function refreshStatus() {
	const observer = await checkProcessStatus('observer');
	const llm = await checkProcessStatus('llm');
	swarmStatus.set({ observer, llm });
}

// Fetch logs for a process
export async function fetchLogs(processName: string, lines = 100): Promise<string[]> {
	const res = await fetch(`/api/swarm/logs?process=${processName}&lines=${lines}`);
	const { logs } = await res.json();
	return logs || [];
}

// Clear logs for a process
export async function clearLogs(processName: string) {
	await fetch('/api/swarm/clear-logs', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ processName })
	});
}

// Send user input (create TranscriptEntry)
export async function sendInput(text: string) {
	try {
		console.log('[SwarmInput] Starting sendInput...');

		// Import here to avoid server-side issues
		const { getConnection } = await import('./spacetime');
		const connection = getConnection();

		console.log('[SwarmInput] Connection:', connection ? 'connected' : 'null');

		if (!connection) {
			throw new Error('Not connected to SpacetimeDB. Please wait for connection to establish.');
		}

		const entryId = `user_${Date.now()}`;

		console.log('[SwarmInput] Calling CreateEntry with:', {
			sessionId: 1,
			entryId,
			durationMs: 0,
			transcript: text.trim(),
			confidence: 1.0,
			audioClipPath: null,
			recordingStartMs: 0,
			recordingEndMs: 0
		});

		// Call CreateEntry reducer directly from client
		await connection.reducers.CreateEntry({
			sessionId: 1,
			entryId,
			durationMs: 0,
			transcript: text.trim(),
			confidence: 1.0,
			audioClipPath: null,
			recordingStartMs: 0,
			recordingEndMs: 0
		});

		console.log(`[SwarmInput] Created entry: ${entryId} - ${text}`);

		return { success: true, entryId };
	} catch (error) {
		console.error('[SwarmInput] Error:', error);
		throw error;
	}
}
