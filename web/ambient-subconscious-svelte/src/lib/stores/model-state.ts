/**
 * Model state store - tracks SNN inference state (sound/silence).
 * In-memory server-side state, similar to avatar store.
 */

import { spawn, type ChildProcess } from 'child_process';
import { resolve } from 'path';

export interface ModelState {
	is_sound: boolean;
	is_running: boolean;
	last_updated: string;
}

let is_sound = false;
let last_updated = new Date().toISOString();
let listenerProcess: ChildProcess | null = null;

export function getState(): ModelState {
	return {
		is_sound,
		is_running: listenerProcess !== null && !listenerProcess.killed,
		last_updated
	};
}

export function setSound(value: boolean): void {
	is_sound = value;
	last_updated = new Date().toISOString();
}

export function startListener(device?: number): { success: boolean; error?: string } {
	// Kill existing if running
	stopListener();

	const scriptPath = resolve('../../snn_experiment/listener.py');
	const args = ['--verbose'];
	if (device !== undefined && device !== null) {
		args.push('--device', String(device));
	}

	try {
		listenerProcess = spawn('python', [scriptPath, ...args], {
			stdio: ['ignore', 'pipe', 'pipe'],
			detached: false
		});

		listenerProcess.stdout?.on('data', (data: Buffer) => {
			console.log(`[SNN Listener] ${data.toString().trim()}`);
		});

		listenerProcess.stderr?.on('data', (data: Buffer) => {
			console.error(`[SNN Listener ERR] ${data.toString().trim()}`);
		});

		listenerProcess.on('exit', (code) => {
			console.log(`[SNN Listener] Process exited with code ${code}`);
			listenerProcess = null;
		});

		return { success: true };
	} catch (e) {
		return { success: false, error: String(e) };
	}
}

export function stopListener(): void {
	if (listenerProcess && !listenerProcess.killed) {
		listenerProcess.kill('SIGTERM');
		listenerProcess = null;
	}
	is_sound = false;
	last_updated = new Date().toISOString();
}

// Kill orphaned listener processes on module load (Windows-compatible)
try {
	const { execSync } = await import('child_process');
	const platform = process.platform;
	if (platform === 'win32') {
		// taskkill silently fails if no matching process
		try {
			execSync('taskkill /F /FI "WINDOWTITLE eq listener.py" 2>nul', { stdio: 'ignore' });
		} catch { /* no orphans */ }
		try {
			execSync('wmic process where "commandline like \'%listener.py%\'" call terminate 2>nul', { stdio: 'ignore' });
		} catch { /* no orphans */ }
	} else {
		try {
			execSync('pkill -f "listener.py"', { stdio: 'ignore' });
		} catch { /* no orphans */ }
	}
} catch { /* cleanup is best-effort */ }
