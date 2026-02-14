<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	interface ModelState {
		is_sound: boolean;
		is_running: boolean;
		last_updated: string;
	}

	let state: ModelState = $state({
		is_sound: false,
		is_running: false,
		last_updated: ''
	});
	let deviceIndex: string = $state('');
	let loading: boolean = $state(false);
	let pollInterval: ReturnType<typeof setInterval> | null = null;

	async function fetchState() {
		try {
			const res = await fetch('/api/model-state');
			if (res.ok) {
				state = await res.json();
			}
		} catch {
			/* polling failure is non-critical */
		}
	}

	async function start() {
		loading = true;
		try {
			const body: Record<string, unknown> = { action: 'start' };
			const idx = String(deviceIndex).trim();
			if (idx !== '') {
				body.device = parseInt(idx, 10);
			}
			await fetch('/api/model-state', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body)
			});
			await fetchState();
		} finally {
			loading = false;
		}
	}

	async function stop() {
		loading = true;
		try {
			await fetch('/api/model-state', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ action: 'stop' })
			});
			await fetchState();
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		fetchState();
		pollInterval = setInterval(fetchState, 500);
	});

	onDestroy(() => {
		if (pollInterval) clearInterval(pollInterval);
	});

	let timeSince = $derived(() => {
		if (!state.last_updated) return '';
		const diff = Date.now() - new Date(state.last_updated).getTime();
		if (diff < 1000) return 'just now';
		if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
		return `${Math.floor(diff / 60000)}m ago`;
	});
</script>

<div class="container">
	<h1>Model State</h1>
	<p class="subtitle">SNN Sound/Silence Classifier</p>

	<div class="status-card">
		<div class="indicator" class:sound={state.is_sound} class:silence={!state.is_sound}>
			<div class="dot"></div>
		</div>
		<div class="label">
			{state.is_sound ? 'Sound Detected' : 'Silence'}
		</div>
		<div class="meta">
			{#if state.last_updated}
				Updated {timeSince()}
			{/if}
		</div>
	</div>

	<div class="controls">
		<div class="field">
			<label for="device-index">Audio Device Index</label>
			<input
				id="device-index"
				type="number"
				bind:value={deviceIndex}
				placeholder="Default"
				min="0"
				disabled={state.is_running}
			/>
			<span class="hint">Run <code>python listener.py --list-devices</code> to see available devices</span>
		</div>

		<div class="actions">
			{#if state.is_running}
				<button class="btn stop" onclick={stop} disabled={loading}>
					Stop Listener
				</button>
			{:else}
				<button class="btn start" onclick={start} disabled={loading}>
					Start Listener
				</button>
			{/if}
		</div>

		<div class="runner-status">
			<span class="runner-dot" class:active={state.is_running}></span>
			{state.is_running ? 'Listener running' : 'Listener stopped'}
		</div>
	</div>
</div>

<style>
	.container {
		max-width: 500px;
		margin: 2rem auto;
		padding: 0 1rem;
		color: var(--color-text, #e0e0e0);
	}

	h1 {
		font-size: 1.5rem;
		font-weight: 600;
		margin: 0;
	}

	.subtitle {
		color: var(--color-text-muted, #888);
		margin: 0.25rem 0 2rem;
		font-size: 0.875rem;
	}

	.status-card {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 1rem;
		padding: 2rem;
		border-radius: 12px;
		background: var(--color-surface, #1a1a2e);
		border: 1px solid var(--color-border, #2a2a3e);
		margin-bottom: 2rem;
	}

	.indicator {
		width: 80px;
		height: 80px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		transition: all 0.3s ease;
	}

	.indicator.sound {
		background: rgba(0, 237, 115, 0.15);
		box-shadow: 0 0 30px rgba(0, 237, 115, 0.3);
	}

	.indicator.silence {
		background: rgba(128, 128, 128, 0.15);
	}

	.dot {
		width: 40px;
		height: 40px;
		border-radius: 50%;
		transition: all 0.3s ease;
	}

	.sound .dot {
		background: #00ed73;
		box-shadow: 0 0 15px rgba(0, 237, 115, 0.5);
	}

	.silence .dot {
		background: #555;
	}

	.label {
		font-size: 1.25rem;
		font-weight: 600;
	}

	.meta {
		font-size: 0.75rem;
		color: var(--color-text-muted, #888);
	}

	.controls {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.field {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.field label {
		font-size: 0.8rem;
		font-weight: 500;
		color: var(--color-text-muted, #aaa);
	}

	.field input {
		padding: 0.5rem 0.75rem;
		border-radius: 6px;
		border: 1px solid var(--color-border, #2a2a3e);
		background: var(--color-surface, #1a1a2e);
		color: var(--color-text, #e0e0e0);
		font-size: 0.875rem;
		width: 120px;
	}

	.hint {
		font-size: 0.7rem;
		color: var(--color-text-muted, #666);
	}

	.hint code {
		background: rgba(255, 255, 255, 0.05);
		padding: 0.1em 0.3em;
		border-radius: 3px;
		font-size: 0.65rem;
	}

	.actions {
		margin-top: 0.5rem;
	}

	.btn {
		padding: 0.6rem 1.5rem;
		border-radius: 6px;
		border: none;
		font-weight: 600;
		font-size: 0.875rem;
		cursor: pointer;
		transition: opacity 0.15s;
	}

	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn.start {
		background: #00ed73;
		color: #000;
	}

	.btn.stop {
		background: #e54;
		color: #fff;
	}

	.runner-status {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.8rem;
		color: var(--color-text-muted, #888);
	}

	.runner-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: #555;
	}

	.runner-dot.active {
		background: #00ed73;
		box-shadow: 0 0 6px rgba(0, 237, 115, 0.5);
	}
</style>
