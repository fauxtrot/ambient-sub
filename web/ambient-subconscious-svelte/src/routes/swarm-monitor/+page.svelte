<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		swarmStatus,
		selectedProcess,
		sendInput,
		refreshStatus,
		fetchLogs,
		clearLogs
	} from '$lib/stores/swarm';

	let logs: Record<string, string[]> = {
		observer: [],
		llm: []
	};

	let userMessage = '';
	let logsContainer: HTMLDivElement | null = null;
	let pollInterval: number | null = null;

	// Observer config
	let visualDeltaThreshold = 20; // 0-100 percentage
	let soundDeltaEnabled = true;

	async function loadLogs(processName: string) {
		const newLogs = await fetchLogs(processName, 500);
		logs[processName] = newLogs;

		// Auto-scroll to bottom
		setTimeout(() => {
			if (logsContainer) {
				logsContainer.scrollTop = logsContainer.scrollHeight;
			}
		}, 0);
	}

	onMount(async () => {
		// Initial load
		await refreshStatus();
		await loadLogs('observer');
		await loadLogs('llm');

		// Poll every 2 seconds
		pollInterval = setInterval(async () => {
			await refreshStatus();
			await loadLogs($selectedProcess);
		}, 2000) as unknown as number;
	});

	onDestroy(() => {
		if (pollInterval !== null) {
			clearInterval(pollInterval);
		}
	});

	async function handleSendMessage() {
		if (!userMessage.trim()) return;

		try {
			await sendInput(userMessage.trim());
			userMessage = '';
		} catch (err) {
			alert(`Failed to send message: ${err}`);
		}
	}

	async function handleClearLogs(processName: string) {
		try {
			await clearLogs(processName);
			logs[processName] = [];
		} catch (err) {
			alert(`Failed to clear logs: ${err}`);
		}
	}

	async function wakeAgent() {
		try {
			// Import getConnection dynamically to avoid server-side issues
			const { getConnection } = await import('$lib/stores/spacetime');
			const connection = getConnection();

			if (!connection) {
				alert('Not connected to SpacetimeDB');
				return;
			}

			// Call UpdateExecutiveContext reducer with agentState="wake"
			await connection.reducers.UpdateExecutiveContext({
				recentVisual: null,
				recentAudio: null,
				baselineVisual: null,
				baselineAudio: null,
				userState: null,
				agentState: 'wake',
				nextCheckIn: null,
				notes: null
			});

			console.log('[SwarmMonitor] Sent wake trigger');
			alert('Wake signal sent to agent!');
		} catch (err) {
			console.error('[SwarmMonitor] Wake agent error:', err);
			alert(`Failed to wake agent: ${err}`);
		}
	}
</script>

<div class="swarm-monitor">
	<header>
		<h1>Swarm Monitor</h1>
		<p class="subtitle">Monitor Observer and Executive LLM processes</p>
	</header>

	<!-- Start Instructions -->
	<div class="start-instructions">
		<h3>Start Processes Manually</h3>
		<div class="instructions-grid">
			<div class="instruction-card">
				<strong>Observer:</strong>
				<code>python -m ambient_subconscious.executive.observer</code>
			</div>
			<div class="instruction-card">
				<strong>Executive LLM:</strong>
				<code>python -m ambient_subconscious.executive.llm_process</code>
			</div>
		</div>
	</div>

	<div class="layout">
		<!-- Process List -->
		<aside class="process-list">
			<h2>Processes</h2>

			{#each ['observer', 'llm'] as processName}
				{@const status = $swarmStatus[processName]}
				<div class="process-card" class:selected={$selectedProcess === processName}>
					<button
						class="process-header"
						onclick={() => selectedProcess.set(processName)}
						type="button"
					>
						<span class="status-indicator" class:running={status?.running}></span>
						<span class="process-name">{processName}</span>
					</button>

					<div class="process-details">
						{#if status?.running}
							<div class="status-text active">Active</div>
							<div class="last-active">
								Last activity: {new Date(status.lastActive || '').toLocaleTimeString()}
							</div>
						{:else}
							<div class="status-text inactive">Inactive</div>
							{#if status?.lastActive}
								<div class="last-active">
									Last seen: {new Date(status.lastActive).toLocaleTimeString()}
								</div>
							{/if}
						{/if}

						<button
							class="btn btn-danger btn-sm"
							onclick={() => handleClearLogs(processName)}
						>
							Clear Logs
						</button>
					</div>
				</div>
			{/each}

			<!-- Configuration Panel -->
			<div class="config-panel">
				<h3>Agent Configuration</h3>

				<div class="config-item">
					<label for="visual-delta">
						Visual Delta Threshold:
						<span class="config-value">{visualDeltaThreshold}%</span>
					</label>
					<input
						id="visual-delta"
						type="range"
						min="0"
						max="100"
						step="5"
						bind:value={visualDeltaThreshold}
						class="slider"
					/>
					<p class="config-help">
						Percentage change in object count to trigger update
					</p>
				</div>

				<div class="config-item">
					<label class="checkbox-label">
						<input type="checkbox" bind:checked={soundDeltaEnabled} />
						<span>Sound Delta Enabled</span>
					</label>
					<p class="config-help">
						Trigger update on any new audio entry
					</p>
				</div>

				<button class="btn btn-primary btn-wake" onclick={wakeAgent}>
					⚡ Wake Agent Now
				</button>

				<p class="config-note">
					Note: Config changes require restarting processes with appropriate flags.
				</p>
			</div>
		</aside>

		<!-- Main Content -->
		<main>
			<!-- Tabs -->
			<div class="tabs">
				{#each ['observer', 'llm'] as processName}
					<button
						class="tab"
						class:active={$selectedProcess === processName}
						onclick={() => selectedProcess.set(processName)}
						type="button"
					>
						{processName}
					</button>
				{/each}
			</div>

			<!-- Logs Viewer -->
			<div class="logs-container" bind:this={logsContainer}>
				{#each logs[$selectedProcess] || [] as log}
					<div class="log-line">{log}</div>
				{/each}
			</div>

			<!-- User Input Form -->
			<div class="user-input-form">
				<input
					type="text"
					class="input"
					placeholder="Type message here..."
					bind:value={userMessage}
					onkeydown={(e) => e.key === 'Enter' && handleSendMessage()}
				/>
				<button class="btn btn-primary" onclick={handleSendMessage}>Send</button>
			</div>
		</main>
	</div>
</div>

<style>
	.swarm-monitor {
		height: 100%;
		display: flex;
		flex-direction: column;
		background: var(--clr-surface);
		overflow: hidden;
	}

	header {
		padding: var(--space-md);
		border-bottom: 1px solid var(--clr-surface-a20);
	}

	header h1 {
		margin: 0;
		font-size: var(--text-xl);
	}

	.subtitle {
		margin: var(--space-xs) 0 0 0;
		color: var(--clr-text-secondary);
		font-size: var(--text-sm);
	}

	.start-instructions {
		padding: var(--space-md);
		background: var(--clr-surface-a10);
		border-bottom: 1px solid var(--clr-surface-a20);
	}

	.start-instructions h3 {
		margin: 0 0 var(--space-sm) 0;
		font-size: var(--text-md);
		color: var(--clr-text-secondary);
	}

	.instructions-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
		gap: var(--space-sm);
	}

	.instruction-card {
		padding: var(--space-sm);
		background: var(--clr-surface-a20);
		border-radius: var(--radius-sm);
		border: 1px solid var(--clr-surface-a30);
	}

	.instruction-card strong {
		display: block;
		margin-bottom: var(--space-xs);
		color: var(--clr-text);
	}

	.instruction-card code {
		display: block;
		padding: var(--space-xs);
		background: var(--clr-surface-a30);
		border-radius: var(--radius-xs);
		font-family: 'Courier New', monospace;
		font-size: var(--text-xs);
		word-break: break-all;
	}

	.layout {
		flex: 1;
		display: grid;
		grid-template-columns: 300px 1fr;
		overflow: hidden;
	}

	.process-list {
		border-right: 1px solid var(--clr-surface-a20);
		padding: var(--space-md);
		overflow-y: auto;
	}

	.process-list h2 {
		margin: 0 0 var(--space-md) 0;
		font-size: var(--text-lg);
	}

	.process-card {
		margin-bottom: var(--space-md);
		border: 1px solid var(--clr-surface-a20);
		border-radius: var(--radius-md);
		padding: var(--space-sm);
		background: var(--clr-surface-a10);
	}

	.process-card.selected {
		border-color: var(--clr-primary);
		background: var(--clr-surface-a20);
	}

	.process-header {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		width: 100%;
		background: none;
		border: none;
		color: inherit;
		cursor: pointer;
		font-size: var(--text-md);
		padding: 0;
		text-align: left;
	}

	.process-header:hover {
		color: var(--clr-primary);
	}

	.status-indicator {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: var(--clr-danger);
		flex-shrink: 0;
	}

	.status-indicator.running {
		background: var(--clr-success);
		animation: pulse 2s ease-in-out infinite;
	}

	@keyframes pulse {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
	}

	.process-name {
		font-weight: 500;
	}

	.process-details {
		margin-top: var(--space-sm);
		font-size: var(--text-sm);
	}

	.status-text {
		margin-bottom: var(--space-xs);
	}

	.status-text.active {
		color: var(--clr-success);
	}

	.status-text.inactive {
		color: var(--clr-danger);
	}

	.last-active {
		color: var(--clr-text-secondary);
		font-size: var(--text-xs);
		margin-bottom: var(--space-xs);
	}

	.btn-sm {
		padding: 4px 8px;
		font-size: var(--text-xs);
		margin-top: var(--space-xs);
	}

	main {
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.tabs {
		display: flex;
		border-bottom: 1px solid var(--clr-surface-a20);
		background: var(--clr-surface-a10);
	}

	.tab {
		padding: var(--space-sm) var(--space-md);
		background: none;
		border: none;
		border-bottom: 2px solid transparent;
		color: var(--clr-text-secondary);
		cursor: pointer;
		font-size: var(--text-sm);
		transition: all 0.2s ease;
	}

	.tab:hover {
		color: var(--clr-text);
		background: var(--clr-surface-a20);
	}

	.tab.active {
		border-bottom-color: var(--clr-primary);
		color: var(--clr-text);
		background: var(--clr-surface-a20);
	}

	.logs-container {
		flex: 1;
		overflow-y: auto;
		background: #1e1e1e;
		color: #00ff00;
		padding: var(--space-md);
		font-family: 'Courier New', monospace;
		font-size: 12px;
		line-height: 1.4;
	}

	.log-line {
		margin-bottom: 2px;
		word-break: break-all;
	}

	.user-input-form {
		display: flex;
		gap: var(--space-sm);
		padding: var(--space-md);
		border-top: 1px solid var(--clr-surface-a20);
		background: var(--clr-surface-a10);
	}

	.user-input-form .input {
		flex: 1;
	}

	/* Configuration Panel */
	.config-panel {
		margin-top: var(--space-lg);
		padding: var(--space-md);
		background: var(--clr-surface-a20);
		border: 1px solid var(--clr-surface-a30);
		border-radius: var(--radius-md);
	}

	.config-panel h3 {
		margin: 0 0 var(--space-md) 0;
		font-size: var(--text-md);
		color: var(--clr-primary);
	}

	.config-item {
		margin-bottom: var(--space-md);
	}

	.config-item label {
		display: block;
		margin-bottom: var(--space-xs);
		font-size: var(--text-sm);
		color: var(--clr-text);
		font-weight: 500;
	}

	.config-value {
		float: right;
		color: var(--clr-primary);
		font-weight: 600;
	}

	.slider {
		width: 100%;
		height: 6px;
		border-radius: 3px;
		background: var(--clr-surface-a30);
		outline: none;
		opacity: 0.9;
		transition: opacity 0.2s;
	}

	.slider:hover {
		opacity: 1;
	}

	.slider::-webkit-slider-thumb {
		appearance: none;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: var(--clr-primary);
		cursor: pointer;
	}

	.slider::-moz-range-thumb {
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: var(--clr-primary);
		cursor: pointer;
		border: none;
	}

	.checkbox-label {
		display: flex;
		align-items: center;
		gap: var(--space-xs);
		cursor: pointer;
		user-select: none;
	}

	.checkbox-label input[type='checkbox'] {
		width: 18px;
		height: 18px;
		cursor: pointer;
	}

	.config-help {
		margin: var(--space-xs) 0 0 0;
		font-size: var(--text-xs);
		color: var(--clr-text-secondary);
		line-height: 1.3;
	}

	.btn-wake {
		width: 100%;
		margin-top: var(--space-sm);
		font-weight: 600;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-xs);
	}

	.config-note {
		margin-top: var(--space-md);
		padding: var(--space-sm);
		background: var(--clr-surface-a30);
		border-radius: var(--radius-sm);
		font-size: var(--text-xs);
		color: var(--clr-text-secondary);
		line-height: 1.4;
	}
</style>
