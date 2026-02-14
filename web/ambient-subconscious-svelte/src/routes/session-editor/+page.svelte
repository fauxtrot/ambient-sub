<script lang="ts">
	import { onMount, onDestroy } from 'svelte';

	interface Session {
		session_id: string;
		started_at: string;
		device_name: string;
		has_enriched_data: boolean;
		segment_count: number;
	}

	interface Segment {
		segment_id: string;
		session_id: string;
		start_time: number;
		end_time: number;
		duration: number;
		has_sound: boolean;
		has_speaker: boolean;
		speaker_changed: boolean;
		speaker_id: string;
		speaker_confidence: number;
		transcription: string;
		language: string;
		detected_by_diart: boolean;
		gap_from_diart: boolean;
	}

	const PALETTE = [
		'#00ed73', '#4da6ff', '#ff6b6b', '#ffd93d',
		'#c084fc', '#fb923c', '#22d3ee', '#f472b6',
		'#a3e635', '#e879f9'
	];

	const SPEAKER_COLORS: Record<string, string> = {
		unknown: '#666'
	};

	let sessions: Session[] = $state([]);
	let selectedSessionId: string = $state('');
	let segments: Segment[] = $state([]);
	let selectedSegmentId: string | null = $state(null);
	let hasUnsavedChanges: boolean = $state(false);
	let saving: boolean = $state(false);
	let loading: boolean = $state(false);
	let isPlaying: boolean = $state(false);

	// Speaker name mapping (cluster ID → display name)
	let speakerNames: Record<string, string> = $state({});

	// Wavesurfer
	let waveformContainer: HTMLDivElement;
	let wavesurfer: any = null;
	let regionsPlugin: any = null;

	// Unique speakers in current session
	let uniqueSpeakers = $derived(
		[...new Set(segments.map((s) => s.speaker_id))].sort()
	);

	let selectedSegment = $derived(
		segments.find((s) => s.segment_id === selectedSegmentId) ?? null
	);

	// Track next speaker index for new speakers
	let nextSpeakerIndex: number = $state(0);

	function speakerColor(id: string): string {
		if (SPEAKER_COLORS[id]) return SPEAKER_COLORS[id];
		// Assign from palette on first use
		const idx = uniqueSpeakers.filter((s) => s !== 'unknown').indexOf(id);
		if (idx >= 0) return PALETTE[idx % PALETTE.length];
		return '#888';
	}

	function addNewSpeaker(): string {
		const id = `speaker${nextSpeakerIndex}`;
		nextSpeakerIndex++;
		// Make sure ID doesn't collide
		while (uniqueSpeakers.includes(`speaker${nextSpeakerIndex - 1}`)) {
			nextSpeakerIndex++;
		}
		const newId = `speaker${nextSpeakerIndex - 1}`;
		if (!speakerNames[newId]) speakerNames = { ...speakerNames, [newId]: '' };
		// We need at least one segment with this speaker for it to appear
		return newId;
	}

	function speakerDisplayName(id: string): string {
		return speakerNames[id] || id;
	}

	function formatTime(seconds: number): string {
		const m = Math.floor(seconds / 60);
		const s = Math.floor(seconds % 60);
		return `${m}:${s.toString().padStart(2, '0')}`;
	}

	async function loadSessions() {
		const res = await fetch('/api/sessions');
		if (res.ok) {
			const data = await res.json();
			sessions = data.sessions.filter((s: Session) => s.has_enriched_data);
		}
	}

	async function loadSession(sessionId: string) {
		if (!sessionId) return;
		loading = true;
		hasUnsavedChanges = false;

		const res = await fetch(`/api/sessions/${sessionId}`);
		if (!res.ok) {
			loading = false;
			return;
		}
		const data = await res.json();
		segments = data.segments;

		// Restore speaker map if saved previously
		const savedMap: Record<string, string> = data.speaker_map || {};

		// Initialize speaker names and find max speaker index
		let maxIdx = -1;
		const newNames: Record<string, string> = {};
		for (const sp of [...new Set(segments.map((s: Segment) => s.speaker_id))]) {
			newNames[sp] = savedMap[sp] || '';
			const match = sp.match(/^speaker(\d+)$/);
			if (match) maxIdx = Math.max(maxIdx, parseInt(match[1], 10));
		}
		speakerNames = newNames;
		nextSpeakerIndex = maxIdx + 1;

		await initWavesurfer(sessionId);
		loading = false;
	}

	async function initWavesurfer(sessionId: string) {
		if (wavesurfer) {
			wavesurfer.destroy();
			wavesurfer = null;
		}

		const WaveSurfer = (await import('wavesurfer.js')).default;
		const RegionsPlugin = (await import('wavesurfer.js/dist/plugins/regions.js')).default;

		regionsPlugin = RegionsPlugin.create();

		wavesurfer = WaveSurfer.create({
			container: waveformContainer,
			waveColor: '#3f3f3f',
			progressColor: '#00ed73',
			cursorColor: '#fff',
			height: 128,
			barWidth: 2,
			barGap: 1,
			normalize: true,
			plugins: [regionsPlugin]
		});

		wavesurfer.load(`/api/sessions/${sessionId}/audio`);

		wavesurfer.on('ready', () => {
			addRegions();
		});

		wavesurfer.on('play', () => { isPlaying = true; });
		wavesurfer.on('pause', () => { isPlaying = false; });
		wavesurfer.on('finish', () => { isPlaying = false; });

		regionsPlugin.on('region-clicked', (region: any, e: MouseEvent) => {
			e.stopPropagation();
			selectedSegmentId = region.id;
			wavesurfer.setTime(region.start);
			wavesurfer.play();
			scrollToSegment(region.id);
		});
	}

	function addRegions() {
		if (!regionsPlugin) return;
		regionsPlugin.clearRegions();

		for (const seg of segments) {
			regionsPlugin.addRegion({
				id: seg.segment_id,
				start: seg.start_time,
				end: seg.end_time,
				color: speakerColor(seg.speaker_id) + '33',
				drag: false,
				resize: false
			});
		}
	}

	function scrollToSegment(segmentId: string) {
		const el = document.getElementById(`seg-${segmentId}`);
		if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
	}

	function selectSegment(seg: Segment) {
		selectedSegmentId = seg.segment_id;
		if (wavesurfer) {
			wavesurfer.setTime(seg.start_time);
			wavesurfer.play();
		}
	}

	function reassignSpeaker(segmentId: string, newSpeaker: string) {
		if (newSpeaker === '__new__') {
			// Find next available speakerN
			let idx = 0;
			while (uniqueSpeakers.includes(`speaker${idx}`)) idx++;
			newSpeaker = `speaker${idx}`;
			speakerNames = { ...speakerNames, [newSpeaker]: '' };
		}
		segments = segments.map((s) =>
			s.segment_id === segmentId ? { ...s, speaker_id: newSpeaker } : s
		);
		hasUnsavedChanges = true;
		updateRegionColor(segmentId, newSpeaker);
	}

	function updateRegionColor(segmentId: string, speakerId: string) {
		if (!regionsPlugin) return;
		const regions = regionsPlugin.getRegions();
		const region = regions.find((r: any) => r.id === segmentId);
		if (region) {
			region.setOptions({ color: speakerColor(speakerId) + '33' });
		}
	}

	function renameSpeaker(oldId: string, newName: string) {
		speakerNames = { ...speakerNames, [oldId]: newName };
		hasUnsavedChanges = true;
	}

	function applyRenames(): Segment[] {
		// Build rename map: oldId → canonical name (only where name is set)
		const renameMap: Record<string, string> = {};
		for (const [id, name] of Object.entries(speakerNames)) {
			const trimmed = name.trim().toLowerCase().replace(/\s+/g, '_');
			if (trimmed) renameMap[id] = trimmed;
		}

		if (Object.keys(renameMap).length === 0) return segments;

		return segments.map((s) => {
			const newId = renameMap[s.speaker_id];
			return newId ? { ...s, speaker_id: newId } : s;
		});
	}

	async function save() {
		if (!selectedSessionId) return;
		saving = true;

		// Apply renames to get canonical speaker IDs
		const finalSegments = applyRenames();

		// Build speaker map for this session
		const speakerMap: Record<string, string> = {};
		for (const [id, name] of Object.entries(speakerNames)) {
			const trimmed = name.trim();
			if (trimmed) speakerMap[id] = trimmed;
		}

		const res = await fetch(`/api/sessions/${selectedSessionId}`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ segments: finalSegments, speaker_map: speakerMap })
		});
		if (res.ok) {
			// Update local state to match what was saved
			segments = finalSegments;
			// Reset speakerNames — canonical IDs are now the names
			const newNames: Record<string, string> = {};
			for (const sp of [...new Set(finalSegments.map((s) => s.speaker_id))]) {
				newNames[sp] = '';  // already canonical
			}
			speakerNames = newNames;
			hasUnsavedChanges = false;
			addRegions();  // refresh region colors
		}
		saving = false;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.code === 'Space' && wavesurfer && (e.target === document.body || (e.target as HTMLElement)?.tagName !== 'INPUT')) {
			e.preventDefault();
			wavesurfer.playPause();
		}
	}

	onMount(() => {
		loadSessions();
		document.addEventListener('keydown', handleKeydown);
	});

	onDestroy(() => {
		document.removeEventListener('keydown', handleKeydown);
		if (wavesurfer) wavesurfer.destroy();
	});

	// Watch session selection
	$effect(() => {
		if (selectedSessionId) {
			loadSession(selectedSessionId);
		}
	});
</script>

<div class="editor">
	<div class="toolbar">
		<select class="select" bind:value={selectedSessionId}>
			<option value="">Select a session...</option>
			{#each sessions as session}
				<option value={session.session_id}>
					{session.session_id} ({session.segment_count} segments)
				</option>
			{/each}
		</select>

		<button class="btn btn-primary" onclick={save} disabled={!hasUnsavedChanges || saving}>
			{saving ? 'Saving...' : 'Save'}
		</button>

		{#if hasUnsavedChanges}
			<span class="unsaved-badge">Unsaved changes</span>
		{/if}
	</div>

	<div class="waveform-row">
		<div class="waveform-container" bind:this={waveformContainer}>
			{#if loading}
				<div class="waveform-loading">Loading audio...</div>
			{/if}
		</div>
		{#if wavesurfer}
			<div class="playback-controls">
				<button class="btn btn-ghost btn-icon" onclick={() => wavesurfer?.playPause()} title="Play/Pause (Space)">
					{isPlaying ? '⏸' : '▶'}
				</button>
				<button class="btn btn-ghost btn-icon" onclick={() => { wavesurfer?.stop(); isPlaying = false; }} title="Stop">
					⏹
				</button>
			</div>
		{/if}
	</div>

	{#if segments.length > 0}
		<div class="speaker-legend">
			{#each uniqueSpeakers as spk}
				<div class="legend-item">
					<span class="legend-dot" style="background: {speakerColor(spk)}"></span>
					<span class="legend-id">{spk}</span>
					<input
						class="legend-name-input"
						type="text"
						placeholder="Name..."
						value={speakerNames[spk] || ''}
						oninput={(e) => renameSpeaker(spk, (e.target as HTMLInputElement).value)}
					/>
					<span class="legend-count">
						({segments.filter((s) => s.speaker_id === spk).length})
					</span>
				</div>
			{/each}
		</div>

		<div class="segment-list">
			<table>
				<thead>
					<tr>
						<th class="col-time">Time</th>
						<th class="col-speaker">Speaker</th>
						<th class="col-conf">Conf</th>
						<th class="col-text">Transcription</th>
					</tr>
				</thead>
				<tbody>
					{#each segments as seg}
						<tr
							id="seg-{seg.segment_id}"
							class:selected={seg.segment_id === selectedSegmentId}
							onclick={() => selectSegment(seg)}
						>
							<td class="col-time">
								{formatTime(seg.start_time)} - {formatTime(seg.end_time)}
							</td>
							<td class="col-speaker">
								<select
									class="speaker-select"
									value={seg.speaker_id}
									style="border-color: {speakerColor(seg.speaker_id)}"
									onchange={(e) =>
										reassignSpeaker(
											seg.segment_id,
											(e.target as HTMLSelectElement).value
										)}
									onclick={(e) => e.stopPropagation()}
								>
									{#each uniqueSpeakers as spk}
										<option value={spk}>
											{speakerDisplayName(spk) || spk}
										</option>
									{/each}
									<option value="__new__">+ New speaker...</option>
								</select>
							</td>
							<td class="col-conf">
								{(seg.speaker_confidence * 100).toFixed(0)}%
							</td>
							<td class="col-text">{seg.transcription || '—'}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>

<style>
	.editor {
		display: flex;
		flex-direction: column;
		height: 100%;
		overflow: hidden;
		padding: var(--space-md);
		gap: var(--space-md);
	}

	.toolbar {
		display: flex;
		align-items: center;
		gap: var(--space-md);
		flex-shrink: 0;
	}

	.toolbar .select {
		min-width: 320px;
	}

	.unsaved-badge {
		font-size: var(--text-xs);
		color: var(--clr-warning-a10, #ffd93d);
		background: rgba(255, 217, 61, 0.15);
		padding: var(--space-xs) var(--space-sm);
		border-radius: var(--radius-full);
	}

	.waveform-row {
		display: flex;
		gap: var(--space-sm);
		align-items: stretch;
		flex-shrink: 0;
	}

	.playback-controls {
		display: flex;
		flex-direction: column;
		justify-content: center;
		gap: var(--space-xs);
	}

	.waveform-container {
		flex: 1;
		min-height: 128px;
		background: var(--clr-surface-a10, #282828);
		border-radius: var(--radius-md);
		border: 1px solid var(--clr-surface-a20, #3f3f3f);
		position: relative;
	}

	.waveform-loading {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--clr-text-a30, #888);
		font-size: var(--text-sm);
	}

	.speaker-legend {
		display: flex;
		gap: var(--space-lg);
		flex-wrap: wrap;
		flex-shrink: 0;
	}

	.legend-item {
		display: flex;
		align-items: center;
		gap: var(--space-xs);
		font-size: var(--text-sm);
	}

	.legend-dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.legend-id {
		color: var(--clr-text-a30, #888);
		font-size: var(--text-xs);
	}

	.legend-name-input {
		width: 100px;
		padding: 2px 6px;
		font-size: var(--text-xs);
		background: var(--clr-surface-a10, #282828);
		border: 1px solid var(--clr-surface-a20, #3f3f3f);
		border-radius: var(--radius-sm);
		color: var(--clr-text-a0, #e0e0e0);
	}

	.legend-count {
		color: var(--clr-text-a30, #888);
		font-size: var(--text-xs);
	}

	.segment-list {
		flex: 1;
		overflow-y: auto;
		border-radius: var(--radius-md);
		border: 1px solid var(--clr-surface-a20, #3f3f3f);
	}

	table {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--text-sm);
	}

	thead {
		position: sticky;
		top: 0;
		background: var(--clr-surface-a10, #282828);
		z-index: 1;
	}

	th {
		text-align: left;
		padding: var(--space-sm) var(--space-md);
		font-weight: 600;
		color: var(--clr-text-a30, #888);
		font-size: var(--text-xs);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		border-bottom: 1px solid var(--clr-surface-a20, #3f3f3f);
	}

	td {
		padding: var(--space-xs) var(--space-md);
		border-bottom: 1px solid var(--clr-surface-a10, #282828);
		color: var(--clr-text-a0, #e0e0e0);
	}

	tr {
		cursor: pointer;
		transition: background var(--transition-fast, 150ms ease);
	}

	tr:hover {
		background: var(--clr-surface-a10, #282828);
	}

	tr.selected {
		background: rgba(0, 237, 115, 0.08);
	}

	.col-time {
		width: 140px;
		white-space: nowrap;
		font-family: var(--font-mono, monospace);
		font-size: var(--text-xs);
	}

	.col-speaker {
		width: 140px;
	}

	.col-conf {
		width: 60px;
		text-align: right;
		font-family: var(--font-mono, monospace);
		font-size: var(--text-xs);
	}

	.col-text {
		max-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.speaker-select {
		padding: 2px 4px;
		font-size: var(--text-xs);
		background: var(--clr-surface-a0, #121212);
		color: var(--clr-text-a0, #e0e0e0);
		border: 1px solid;
		border-radius: var(--radius-sm);
		cursor: pointer;
	}
</style>
