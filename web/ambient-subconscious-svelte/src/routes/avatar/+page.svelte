<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { EXPRESSIONS, type Expression } from '$lib/stores/avatar';

	let currentExpression = $state<Expression>('neutral');
	let videoSrc = $state('/api/avatar/video/General.mp4');
	let showControls = $state(false);
	let videoElement: HTMLVideoElement;
	let pollInterval: ReturnType<typeof setInterval>;

	// Map expression to video path
	const expressionToVideo: Record<Expression, string> = {
		neutral: '/api/avatar/video/General.mp4',
		happy: '/api/avatar/video/Happy.mp4',
		sad: '/api/avatar/video/sad.mp4',
		angry: '/api/avatar/video/angry.mp4',
		confused: '/api/avatar/video/confused.mp4'
	};

	async function fetchState() {
		try {
			const res = await fetch('/api/avatar');
			const data = await res.json();
			if (data.expression !== currentExpression) {
				currentExpression = data.expression;
				const newSrc = expressionToVideo[currentExpression];
				if (newSrc !== videoSrc) {
					videoSrc = newSrc;
				}
			}
		} catch (err) {
			console.error('Failed to fetch avatar state:', err);
		}
	}

	async function setExpression(expression: Expression) {
		try {
			await fetch('/api/avatar', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ expression })
			});
			currentExpression = expression;
			videoSrc = expressionToVideo[expression];
		} catch (err) {
			console.error('Failed to set expression:', err);
		}
	}

	function toggleControls() {
		showControls = !showControls;
	}

	function handleKeydown(e: KeyboardEvent) {
		// Press 'c' to toggle controls
		if (e.key === 'c' || e.key === 'C') {
			toggleControls();
		}
		// Number keys 1-5 for quick expression switching
		const num = parseInt(e.key);
		if (num >= 1 && num <= 5) {
			setExpression(EXPRESSIONS[num - 1]);
		}
	}

	onMount(() => {
		// Initial fetch
		fetchState();

		// Poll for changes every 500ms
		pollInterval = setInterval(fetchState, 500);

		// Keyboard listener
		window.addEventListener('keydown', handleKeydown);
	});

	onDestroy(() => {
		if (pollInterval) clearInterval(pollInterval);
		window.removeEventListener('keydown', handleKeydown);
	});
</script>

<svelte:head>
	<title>Avatar - {currentExpression}</title>
</svelte:head>

<div class="avatar-container" role="button" tabindex="0" onclick={toggleControls} onkeydown={handleKeydown}>
	<video
		bind:this={videoElement}
		src={videoSrc}
		autoplay
		loop
		muted
		playsinline
		class="avatar-video"
	>
		<track kind="captions" />
	</video>

	{#if showControls}
		<div class="controls" role="toolbar" onclick={(e) => e.stopPropagation()} onkeydown={(e) => e.stopPropagation()}>
			<div class="controls-header">
				<span class="current-state">{currentExpression}</span>
				<button class="close-btn" onclick={toggleControls}>×</button>
			</div>
			<div class="expression-buttons">
				{#each EXPRESSIONS as expr, i}
					<button
						class="expr-btn"
						class:active={currentExpression === expr}
						onclick={() => setExpression(expr)}
					>
						<span class="hotkey">{i + 1}</span>
						{expr}
					</button>
				{/each}
			</div>
			<div class="hint">Press 'C' to hide controls, 1-5 for expressions</div>
		</div>
	{/if}
</div>

<style>
	.avatar-container {
		width: 100vw;
		height: 100vh;
		background: #000;
		position: relative;
		cursor: pointer;
		overflow: hidden;
	}

	.avatar-video {
		width: 100%;
		height: 100%;
		object-fit: cover;
	}

	.controls {
		position: absolute;
		bottom: 20px;
		left: 50%;
		transform: translateX(-50%);
		background: rgba(0, 0, 0, 0.85);
		border-radius: 12px;
		padding: 16px;
		min-width: 300px;
		backdrop-filter: blur(10px);
		border: 1px solid rgba(255, 255, 255, 0.1);
	}

	.controls-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 12px;
	}

	.current-state {
		font-size: 14px;
		color: #888;
		text-transform: uppercase;
		letter-spacing: 0.1em;
	}

	.close-btn {
		background: none;
		border: none;
		color: #888;
		font-size: 24px;
		cursor: pointer;
		padding: 0;
		line-height: 1;
	}

	.close-btn:hover {
		color: #fff;
	}

	.expression-buttons {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
		justify-content: center;
	}

	.expr-btn {
		background: rgba(255, 255, 255, 0.1);
		border: 1px solid rgba(255, 255, 255, 0.2);
		color: #fff;
		padding: 8px 16px;
		border-radius: 6px;
		cursor: pointer;
		font-size: 14px;
		text-transform: capitalize;
		transition: all 0.2s;
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.expr-btn:hover {
		background: rgba(255, 255, 255, 0.2);
	}

	.expr-btn.active {
		background: rgba(99, 102, 241, 0.6);
		border-color: rgba(99, 102, 241, 0.8);
	}

	.hotkey {
		background: rgba(0, 0, 0, 0.3);
		padding: 2px 6px;
		border-radius: 3px;
		font-size: 11px;
		font-family: monospace;
	}

	.hint {
		margin-top: 12px;
		font-size: 11px;
		color: #666;
		text-align: center;
	}
</style>
