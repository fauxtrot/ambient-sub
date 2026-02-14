<script lang="ts">
  import Icon from './Icon.svelte';

  interface Props {
    audioPath?: string | null;
    durationMs: number;
  }

  let { audioPath, durationMs }: Props = $props();

  let audioElement: HTMLAudioElement | null = $state(null);
  let isPlaying = $state(false);
  let currentTime = $state(0);
  let audioDuration = $state<number | null>(null);

  // Use audio element duration if available, otherwise fall back to prop
  let duration = $derived(audioDuration ?? durationMs / 1000);

  // Format time as mm:ss or ss.s
  function formatTime(seconds: number): string {
    if (seconds >= 60) {
      const mins = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    return `${seconds.toFixed(1)}s`;
  }

  let progress = $derived(duration > 0 ? (currentTime / duration) * 100 : 0);

  function togglePlay() {
    if (!audioElement) return;

    if (isPlaying) {
      audioElement.pause();
    } else {
      audioElement.play();
    }
  }

  function handleTimeUpdate() {
    if (audioElement) {
      currentTime = audioElement.currentTime;
    }
  }

  function handleLoadedMetadata() {
    if (audioElement && !isNaN(audioElement.duration)) {
      audioDuration = audioElement.duration;
    }
  }

  function handleEnded() {
    isPlaying = false;
    currentTime = 0;
    if (audioElement) {
      audioElement.currentTime = 0;
    }
  }

  function handlePlay() {
    isPlaying = true;
  }

  function handlePause() {
    isPlaying = false;
  }

  function seek(event: MouseEvent) {
    if (!audioElement) return;

    const target = event.currentTarget as HTMLDivElement;
    const rect = target.getBoundingClientRect();

    // Guard against zero-width container
    if (rect.width === 0) return;

    const percent = (event.clientX - rect.left) / rect.width;
    const newTime = percent * duration;

    // Guard against non-finite values
    if (!Number.isFinite(newTime) || newTime < 0) return;

    audioElement.currentTime = newTime;
    currentTime = newTime;
  }

  // Build audio URL from path - use Vite dev server
  // The audioClipPath may be a full path or relative path
  let audioUrl = $derived.by(() => {
    if (!audioPath) return null;
    // Normalize path separators
    const normalizedPath = audioPath.replace(/\\/g, '/');
    // Find the /audio/ marker and extract everything after it
    const audioMarker = '/audio/';
    const markerIndex = normalizedPath.toLowerCase().indexOf(audioMarker);
    const relativePath = markerIndex !== -1
      ? normalizedPath.slice(markerIndex + audioMarker.length)
      : normalizedPath;
    return `/audio/${relativePath}`;
  });
</script>

<div class="audio-player" class:disabled={!audioPath}>
  {#if audioUrl}
    <audio
      bind:this={audioElement}
      src={audioUrl}
      ontimeupdate={handleTimeUpdate}
      onloadedmetadata={handleLoadedMetadata}
      onended={handleEnded}
      onplay={handlePlay}
      onpause={handlePause}
      preload="metadata"
    ></audio>
  {/if}

  <button
    class="play-btn"
    onclick={togglePlay}
    disabled={!audioPath}
    aria-label={isPlaying ? 'Pause' : 'Play'}
  >
    <Icon name={isPlaying ? 'pause' : 'play'} size={16} />
  </button>

  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="progress-container" onclick={audioPath ? seek : undefined}>
    <div class="progress-bar">
      <div class="progress-fill" style="width: {progress}%"></div>
    </div>
  </div>

  <div class="time-display">
    <span class="current">{formatTime(currentTime)}</span>
    <span class="separator">/</span>
    <span class="total">{formatTime(duration)}</span>
  </div>
</div>

<style>
  .audio-player {
    display: flex;
    align-items: center;
    gap: var(--space-md);
    padding: var(--space-md);
    background: var(--clr-surface-a10);
    border-radius: var(--radius-lg);
  }

  .audio-player.disabled {
    opacity: 0.5;
  }

  .play-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: var(--clr-primary-a0);
    border-radius: 50%;
    color: var(--clr-dark);
    transition: all var(--transition-fast);
    flex-shrink: 0;
  }

  .play-btn:hover:not(:disabled) {
    background: var(--clr-primary-a10);
    transform: scale(1.05);
  }

  .play-btn:disabled {
    background: var(--clr-surface-a20);
    color: var(--clr-surface-a40);
    cursor: not-allowed;
  }

  .progress-container {
    flex: 1;
    cursor: pointer;
    padding: var(--space-xs) 0;
  }

  .progress-bar {
    height: 4px;
    background: var(--clr-surface-a20);
    border-radius: var(--radius-full);
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--clr-primary-a0);
    border-radius: var(--radius-full);
    transition: width 0.1s linear;
  }

  .time-display {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    color: var(--clr-surface-a50);
    white-space: nowrap;
  }

  .current {
    color: var(--clr-light);
  }

  .separator {
    color: var(--clr-surface-a30);
  }
</style>
