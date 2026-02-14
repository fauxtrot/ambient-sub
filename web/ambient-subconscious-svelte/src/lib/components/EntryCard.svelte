<script lang="ts">
  import type { TranscriptEntry } from '$lib/types';
  import { timestampToDate } from '$lib/stores/spacetime';
  import Icon from './Icon.svelte';

  interface Props {
    entry: TranscriptEntry;
    selected?: boolean;
    onclick?: () => void;
  }

  let { entry, selected = false, onclick }: Props = $props();

  let time = $derived.by(() => {
    const date = timestampToDate(entry.timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  });

  let duration = $derived.by(() => {
    const seconds = entry.durationMs / 1000;
    return seconds >= 60
      ? `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
      : `${seconds.toFixed(1)}s`;
  });

  let confidence = $derived(
    entry.confidence !== null && entry.confidence !== undefined
      ? Math.round(entry.confidence * 100)
      : null
  );

  let truncatedTranscript = $derived.by(() => {
    const text = entry.transcript || '(No transcript)';
    return text.length > 100 ? text.slice(0, 100) + '...' : text;
  });
</script>

<button class="entry-card" class:selected onclick={onclick}>
  <div class="time-col">
    <span class="time">{time}</span>
    {#if entry.reviewed}
      <span class="reviewed-badge" title="Reviewed">
        <Icon name="check" size={12} />
      </span>
    {/if}
  </div>

  <div class="content-col">
    <p class="transcript" class:empty={!entry.transcript}>{truncatedTranscript}</p>

    <div class="meta">
      {#if entry.speaker}
        <span class="speaker">
          <Icon name="user" size={12} />
          {entry.speaker}
        </span>
      {/if}

      <span class="duration">{duration}</span>

      {#if confidence !== null}
        <span class="confidence" class:low={confidence < 70} class:medium={confidence >= 70 && confidence < 90}>
          {confidence}%
        </span>
      {/if}
    </div>
  </div>
</button>

<style>
  .entry-card {
    display: flex;
    gap: var(--space-md);
    width: 100%;
    padding: var(--space-md);
    background: var(--clr-surface-a10);
    border: 1px solid transparent;
    border-radius: var(--radius-lg);
    text-align: left;
    cursor: pointer;
    transition: all var(--transition-fast);
  }

  .entry-card:hover {
    background: var(--clr-surface-tonal-a0);
    border-color: var(--clr-surface-a20);
  }

  .entry-card.selected {
    background: var(--clr-surface-tonal-a10);
    border-color: var(--clr-primary-a0);
  }

  .time-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--space-xs);
    min-width: 60px;
  }

  .time {
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--clr-light);
    font-variant-numeric: tabular-nums;
  }

  .reviewed-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    background: var(--clr-success-a0);
    border-radius: 50%;
    color: var(--clr-light);
  }

  .content-col {
    flex: 1;
    min-width: 0;
  }

  .transcript {
    font-size: var(--text-sm);
    color: var(--clr-light);
    line-height: var(--leading-relaxed);
    margin-bottom: var(--space-sm);
  }

  .transcript.empty {
    color: var(--clr-surface-a40);
    font-style: italic;
  }

  .meta {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-sm);
    font-size: var(--text-xs);
    color: var(--clr-surface-a50);
  }

  .speaker {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px var(--space-xs);
    background: var(--clr-surface-a20);
    border-radius: var(--radius-sm);
    color: var(--clr-primary-a10);
  }

  .duration {
    font-variant-numeric: tabular-nums;
  }

  .confidence {
    font-weight: 500;
    color: var(--clr-success-a10);
  }

  .confidence.medium {
    color: var(--clr-warning-a10);
  }

  .confidence.low {
    color: var(--clr-danger-a10);
  }
</style>
