<script lang="ts">
  import type { Frame, Detection } from '$lib/types';
  import { timestampToDate } from '$lib/stores/spacetime';
  import Icon from './Icon.svelte';

  interface Props {
    frame: Frame;
    selected?: boolean;
    onclick?: () => void;
  }

  let { frame, selected = false, onclick }: Props = $props();

  let time = $derived.by(() => {
    const date = timestampToDate(frame.timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  });

  let detectionCount = $derived.by(() => {
    try {
      const detections: Detection[] = JSON.parse(frame.detections);
      return detections.length;
    } catch {
      return 0;
    }
  });

  let topDetections = $derived.by(() => {
    try {
      const detections: Detection[] = JSON.parse(frame.detections);
      return detections
        .slice(0, 3)
        .map(d => d.class)
        .join(', ');
    } catch {
      return 'No detections';
    }
  });
</script>

<button class="frame-card" class:selected onclick={onclick}>
  <div class="thumbnail-col">
    <img src={"/" + frame.imagePath} alt="Frame thumbnail" class="thumbnail" />
    <span class="frame-type-badge">{frame.frameType}</span>
  </div>

  <div class="content-col">
    <div class="frame-header-row">
      <span class="time">{time}</span>
      {#if frame.reviewed}
        <span class="reviewed-badge" title="Reviewed">
          <Icon name="check" size={12} />
        </span>
      {/if}
    </div>

    <p class="detections-summary" class:empty={detectionCount === 0}>
      {topDetections}
    </p>

    <div class="meta">
      <span class="detection-count">
        <Icon name="box" size={12} />
        {detectionCount} {detectionCount === 1 ? 'object' : 'objects'}
      </span>
    </div>
  </div>
</button>

<style>
  .frame-card {
    display: flex;
    gap: var(--space-md);
    width: 100%;
    padding: var(--space-md);
    background: var(--clr-surface-a10);
    border: 1px solid transparent;
    border-radius: var(--radius-md);
    text-align: left;
    transition: all 0.2s;
    cursor: pointer;
  }

  .frame-card:hover {
    background: var(--clr-surface-a20);
    border-color: var(--clr-surface-a30);
  }

  .frame-card.selected {
    background: var(--clr-primary-a10);
    border-color: var(--clr-primary-a30);
  }

  .thumbnail-col {
    position: relative;
    flex-shrink: 0;
    width: 80px;
    height: 60px;
    border-radius: var(--radius-sm);
    overflow: hidden;
    background: var(--clr-surface-a30);
  }

  .thumbnail {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .frame-type-badge {
    position: absolute;
    bottom: 2px;
    right: 2px;
    padding: 2px 4px;
    background: rgba(0, 0, 0, 0.7);
    color: white;
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    border-radius: 2px;
  }

  .content-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: var(--space-xs);
    min-width: 0;
  }

  .frame-header-row {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
  }

  .time {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--clr-text-a80);
  }

  .reviewed-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    background: var(--clr-success-a10);
    color: var(--clr-success-a0);
    border-radius: 50%;
  }

  .detections-summary {
    font-size: var(--text-sm);
    color: var(--clr-text-a70);
    margin: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .detections-summary.empty {
    color: var(--clr-text-a40);
    font-style: italic;
  }

  .meta {
    display: flex;
    align-items: center;
    gap: var(--space-md);
    font-size: var(--text-xs);
    color: var(--clr-text-a60);
  }

  .detection-count {
    display: flex;
    align-items: center;
    gap: var(--space-xs);
  }
</style>
