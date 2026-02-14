<script lang="ts">
  import { frames, selectedFrameId, selectedDate, timestampToDate } from '$lib/stores/spacetime';
  import FrameCard from './FrameCard.svelte';

  // Filter frames by selected date
  let framesForDate = $derived.by(() => {
    return $frames
      .filter(f => {
        const frameDate = timestampToDate(f.timestamp)
          .toISOString()
          .split('T')[0];
        return frameDate === $selectedDate;
      })
      .sort((a, b) =>
        timestampToDate(b.timestamp).getTime() - timestampToDate(a.timestamp).getTime()
      );
  });

  function selectFrame(frameId: number) {
    selectedFrameId.set(frameId);
  }
</script>

<div class="frame-list">
  <div class="list-header">
    <h2 class="list-title">Frames ({framesForDate.length})</h2>
  </div>

  <div class="list-body">
    {#if framesForDate.length === 0}
      <div class="empty-state">
        <p>No frames captured for this date</p>
      </div>
    {:else}
      <div class="frames-grid">
        {#each framesForDate as frame (frame.id)}
          <FrameCard
            {frame}
            selected={$selectedFrameId === frame.id}
            onclick={() => selectFrame(frame.id)}
          />
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .frame-list {
    display: flex;
    flex-direction: column;
    width: 100%;
    max-width: 1000px;
    height: 100%;
    overflow: hidden;
  }

  .list-header {
    padding: var(--space-lg);
    border-bottom: 1px solid var(--clr-surface-a20);
    background: var(--clr-surface-a5);
  }

  .list-title {
    font-size: var(--text-lg);
    font-weight: 700;
    color: var(--clr-text-a90);
    margin: 0;
  }

  .list-body {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-lg);
  }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    padding: var(--space-xl);
    text-align: center;
    color: var(--clr-text-a40);
  }

  .frames-grid {
    display: flex;
    flex-direction: column;
    gap: var(--space-md);
  }
</style>
