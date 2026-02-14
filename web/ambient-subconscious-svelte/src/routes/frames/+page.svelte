<script lang="ts">
  import Timeline from '$lib/components/Timeline.svelte';
  import FrameList from '$lib/components/FrameList.svelte';
  import FrameViewer from '$lib/components/FrameViewer.svelte';
  import { frames, selectedFrameId } from '$lib/stores/spacetime';

  // Track if we're on mobile with a frame selected
  let showViewer = $derived($selectedFrameId !== null);

  // Get selected frame
  let selectedFrame = $derived.by(() => {
    if ($selectedFrameId === null) return null;
    return $frames.find(f => f.id === $selectedFrameId) ?? null;
  });
</script>

<div class="page">
  <Timeline />

  <div class="content">
    <div class="list-panel" class:hidden-mobile={showViewer}>
      <FrameList />
    </div>

    <div class="viewer-panel" class:hidden-mobile={!showViewer}>
      {#if showViewer}
        <button
          class="back-btn mobile-only"
          onclick={() => selectedFrameId.set(null)}
        >
          ← Back to list
        </button>
      {/if}

      {#if selectedFrame}
        <div class="viewer-container">
          <FrameViewer frame={selectedFrame} />
        </div>
      {:else}
        <div class="empty-viewer">
          <p>Select a frame to view details</p>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .page {
    display: flex;
    flex-direction: column;
    flex: 1;
    overflow: hidden;
  }

  .content {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  .list-panel {
    display: flex;
    flex-shrink: 0;
    overflow: hidden;
  }

  .viewer-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    background: var(--clr-surface-tonal-a0);
  }

  .back-btn {
    display: none;
    padding: var(--space-md);
    color: var(--clr-primary-a0);
    font-size: var(--text-sm);
    font-weight: 500;
    text-align: left;
    border-bottom: 1px solid var(--clr-surface-a20);
    background: var(--clr-surface-a10);
  }

  .back-btn:hover {
    background: var(--clr-surface-a20);
  }

  .viewer-container {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-lg);
  }

  .empty-viewer {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
    color: var(--clr-text-a40);
  }

  @media (max-width: 768px) {
    .content {
      flex-direction: column;
    }

    .list-panel,
    .viewer-panel {
      width: 100%;
    }

    .hidden-mobile {
      display: none;
    }

    .mobile-only {
      display: block;
    }

    .back-btn {
      display: block;
    }
  }
</style>
