<script lang="ts">
  import Timeline from '$lib/components/Timeline.svelte';
  import EntryList from '$lib/components/EntryList.svelte';
  import EntryEditor from '$lib/components/EntryEditor.svelte';
  import { selectedEntryId } from '$lib/stores/spacetime';

  // Track if we're on mobile with an entry selected
  let showEditor = $derived($selectedEntryId !== null);
</script>

<div class="page">
  <Timeline />

  <div class="content">
    <div class="list-panel" class:hidden-mobile={showEditor}>
      <EntryList />
    </div>

    <div class="editor-panel" class:hidden-mobile={!showEditor}>
      {#if showEditor}
        <button
          class="back-btn mobile-only"
          onclick={() => selectedEntryId.set(null)}
        >
          Back to list
        </button>
      {/if}
      <EntryEditor />
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

  .editor-panel {
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

  @media (max-width: 768px) {
    .content {
      flex-direction: column;
    }

    .list-panel,
    .editor-panel {
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
