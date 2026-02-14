<script lang="ts">
  import { entriesForDate, selectedEntryId, isConnected, selectedDate } from '$lib/stores/spacetime';
  import EntryCard from './EntryCard.svelte';
  import { fade } from 'svelte/transition';

  function formatSelectedDate(): string {
    const date = new Date($selectedDate + 'T00:00:00');
    return date.toLocaleDateString('en-US', {
      weekday: 'long',
      month: 'long',
      day: 'numeric'
    });
  }

  function selectEntry(id: number) {
    selectedEntryId.set(id);
  }
</script>

<div class="entry-list">
  <header class="list-header">
    <h2 class="date-title">{formatSelectedDate()}</h2>
    <span class="count">{$entriesForDate.length} entries</span>
  </header>

  <div class="list-content">
    {#if !$isConnected}
      <div class="empty-state" transition:fade={{ duration: 200 }}>
        <p>Connecting to SpacetimeDB...</p>
      </div>
    {:else if $entriesForDate.length === 0}
      <div class="empty-state" transition:fade={{ duration: 200 }}>
        <p>No entries for this date</p>
        <span class="hint">Entries will appear here when audio is captured</span>
      </div>
    {:else}
      <div class="entries">
        {#each $entriesForDate as entry (entry.id)}
          <div transition:fade={{ duration: 150 }}>
            <EntryCard
              {entry}
              selected={$selectedEntryId === entry.id}
              onclick={() => selectEntry(entry.id)}
            />
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .entry-list {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--clr-surface-a0);
    border-right: 1px solid var(--clr-surface-a20);
    min-width: var(--sidebar-width);
    max-width: var(--sidebar-width);
  }

  .list-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-md);
    border-bottom: 1px solid var(--clr-surface-a20);
  }

  .date-title {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--clr-light);
  }

  .count {
    font-size: var(--text-xs);
    color: var(--clr-surface-a50);
  }

  .list-content {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-sm);
  }

  .entries {
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    text-align: center;
    color: var(--clr-surface-a50);
  }

  .empty-state p {
    font-size: var(--text-sm);
  }

  .empty-state .hint {
    font-size: var(--text-xs);
    margin-top: var(--space-xs);
    color: var(--clr-surface-a40);
  }

  @media (max-width: 768px) {
    .entry-list {
      min-width: 100%;
      max-width: 100%;
      border-right: none;
    }
  }
</style>
