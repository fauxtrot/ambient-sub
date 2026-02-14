<script lang="ts">
  import { selectedDate, availableDates, selectedEntryId } from '$lib/stores/spacetime';
  import Icon from './Icon.svelte';

  // Generate dates around the selected date
  let visibleDates = $derived.by(() => {
    const current = new Date($selectedDate);
    const dates: string[] = [];

    // Show 3 days before and after
    for (let i = -3; i <= 3; i++) {
      const date = new Date(current);
      date.setDate(date.getDate() + i);
      dates.push(date.toISOString().split('T')[0]);
    }

    return dates;
  });

  // Current time display
  let currentTime = $state('');

  function updateTime() {
    const now = new Date();
    currentTime = now.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  }

  $effect(() => {
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  });

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr + 'T00:00:00');
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
  }

  function isToday(dateStr: string): boolean {
    return dateStr === new Date().toISOString().split('T')[0];
  }

  function navigate(days: number) {
    const current = new Date($selectedDate);
    current.setDate(current.getDate() + days);
    selectedDate.set(current.toISOString().split('T')[0]);
    selectedEntryId.set(null);
  }

  function selectDate(dateStr: string) {
    selectedDate.set(dateStr);
    selectedEntryId.set(null);
  }

  function hasEntries(dateStr: string): boolean {
    return $availableDates.includes(dateStr);
  }
</script>

<div class="timeline">
  <button class="nav-btn" onclick={() => navigate(-7)} aria-label="Previous week">
    <Icon name="chevron-left" size={18} />
  </button>

  <div class="dates">
    {#each visibleDates as dateStr}
      <button
        class="date-btn"
        class:selected={dateStr === $selectedDate}
        class:today={isToday(dateStr)}
        class:has-entries={hasEntries(dateStr)}
        onclick={() => selectDate(dateStr)}
      >
        <span class="date-label">{formatDate(dateStr)}</span>
        {#if hasEntries(dateStr)}
          <span class="entry-dot"></span>
        {/if}
      </button>
    {/each}
  </div>

  <button class="nav-btn" onclick={() => navigate(7)} aria-label="Next week">
    <Icon name="chevron-right" size={18} />
  </button>

  <div class="current-time">
    <Icon name="clock" size={14} />
    <span>{currentTime}</span>
  </div>
</div>

<style>
  .timeline {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-sm) var(--space-md);
    background: var(--clr-surface-a10);
    border-bottom: 1px solid var(--clr-surface-a20);
    min-height: var(--timeline-height);
  }

  .nav-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--space-xs);
    border-radius: var(--radius-md);
    color: var(--clr-surface-a50);
    transition: all var(--transition-fast);
  }

  .nav-btn:hover {
    background: var(--clr-surface-a20);
    color: var(--clr-light);
  }

  .dates {
    display: flex;
    gap: var(--space-xs);
    flex: 1;
    justify-content: center;
    overflow-x: auto;
  }

  .date-btn {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: var(--space-xs) var(--space-md);
    border-radius: var(--radius-md);
    color: var(--clr-surface-a50);
    font-size: var(--text-sm);
    transition: all var(--transition-fast);
    white-space: nowrap;
  }

  .date-btn:hover {
    background: var(--clr-surface-a20);
    color: var(--clr-light);
  }

  .date-btn.selected {
    background: var(--clr-primary-a0);
    color: var(--clr-dark);
  }

  .date-btn.today:not(.selected) {
    border: 1px solid var(--clr-primary-a0);
  }

  .entry-dot {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.6;
  }

  .date-btn.selected .entry-dot {
    background: var(--clr-dark);
    opacity: 1;
  }

  .current-time {
    display: flex;
    align-items: center;
    gap: var(--space-xs);
    padding: var(--space-xs) var(--space-md);
    color: var(--clr-surface-a50);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
    border-left: 1px solid var(--clr-surface-a20);
    margin-left: var(--space-sm);
  }

  @media (max-width: 768px) {
    .dates {
      gap: 2px;
    }

    .date-btn {
      padding: var(--space-xs) var(--space-sm);
      font-size: var(--text-xs);
    }
  }
</style>
