<script lang="ts">
  import {
    selectedEntry,
    speakers,
    updateEntry,
    markReviewed,
    deleteEntry,
    createSpeaker,
    selectedEntryId,
    timestampToDate
  } from '$lib/stores/spacetime';
  import AudioPlayer from './AudioPlayer.svelte';
  import Icon from './Icon.svelte';
  import { fade, fly } from 'svelte/transition';

  // Local state for editing
  let editedTranscript = $state('');
  let editedSpeaker = $state('');
  let editedSentiment = $state('');
  let editedNotes = $state('');
  let newSpeakerName = $state('');
  let showNewSpeaker = $state(false);

  // Sync local state when selected entry changes
  $effect(() => {
    if ($selectedEntry) {
      editedTranscript = $selectedEntry.transcript ?? '';
      editedSpeaker = $selectedEntry.speaker ?? '';
      editedSentiment = $selectedEntry.sentiment ?? '';
      editedNotes = $selectedEntry.notes ?? '';
    }
  });

  // Derived values
  let timestamp = $derived.by(() => {
    if (!$selectedEntry) return '';
    const date = timestampToDate($selectedEntry.timestamp);
    return date.toLocaleString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  });

  let confidence = $derived.by(() => {
    if (!$selectedEntry?.confidence) return null;
    return Math.round($selectedEntry.confidence * 100);
  });

  let duration = $derived.by(() => {
    if (!$selectedEntry) return '0s';
    const seconds = $selectedEntry.durationMs / 1000;
    return seconds >= 60
      ? `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
      : `${seconds.toFixed(1)}s`;
  });

  // Handlers
  function handleSave() {
    if (!$selectedEntry) return;

    updateEntry($selectedEntry.id, {
      transcript: editedTranscript || undefined,
      speaker: editedSpeaker || undefined,
      sentiment: editedSentiment || undefined,
      notes: editedNotes || undefined
    });
  }

  function handleMarkReviewed() {
    if (!$selectedEntry) return;
    markReviewed($selectedEntry.id, !$selectedEntry.reviewed);
  }

  function handleDelete() {
    if (!$selectedEntry) return;
    if (confirm('Are you sure you want to delete this entry?')) {
      deleteEntry($selectedEntry.id);
      selectedEntryId.set(null);
    }
  }

  function handleCreateSpeaker() {
    if (!newSpeakerName.trim()) return;
    createSpeaker(newSpeakerName.trim());
    editedSpeaker = newSpeakerName.trim();
    newSpeakerName = '';
    showNewSpeaker = false;
    handleSave();
  }

  function handleSpeakerChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    if (target.value === '__new__') {
      showNewSpeaker = true;
      editedSpeaker = '';
    } else {
      editedSpeaker = target.value;
      showNewSpeaker = false;
      handleSave();
    }
  }

  function handleTranscriptBlur() {
    if ($selectedEntry && editedTranscript !== ($selectedEntry.transcript ?? '')) {
      handleSave();
    }
  }

  function handleNotesBlur() {
    if ($selectedEntry && editedNotes !== ($selectedEntry.notes ?? '')) {
      handleSave();
    }
  }

  const sentimentOptions = [
    { value: '', label: 'Not set' },
    { value: 'positive', label: 'Positive' },
    { value: 'neutral', label: 'Neutral' },
    { value: 'negative', label: 'Negative' }
  ];
</script>

<div class="entry-editor">
  {#if !$selectedEntry}
    <div class="empty-state" transition:fade={{ duration: 200 }}>
      <p>Select an entry to view details</p>
    </div>
  {:else}
    <div class="editor-content" transition:fly={{ x: 20, duration: 200 }}>
      <header class="editor-header">
        <div class="timestamp">{timestamp}</div>
        <div class="meta-badges">
          <span class="badge badge-neutral">{duration}</span>
          {#if confidence !== null}
            <span
              class="badge"
              class:badge-success={confidence >= 90}
              class:badge-warning={confidence >= 70 && confidence < 90}
              class:badge-danger={confidence < 70}
            >
              {confidence}% confidence
            </span>
          {/if}
          {#if $selectedEntry.reviewed}
            <span class="badge badge-success">
              <Icon name="check" size={12} />
              Reviewed
            </span>
          {/if}
        </div>
      </header>

      <div class="editor-section">
        <label class="label" for="transcript">Transcript</label>
        <textarea
          id="transcript"
          class="textarea"
          bind:value={editedTranscript}
          onblur={handleTranscriptBlur}
          placeholder="No transcript available"
          rows={4}
        ></textarea>
      </div>

      <div class="editor-section">
        <AudioPlayer
          audioPath={$selectedEntry.audioClipPath}
          durationMs={$selectedEntry.durationMs}
        />
      </div>

      <div class="editor-row">
        <div class="editor-section half">
          <label class="label" for="speaker">Speaker</label>
          {#if showNewSpeaker}
            <div class="new-speaker-form">
              <input
                type="text"
                class="input"
                bind:value={newSpeakerName}
                placeholder="Enter speaker name"
              />
              <button class="btn btn-primary btn-sm" onclick={handleCreateSpeaker}>
                Add
              </button>
              <button
                class="btn btn-ghost btn-sm"
                onclick={() => {
                  showNewSpeaker = false;
                  newSpeakerName = '';
                }}
              >
                Cancel
              </button>
            </div>
          {:else}
            <select id="speaker" class="select" value={editedSpeaker} onchange={handleSpeakerChange}>
              <option value="">Not assigned</option>
              {#each $speakers as speaker}
                <option value={speaker.name}>{speaker.name}</option>
              {/each}
              <option value="__new__">+ New speaker...</option>
            </select>
          {/if}
        </div>

        <div class="editor-section half">
          <label class="label" for="sentiment">Sentiment</label>
          <select
            id="sentiment"
            class="select"
            bind:value={editedSentiment}
            onchange={handleSave}
          >
            {#each sentimentOptions as opt}
              <option value={opt.value}>{opt.label}</option>
            {/each}
          </select>
        </div>
      </div>

      <div class="editor-section">
        <label class="label" for="notes">Notes</label>
        <textarea
          id="notes"
          class="textarea"
          bind:value={editedNotes}
          onblur={handleNotesBlur}
          placeholder="Add notes..."
          rows={2}
        ></textarea>
      </div>

      <div class="divider"></div>

      <div class="editor-actions">
        <button
          class="btn"
          class:btn-secondary={$selectedEntry.reviewed}
          class:btn-primary={!$selectedEntry.reviewed}
          onclick={handleMarkReviewed}
        >
          <Icon name="check" size={16} />
          {$selectedEntry.reviewed ? 'Unmark Reviewed' : 'Mark Reviewed'}
        </button>

        <button class="btn btn-danger" onclick={handleDelete}>
          <Icon name="trash" size={16} />
          Delete
        </button>
      </div>
    </div>
  {/if}
</div>

<style>
  .entry-editor {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: var(--space-lg);
  }

  .empty-state {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--clr-surface-a50);
    font-size: var(--text-sm);
  }

  .editor-content {
    max-width: 640px;
    width: 100%;
  }

  .editor-header {
    margin-bottom: var(--space-lg);
  }

  .timestamp {
    font-size: var(--text-lg);
    font-weight: 600;
    color: var(--clr-light);
    margin-bottom: var(--space-sm);
  }

  .meta-badges {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-sm);
  }

  .editor-section {
    margin-bottom: var(--space-md);
  }

  .editor-row {
    display: flex;
    gap: var(--space-md);
  }

  .editor-section.half {
    flex: 1;
  }

  .new-speaker-form {
    display: flex;
    gap: var(--space-sm);
    align-items: center;
  }

  .new-speaker-form .input {
    flex: 1;
  }

  .btn-sm {
    padding: var(--space-xs) var(--space-sm);
    font-size: var(--text-xs);
  }

  .editor-actions {
    display: flex;
    gap: var(--space-md);
    justify-content: space-between;
  }

  @media (max-width: 768px) {
    .entry-editor {
      padding: var(--space-md);
    }

    .editor-row {
      flex-direction: column;
    }

    .editor-actions {
      flex-direction: column;
    }

    .editor-actions .btn {
      width: 100%;
    }
  }
</style>
