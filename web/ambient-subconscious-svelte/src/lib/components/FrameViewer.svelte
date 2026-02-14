<script lang="ts">
  import type { Frame, Detection } from '$lib/types';
  import { timestampToDate, updateFrameDetections } from '$lib/stores/spacetime';
  import Icon from './Icon.svelte';

  interface Props {
    frame: Frame;
  }

  let { frame }: Props = $props();

  // Parse detections from JSON
  let detections = $derived.by((): Detection[] => {
    try {
      return JSON.parse(frame.detections);
    } catch {
      return [];
    }
  });

  let time = $derived.by(() => {
    const date = timestampToDate(frame.timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    });
  });

  // Editing state
  let editingDetectionIndex: number | null = $state(null);
  let editingLabel: string = $state('');

  function startEditing(index: number, currentLabel: string) {
    editingDetectionIndex = index;
    editingLabel = currentLabel;
  }

  function cancelEditing() {
    editingDetectionIndex = null;
    editingLabel = '';
  }

  function saveLabel(index: number) {
    if (!editingLabel.trim()) {
      cancelEditing();
      return;
    }

    // Update the detection
    const updatedDetections = [...detections];
    updatedDetections[index] = {
      ...updatedDetections[index],
      class: editingLabel.trim()
    };

    // Save to database
    updateFrameDetections(frame.id, JSON.stringify(updatedDetections));
    cancelEditing();
  }

  function handleKeyDown(event: KeyboardEvent, index: number) {
    if (event.key === 'Enter') {
      event.preventDefault();
      saveLabel(index);
    } else if (event.key === 'Escape') {
      cancelEditing();
    }
  }
</script>

<div class="frame-viewer">
  <div class="frame-header">
    <div class="frame-info">
      <span class="frame-type">{frame.frameType}</span>
      <span class="frame-time">{time}</span>
      {#if frame.reviewed}
        <span class="reviewed-badge" title="Reviewed">
          <Icon name="check" size={14} />
        </span>
      {/if}
    </div>
  </div>

  <div class="frame-body">
    <div class="image-container">
      <img src={"/" + frame.imagePath} alt="Captured frame" />
    </div>

    <div class="detections-panel">
      <h3 class="detections-title">
        Detected Objects ({detections.length})
      </h3>

      {#if detections.length === 0}
        <p class="no-detections">No objects detected</p>
      {:else}
        <ul class="detections-list">
          {#each detections as detection, i}
            <li class="detection-item">
              {#if editingDetectionIndex === i}
                <input
                  type="text"
                  class="label-input"
                  bind:value={editingLabel}
                  onkeydown={(e) => handleKeyDown(e, i)}
                  autofocus
                />
                <div class="edit-actions">
                  <button class="btn-save" onclick={() => saveLabel(i)}>
                    <Icon name="check" size={14} />
                  </button>
                  <button class="btn-cancel" onclick={cancelEditing}>
                    <Icon name="x" size={14} />
                  </button>
                </div>
              {:else}
                <button
                  class="detection-label"
                  onclick={() => startEditing(i, detection.class)}
                  title="Click to edit label"
                >
                  <span class="label-text">{detection.class}</span>
                  <span class="confidence">{Math.round(detection.confidence * 100)}%</span>
                  <Icon name="edit" size={12} />
                </button>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}

      {#if frame.notes}
        <div class="notes-section">
          <h4 class="notes-title">Notes</h4>
          <p class="notes-text">{frame.notes}</p>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .frame-viewer {
    display: flex;
    flex-direction: column;
    background: var(--clr-surface-a10);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }

  .frame-header {
    padding: var(--space-md);
    border-bottom: 1px solid var(--clr-surface-a20);
    background: var(--clr-surface-a5);
  }

  .frame-info {
    display: flex;
    align-items: center;
    gap: var(--space-md);
  }

  .frame-type {
    padding: var(--space-xs) var(--space-sm);
    background: var(--clr-primary-a10);
    color: var(--clr-primary-a0);
    font-size: var(--text-xs);
    font-weight: 600;
    text-transform: uppercase;
    border-radius: var(--radius-sm);
  }

  .frame-time {
    font-size: var(--text-sm);
    color: var(--clr-text-a60);
  }

  .reviewed-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    background: var(--clr-success-a10);
    color: var(--clr-success-a0);
    border-radius: 50%;
  }

  .frame-body {
    display: flex;
    flex-direction: column;
    gap: var(--space-lg);
    padding: var(--space-lg);
  }

  .image-container {
    width: 100%;
    background: var(--clr-surface-a20);
    border-radius: var(--radius-md);
    overflow: hidden;
  }

  .image-container img {
    width: 100%;
    height: auto;
    display: block;
  }

  .detections-panel {
    display: flex;
    flex-direction: column;
    gap: var(--space-md);
  }

  .detections-title {
    font-size: var(--text-md);
    font-weight: 600;
    color: var(--clr-text-a80);
  }

  .no-detections {
    padding: var(--space-lg);
    text-align: center;
    color: var(--clr-text-a40);
    font-size: var(--text-sm);
  }

  .detections-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-sm);
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .detection-item {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
  }

  .detection-label {
    flex: 1;
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-sm) var(--space-md);
    background: var(--clr-surface-a20);
    border: 1px solid var(--clr-surface-a30);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all 0.2s;
  }

  .detection-label:hover {
    background: var(--clr-surface-a30);
    border-color: var(--clr-primary-a30);
  }

  .label-text {
    flex: 1;
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--clr-text-a80);
  }

  .confidence {
    font-size: var(--text-xs);
    color: var(--clr-text-a60);
    font-weight: 600;
  }

  .label-input {
    flex: 1;
    padding: var(--space-sm) var(--space-md);
    background: var(--clr-surface-a30);
    border: 2px solid var(--clr-primary-a50);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
    color: var(--clr-text-a90);
  }

  .label-input:focus {
    outline: none;
    border-color: var(--clr-primary-a0);
  }

  .edit-actions {
    display: flex;
    gap: var(--space-xs);
  }

  .btn-save,
  .btn-cancel {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: var(--radius-sm);
    transition: all 0.2s;
  }

  .btn-save {
    background: var(--clr-success-a10);
    color: var(--clr-success-a0);
  }

  .btn-save:hover {
    background: var(--clr-success-a20);
  }

  .btn-cancel {
    background: var(--clr-surface-a30);
    color: var(--clr-text-a60);
  }

  .btn-cancel:hover {
    background: var(--clr-surface-a40);
  }

  .notes-section {
    margin-top: var(--space-md);
    padding: var(--space-md);
    background: var(--clr-surface-a20);
    border-radius: var(--radius-md);
  }

  .notes-title {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--clr-text-a70);
    margin-bottom: var(--space-sm);
  }

  .notes-text {
    font-size: var(--text-sm);
    color: var(--clr-text-a80);
    line-height: 1.5;
  }

  @media (min-width: 768px) {
    .frame-body {
      flex-direction: row;
    }

    .image-container {
      flex: 2;
    }

    .detections-panel {
      flex: 1;
      min-width: 300px;
    }
  }
</style>
