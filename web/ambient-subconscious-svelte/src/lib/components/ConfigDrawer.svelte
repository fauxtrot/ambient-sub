<script lang="ts">
  import {
    drawerOpen,
    currentDevice,
    currentConfig,
    inputsForDevice,
    isConnected,
    setSilenceThreshold,
    setAudioInput,
    setCapturePaused,
    setCaptureMuted,
    updateSpeakerEmbeddingConfig
  } from '$lib/stores/spacetime';
  import Icon from './Icon.svelte';
  import { slide, fade } from 'svelte/transition';

  // Local state for threshold slider
  let thresholdValue = $state(0.01);

  // Sync with config
  $effect(() => {
    if ($currentConfig) {
      thresholdValue = $currentConfig.silenceThreshold;
    }
  });

  function closeDrawer() {
    drawerOpen.set(false);
  }

  function handleThresholdChange(event: Event) {
    const target = event.target as HTMLInputElement;
    thresholdValue = parseFloat(target.value);
  }

  function handleThresholdCommit() {
    if ($currentDevice) {
      setSilenceThreshold($currentDevice.id, thresholdValue);
    }
  }

  function handleAudioInputChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    if ($currentDevice) {
      setAudioInput($currentDevice.id, parseInt(target.value, 10));
    }
  }

  function handlePauseToggle() {
    if ($currentDevice && $currentConfig) {
      setCapturePaused($currentDevice.id, !$currentConfig.isPaused);
    }
  }

  function handleMuteToggle() {
    if ($currentDevice && $currentConfig) {
      setCaptureMuted($currentDevice.id, !$currentConfig.isMuted);
    }
  }

  let isPaused = $derived($currentConfig?.isPaused ?? false);
  let isMuted = $derived($currentConfig?.isMuted ?? false);

  // Speaker embedding state
  let speakerEmbeddingEnabled = $state(false);
  let speakerMatchThreshold = $state(0.7);
  let createUnknownSpeakers = $state(true);

  $effect(() => {
    if ($currentConfig) {
      speakerEmbeddingEnabled = $currentConfig.speakerEmbeddingEnabled;
      speakerMatchThreshold = $currentConfig.speakerMatchThreshold;
      createUnknownSpeakers = $currentConfig.createUnknownSpeakers;
    }
  });

  function handleSpeakerEmbeddingToggle() {
    speakerEmbeddingEnabled = !speakerEmbeddingEnabled;
    commitSpeakerEmbeddingConfig();
  }

  function handleCreateUnknownToggle() {
    createUnknownSpeakers = !createUnknownSpeakers;
    commitSpeakerEmbeddingConfig();
  }

  function handleThresholdSliderChange(event: Event) {
    const target = event.target as HTMLInputElement;
    speakerMatchThreshold = parseFloat(target.value);
  }

  function handleThresholdSliderCommit() {
    commitSpeakerEmbeddingConfig();
  }

  function commitSpeakerEmbeddingConfig() {
    if ($currentDevice && $currentConfig) {
      updateSpeakerEmbeddingConfig(
        $currentDevice.id,
        speakerEmbeddingEnabled,
        $currentConfig.speakerEmbeddingModel || '3dspeaker_speech_eres2net_base',
        speakerMatchThreshold,
        createUnknownSpeakers
      );
    }
  }
</script>

{#if $drawerOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="drawer-backdrop" onclick={closeDrawer} transition:fade={{ duration: 200 }}></div>

  <aside class="config-drawer" transition:slide={{ axis: 'x', duration: 250 }}>
    <header class="drawer-header">
      <h2>Configuration</h2>
      <button class="btn btn-ghost btn-icon" onclick={closeDrawer} aria-label="Close drawer">
        <Icon name="close" size={20} />
      </button>
    </header>

    <div class="drawer-content">
      {#if !$isConnected}
        <div class="status-section">
          <p class="disconnected">Disconnected from SpacetimeDB</p>
        </div>
      {:else if !$currentDevice}
        <div class="status-section">
          <p class="no-device">No capture device registered</p>
          <span class="hint">Start the capture client to see device info</span>
        </div>
      {:else}
        <section class="config-section">
          <h3>Device</h3>
          <div class="device-info">
            <div class="device-name">{$currentDevice.name}</div>
            <div class="device-meta">
              <span class="platform">{$currentDevice.platform}</span>
              <span class="status-indicator" class:online={$currentDevice.status === 'online'}>
                {$currentDevice.status}
              </span>
            </div>
          </div>
        </section>

        {#if $inputsForDevice.length > 0}
          <section class="config-section">
            <h3>Audio Input</h3>
            <select
              class="select"
              value={$currentConfig?.selectedAudioInputIndex ?? -1}
              onchange={handleAudioInputChange}
            >
              <option value={-1}>System Default</option>
              {#each $inputsForDevice as input}
                <option value={input.inputIndex}>
                  {input.name}
                  {input.isDefault ? ' (Default)' : ''}
                </option>
              {/each}
            </select>
          </section>
        {/if}

        <section class="config-section">
          <h3>Silence Threshold</h3>
          <div class="slider-container">
            <input
              type="range"
              class="slider"
              min="0.001"
              max="0.1"
              step="0.001"
              bind:value={thresholdValue}
              oninput={handleThresholdChange}
              onchange={handleThresholdCommit}
            />
            <span class="slider-value">{thresholdValue.toFixed(3)}</span>
          </div>
          <p class="config-hint">
            Lower = more sensitive (picks up quieter sounds)
          </p>
        </section>

        <section class="config-section">
          <h3>Capture Controls</h3>
          <div class="control-buttons">
            <button
              class="btn control-btn"
              class:active={isPaused}
              onclick={handlePauseToggle}
            >
              <Icon name={isPaused ? 'play' : 'pause'} size={18} />
              {isPaused ? 'Resume' : 'Pause'}
            </button>

            <button
              class="btn control-btn"
              class:active={isMuted}
              onclick={handleMuteToggle}
            >
              <Icon name={isMuted ? 'volume' : 'volume-off'} size={18} />
              {isMuted ? 'Unmute' : 'Mute'}
            </button>
          </div>
          <p class="config-hint">
            {#if isPaused}
              Capture is paused - audio is not being recorded
            {:else if isMuted}
              Capture is muted - audio recorded but not transcribed
            {:else}
              Capture active - recording and transcribing
            {/if}
          </p>
        </section>

        <section class="config-section">
          <h3>Speaker Identification</h3>
          <div class="control-buttons">
            <button
              class="btn control-btn"
              class:active={speakerEmbeddingEnabled}
              onclick={handleSpeakerEmbeddingToggle}
            >
              <Icon name={speakerEmbeddingEnabled ? 'check' : 'close'} size={18} />
              {speakerEmbeddingEnabled ? 'Enabled' : 'Disabled'}
            </button>
          </div>

          {#if speakerEmbeddingEnabled}
            <div class="slider-container" style="margin-top: var(--space-md);">
              <span class="setting-label">Match threshold</span>
              <input
                type="range"
                class="slider"
                min="0.5"
                max="0.95"
                step="0.05"
                bind:value={speakerMatchThreshold}
                oninput={handleThresholdSliderChange}
                onchange={handleThresholdSliderCommit}
              />
              <span class="slider-value">{speakerMatchThreshold.toFixed(2)}</span>
            </div>

            <div class="control-buttons" style="margin-top: var(--space-sm);">
              <button
                class="btn control-btn"
                class:active={createUnknownSpeakers}
                onclick={handleCreateUnknownToggle}
              >
                Auto-create unknown speakers: {createUnknownSpeakers ? 'On' : 'Off'}
              </button>
            </div>
          {/if}

          <p class="config-hint">
            {#if speakerEmbeddingEnabled}
              Real-time speaker ID via sherpa-onnx
            {:else}
              Enable to identify speakers in real-time during capture
            {/if}
          </p>
        </section>

        {#if $currentConfig}
          <section class="config-section">
            <h3>Advanced Settings</h3>
            <div class="settings-grid">
              <div class="setting-item">
                <span class="setting-label">Min Speech</span>
                <span class="setting-value">{$currentConfig.minSpeechDurationMs}ms</span>
              </div>
              <div class="setting-item">
                <span class="setting-label">Max Speech</span>
                <span class="setting-value">{$currentConfig.maxSpeechDurationMs}ms</span>
              </div>
              <div class="setting-item">
                <span class="setting-label">Silence Duration</span>
                <span class="setting-value">{$currentConfig.silenceDurationMs}ms</span>
              </div>
              <div class="setting-item">
                <span class="setting-label">Language</span>
                <span class="setting-value">{$currentConfig.language}</span>
              </div>
            </div>
          </section>
        {/if}
      {/if}
    </div>
  </aside>
{/if}

<style>
  .drawer-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: var(--z-overlay);
  }

  .config-drawer {
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    width: var(--drawer-width);
    max-width: 90vw;
    background: var(--clr-surface-a0);
    border-right: 1px solid var(--clr-surface-a20);
    z-index: var(--z-modal);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .drawer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-md) var(--space-lg);
    border-bottom: 1px solid var(--clr-surface-a20);
  }

  .drawer-header h2 {
    font-size: var(--text-lg);
    font-weight: 600;
  }

  .drawer-content {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-lg);
  }

  .status-section {
    text-align: center;
    padding: var(--space-xl);
    color: var(--clr-surface-a50);
  }

  .disconnected {
    color: var(--clr-danger-a10);
  }

  .no-device {
    margin-bottom: var(--space-xs);
  }

  .hint {
    font-size: var(--text-xs);
    color: var(--clr-surface-a40);
  }

  .config-section {
    margin-bottom: var(--space-xl);
  }

  .config-section h3 {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--clr-surface-a50);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: var(--space-sm);
  }

  .device-info {
    padding: var(--space-md);
    background: var(--clr-surface-a10);
    border-radius: var(--radius-lg);
  }

  .device-name {
    font-weight: 500;
    margin-bottom: var(--space-xs);
  }

  .device-meta {
    display: flex;
    gap: var(--space-md);
    font-size: var(--text-sm);
    color: var(--clr-surface-a50);
  }

  .status-indicator {
    color: var(--clr-surface-a40);
  }

  .status-indicator.online {
    color: var(--clr-success-a10);
  }

  .slider-container {
    display: flex;
    align-items: center;
    gap: var(--space-md);
  }

  .slider {
    flex: 1;
  }

  .slider-value {
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--clr-primary-a0);
    min-width: 50px;
    text-align: right;
  }

  .config-hint {
    font-size: var(--text-xs);
    color: var(--clr-surface-a40);
    margin-top: var(--space-sm);
  }

  .control-buttons {
    display: flex;
    gap: var(--space-sm);
  }

  .control-btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-sm);
    padding: var(--space-md);
    background: var(--clr-surface-a10);
    border: 1px solid var(--clr-surface-a20);
    border-radius: var(--radius-lg);
    font-size: var(--text-sm);
    transition: all var(--transition-fast);
  }

  .control-btn:hover {
    background: var(--clr-surface-a20);
  }

  .control-btn.active {
    background: var(--clr-warning-a0);
    border-color: var(--clr-warning-a0);
    color: var(--clr-light);
  }

  .settings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-sm);
  }

  .setting-item {
    display: flex;
    flex-direction: column;
    padding: var(--space-sm);
    background: var(--clr-surface-a10);
    border-radius: var(--radius-md);
  }

  .setting-label {
    font-size: var(--text-xs);
    color: var(--clr-surface-a40);
  }

  .setting-value {
    font-size: var(--text-sm);
    font-family: var(--font-mono);
    color: var(--clr-light);
  }
</style>
