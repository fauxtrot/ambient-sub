<script lang="ts">
  import { isConnected, isConnecting, currentDevice, currentSession, currentConfig } from '$lib/stores/spacetime';

  // Derive status from stores
  let statusText = $derived.by(() => {
    if ($isConnecting) return 'Connecting...';
    if (!$isConnected) return 'Disconnected';
    if (!$currentDevice) return 'No Device';
    if ($currentConfig?.isPaused) return 'Paused';
    if ($currentConfig?.isMuted) return 'Muted';
    if ($currentSession) return 'Recording';
    return $currentDevice.status === 'online' ? 'Online' : 'Offline';
  });

  let statusClass = $derived.by(() => {
    if ($isConnecting) return 'connecting';
    if (!$isConnected) return 'disconnected';
    if ($currentConfig?.isPaused) return 'paused';
    if ($currentConfig?.isMuted) return 'muted';
    if ($currentSession) return 'recording';
    if ($currentDevice?.status === 'online') return 'online';
    return 'offline';
  });

  let isRecording = $derived($currentSession !== null && !$currentConfig?.isPaused && !$currentConfig?.isMuted);
</script>

<div class="status-badge {statusClass}">
  <span class="indicator" class:pulse={isRecording}></span>
  <span class="text">{statusText}</span>
</div>

<style>
  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-xs) var(--space-md);
    border-radius: var(--radius-full);
    font-size: var(--text-sm);
    font-weight: 500;
    background: var(--clr-surface-a10);
  }

  .indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--clr-surface-a40);
  }

  .status-badge.connecting .indicator {
    background: var(--clr-warning-a10);
  }

  .status-badge.disconnected .indicator {
    background: var(--clr-danger-a10);
  }

  .status-badge.recording .indicator {
    background: var(--clr-primary-a0);
  }

  .status-badge.online .indicator {
    background: var(--clr-success-a10);
  }

  .status-badge.paused .indicator {
    background: var(--clr-warning-a10);
  }

  .status-badge.muted .indicator {
    background: var(--clr-surface-a40);
  }

  .pulse {
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.6;
      transform: scale(1.2);
    }
  }

  .text {
    color: var(--clr-surface-a50);
  }
</style>
