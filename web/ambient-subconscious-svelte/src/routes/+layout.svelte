<script lang="ts">
  import { onMount } from 'svelte';
  import { connect, drawerOpen, isConnected, isConnecting, connectionError } from '$lib/stores/spacetime';
  import StatusBadge from '$lib/components/StatusBadge.svelte';
  import ConfigDrawer from '$lib/components/ConfigDrawer.svelte';
  import Icon from '$lib/components/Icon.svelte';
  import '$lib/styles/global.css';

  let { children } = $props();

  onMount(() => {
    connect();
  });

  function toggleDrawer() {
    drawerOpen.update((v) => !v);
  }
</script>

<div class="app">
  <header class="app-header">
    <div class="header-left">
      <button class="btn btn-ghost btn-icon" onclick={toggleDrawer} aria-label="Open menu">
        <Icon name="menu" size={20} />
      </button>
      <h1 class="app-title">Ambient Listener</h1>
      <nav class="header-nav">
        <a href="/">Dashboard</a>
        <a href="/frames">Frames</a>
        <a href="/model-state">Model State</a>
        <a href="/session-editor">Sessions</a>
        <a href="/swarm-monitor">Swarm Monitor</a>
        <a href="/avatar">Avatar</a>
      </nav>
    </div>

    <div class="header-right">
      {#if $connectionError}
        <span class="connection-error" title={$connectionError}>
          <Icon name="alert" size={16} />
          Error
        </span>
      {/if}
      <StatusBadge />
    </div>
  </header>

  <main class="app-main">
    {@render children()}
  </main>

  <ConfigDrawer />
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: var(--clr-surface-a0);
    overflow: hidden;
  }

  .app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 var(--space-md);
    height: var(--header-height);
    background: var(--clr-surface-a0);
    border-bottom: 1px solid var(--clr-surface-a20);
    position: sticky;
    top: 0;
    z-index: var(--z-sticky);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
  }

  .header-nav {
    display: flex;
    gap: var(--space-md);
    margin-left: var(--space-lg);
  }

  .header-nav a {
    font-size: var(--text-sm);
    color: var(--clr-text-a30, #888);
    text-decoration: none;
    transition: color 0.15s;
  }

  .header-nav a:hover {
    color: var(--clr-text-a0, #e0e0e0);
  }

  .app-title {
    font-size: var(--text-lg);
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--clr-primary-a0), var(--clr-primary-a20));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: var(--space-md);
  }

  .connection-error {
    display: inline-flex;
    align-items: center;
    gap: var(--space-xs);
    padding: var(--space-xs) var(--space-sm);
    font-size: var(--text-xs);
    color: var(--clr-danger-a10);
    background: var(--clr-danger-a0);
    background: rgba(156, 33, 33, 0.2);
    border-radius: var(--radius-full);
    cursor: help;
  }

  .app-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
</style>
