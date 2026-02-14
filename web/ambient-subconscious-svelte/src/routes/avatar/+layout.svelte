<script lang="ts">
	// Minimal layout for avatar page - no header, full screen
	let { children } = $props();
</script>

<div class="avatar-layout">
	{@render children()}
</div>

<style>
	.avatar-layout {
		width: 100vw;
		height: 100vh;
		margin: 0;
		padding: 0;
		overflow: hidden;
		background: #000;
	}

	:global(body) {
		margin: 0;
		padding: 0;
	}
</style>
