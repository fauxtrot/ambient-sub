import type { Handle } from '@sveltejs/kit';

// SvelteKit handle function
export const handle: Handle = async ({ event, resolve }) => {
	// No ProcessManager - processes run independently
	return resolve(event);
};
