import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import fs from 'fs';
import path from 'path';

// Audio storage base path - should match AUDIO_STORAGE_PATH from .env
const AUDIO_BASE_PATH = 'D:/ai-projects/ambient-listener/audio';

export default defineConfig({
	plugins: [
		sveltekit(),
		// Custom plugin to serve audio files
		{
			name: 'serve-audio-files',
			configureServer(server) {
				server.middlewares.use('/audio', (req, res, next) => {
					const filePath = path.join(AUDIO_BASE_PATH, decodeURIComponent(req.url || ''));
					if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
						res.setHeader('Content-Type', 'audio/wav');
						fs.createReadStream(filePath).pipe(res);
					} else {
						next();
					}
				});
			}
		}
	],
	server: {
		port: 5174,
		strictPort: false,
		fs: {
			allow: ['..', AUDIO_BASE_PATH]
		}
	}
});
