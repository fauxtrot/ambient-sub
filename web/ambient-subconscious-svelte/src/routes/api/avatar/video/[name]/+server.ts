/**
 * Serve avatar video files from the .video directory.
 *
 * GET /api/avatar/video/General.mp4
 * GET /api/avatar/video/Happy.mp4
 * etc.
 */
import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { readFile, stat } from 'fs/promises';
import { join, resolve } from 'path';

// Videos are stored in project root's .video folder
// Use VIDEOS_DIR env var if set, otherwise resolve relative to cwd
const VIDEOS_DIR = process.env.VIDEOS_DIR || resolve(process.cwd(), '..', '..', '.video');

// Allowed video files (security: prevent directory traversal)
const ALLOWED_VIDEOS = new Set([
	'General.mp4',
	'Happy.mp4',
	'sad.mp4',
	'angry.mp4',
	'confused.mp4'
]);

export const GET: RequestHandler = async ({ params, request }) => {
	const { name } = params;

	if (!name || !ALLOWED_VIDEOS.has(name)) {
		throw error(404, 'Video not found');
	}

	const videoPath = join(VIDEOS_DIR, name);

	try {
		const stats = await stat(videoPath);
		const fileSize = stats.size;

		// Handle range requests for video seeking
		const range = request.headers.get('range');

		if (range) {
			const parts = range.replace(/bytes=/, '').split('-');
			const start = parseInt(parts[0], 10);
			const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
			const chunkSize = end - start + 1;

			// Read the specific chunk
			const { createReadStream } = await import('fs');
			const stream = createReadStream(videoPath, { start, end });

			// Convert Node stream to Web ReadableStream
			const webStream = new ReadableStream({
				start(controller) {
					stream.on('data', (chunk) => controller.enqueue(chunk));
					stream.on('end', () => controller.close());
					stream.on('error', (err) => controller.error(err));
				},
				cancel() {
					stream.destroy();
				}
			});

			return new Response(webStream, {
				status: 206,
				headers: {
					'Content-Range': `bytes ${start}-${end}/${fileSize}`,
					'Accept-Ranges': 'bytes',
					'Content-Length': chunkSize.toString(),
					'Content-Type': 'video/mp4'
				}
			});
		}

		// No range request - return full file
		const videoData = await readFile(videoPath);

		return new Response(videoData, {
			headers: {
				'Content-Type': 'video/mp4',
				'Content-Length': fileSize.toString(),
				'Accept-Ranges': 'bytes',
				'Cache-Control': 'public, max-age=3600'
			}
		});
	} catch (err) {
		console.error(`[Avatar Video] Error serving ${name}:`, err);
		throw error(404, 'Video not found');
	}
};
