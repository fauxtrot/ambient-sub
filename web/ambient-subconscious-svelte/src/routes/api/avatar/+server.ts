/**
 * Avatar state API endpoint.
 *
 * GET  /api/avatar - Get current expression
 * POST /api/avatar - Set expression (body: { expression: "happy" })
 *
 * Example usage:
 *   curl http://localhost:5173/api/avatar
 *   curl -X POST -H "Content-Type: application/json" -d '{"expression":"happy"}' http://localhost:5173/api/avatar
 */
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import {
	getExpression,
	setExpression,
	EXPRESSIONS,
	EXPRESSION_VIDEOS,
	type Expression
} from '$lib/stores/avatar';

export const GET: RequestHandler = async () => {
	const expression = getExpression();
	return json({
		expression,
		video: EXPRESSION_VIDEOS[expression],
		available: EXPRESSIONS
	});
};

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = await request.json();
		const { expression } = body;

		if (!expression || !EXPRESSIONS.includes(expression as Expression)) {
			return json(
				{
					error: 'Invalid expression',
					available: EXPRESSIONS
				},
				{ status: 400 }
			);
		}

		setExpression(expression as Expression);
		console.log(`[Avatar] Expression changed to: ${expression}`);

		return json({
			expression,
			video: EXPRESSION_VIDEOS[expression as Expression]
		});
	} catch {
		return json({ error: 'Invalid JSON body' }, { status: 400 });
	}
};
