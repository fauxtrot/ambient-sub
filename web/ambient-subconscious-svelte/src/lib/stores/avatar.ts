/**
 * Avatar state store - tracks current expression for the video avatar.
 *
 * Available expressions: neutral, happy, sad, angry, confused
 */

export type Expression = 'neutral' | 'happy' | 'sad' | 'angry' | 'confused';

export const EXPRESSIONS: Expression[] = ['neutral', 'happy', 'sad', 'angry', 'confused'];

export const EXPRESSION_VIDEOS: Record<Expression, string> = {
	neutral: 'General.mp4',
	happy: 'Happy.mp4',
	sad: 'sad.mp4',
	angry: 'angry.mp4',
	confused: 'confused.mp4'
};

// Server-side state (simple in-memory for single server)
let currentExpression: Expression = 'neutral';

export function getExpression(): Expression {
	return currentExpression;
}

export function setExpression(expression: Expression): void {
	if (EXPRESSIONS.includes(expression)) {
		currentExpression = expression;
	}
}
