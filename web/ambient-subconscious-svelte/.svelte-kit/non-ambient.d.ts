
// this file is generated — do not edit it


declare module "svelte/elements" {
	export interface HTMLAttributes<T> {
		'data-sveltekit-keepfocus'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-noscroll'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-preload-code'?:
			| true
			| ''
			| 'eager'
			| 'viewport'
			| 'hover'
			| 'tap'
			| 'off'
			| undefined
			| null;
		'data-sveltekit-preload-data'?: true | '' | 'hover' | 'tap' | 'off' | undefined | null;
		'data-sveltekit-reload'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-replacestate'?: true | '' | 'off' | undefined | null;
	}
}

export {};


declare module "$app/types" {
	export interface AppTypes {
		RouteId(): "/" | "/api" | "/api/assistant-response" | "/api/avatar" | "/api/avatar/video" | "/api/avatar/video/[name]" | "/api/diarization" | "/api/entry" | "/api/entry/enrich" | "/api/entry/query" | "/api/entry/update" | "/api/frame" | "/api/frame/create" | "/api/frame/query" | "/api/frame/update" | "/api/model-state" | "/api/process-bridge" | "/api/sessions" | "/api/sessions/[id]" | "/api/swarm" | "/api/swarm/clear-logs" | "/api/transcript-stream" | "/avatar" | "/frames" | "/model-state" | "/session-editor" | "/swarm-monitor";
		RouteParams(): {
			"/api/avatar/video/[name]": { name: string };
			"/api/sessions/[id]": { id: string }
		};
		LayoutParams(): {
			"/": { name?: string; id?: string };
			"/api": { name?: string; id?: string };
			"/api/assistant-response": Record<string, never>;
			"/api/avatar": { name?: string };
			"/api/avatar/video": { name?: string };
			"/api/avatar/video/[name]": { name: string };
			"/api/diarization": Record<string, never>;
			"/api/entry": Record<string, never>;
			"/api/entry/enrich": Record<string, never>;
			"/api/entry/query": Record<string, never>;
			"/api/entry/update": Record<string, never>;
			"/api/frame": Record<string, never>;
			"/api/frame/create": Record<string, never>;
			"/api/frame/query": Record<string, never>;
			"/api/frame/update": Record<string, never>;
			"/api/model-state": Record<string, never>;
			"/api/process-bridge": Record<string, never>;
			"/api/sessions": { id?: string };
			"/api/sessions/[id]": { id: string };
			"/api/swarm": Record<string, never>;
			"/api/swarm/clear-logs": Record<string, never>;
			"/api/transcript-stream": Record<string, never>;
			"/avatar": Record<string, never>;
			"/frames": Record<string, never>;
			"/model-state": Record<string, never>;
			"/session-editor": Record<string, never>;
			"/swarm-monitor": Record<string, never>
		};
		Pathname(): "/" | "/api" | "/api/" | "/api/assistant-response" | "/api/assistant-response/" | "/api/avatar" | "/api/avatar/" | "/api/avatar/video" | "/api/avatar/video/" | `/api/avatar/video/${string}` & {} | `/api/avatar/video/${string}/` & {} | "/api/diarization" | "/api/diarization/" | "/api/entry" | "/api/entry/" | "/api/entry/enrich" | "/api/entry/enrich/" | "/api/entry/query" | "/api/entry/query/" | "/api/entry/update" | "/api/entry/update/" | "/api/frame" | "/api/frame/" | "/api/frame/create" | "/api/frame/create/" | "/api/frame/query" | "/api/frame/query/" | "/api/frame/update" | "/api/frame/update/" | "/api/model-state" | "/api/model-state/" | "/api/process-bridge" | "/api/process-bridge/" | "/api/sessions" | "/api/sessions/" | `/api/sessions/${string}` & {} | `/api/sessions/${string}/` & {} | "/api/swarm" | "/api/swarm/" | "/api/swarm/clear-logs" | "/api/swarm/clear-logs/" | "/api/transcript-stream" | "/api/transcript-stream/" | "/avatar" | "/avatar/" | "/frames" | "/frames/" | "/model-state" | "/model-state/" | "/session-editor" | "/session-editor/" | "/swarm-monitor" | "/swarm-monitor/";
		ResolvedPathname(): `${"" | `/${string}`}${ReturnType<AppTypes['Pathname']>}`;
		Asset(): string & {};
	}
}