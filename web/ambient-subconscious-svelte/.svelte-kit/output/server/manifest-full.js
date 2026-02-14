export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.DDK1QHZq.js",app:"_app/immutable/entry/app.GhpLuVM4.js",imports:["_app/immutable/entry/start.DDK1QHZq.js","_app/immutable/chunks/Dg_j1qZU.js","_app/immutable/chunks/ocTC9fog.js","_app/immutable/chunks/gUtJ7mXj.js","_app/immutable/entry/app.GhpLuVM4.js","_app/immutable/chunks/ocTC9fog.js","_app/immutable/chunks/Dcw-4Js5.js","_app/immutable/chunks/gUtJ7mXj.js","_app/immutable/chunks/CeYpcBse.js","_app/immutable/chunks/Bb87sCMO.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
