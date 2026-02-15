# Stage HTTP API v0.2

HTTP server for external control of the VTuber stage. Runs on the main thread (non-blocking poll), accessible on all network interfaces.

**Base URL:** `http://<host>:7860`

**Keyboard shortcuts (in Godot window):**
- `S` — Show debug UI panels
- `H` — Hide debug UI panels

---

## Endpoints

### `GET /health`

Health check.

**Response:**
```json
{"status": "ok", "server": "stage-vtuber", "port": 7860}
```

---

### `GET /reactions`

List all available reaction presets (from `res://reactions/*.json`).

**Response:**
```json
{"reactions": ["default", "excited", "disgust", "love", "neko", "wink", ...]}
```

---

### `GET /status`

Current reaction state.

**Response:**
```json
{
  "current_reaction": "excited",
  "transitioning_to": "",
  "reaction_active": true
}
```

---

### `POST /react`

Apply a named reaction preset.

**Request:**
```json
{"name": "excited", "instant": false}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Reaction preset name (filename without `.json`) |
| `instant` | bool | no | Skip lerp transition, apply immediately. Default `false` |

**Response:**
```json
{"applied": "excited", "instant": false}
```

---

### `POST /react/reset`

Reset all blend shapes to zero and clear all reaction state (including sequences and mouth masking).

**Response:**
```json
{"reset": true}
```

---

## Speech

### `POST /speak`

Stream audio + lip sync data. Accepts chunked delivery — send multiple requests to build up a stream.

The server buffers the first 3 chunks before starting playback to avoid stuttering. If `done` is set on an earlier chunk, playback starts immediately with whatever has been buffered.

**Request:**
```json
{
  "audio": "<base64 float32 PCM>",
  "sample_rate": 24000,
  "visemes": [
	{"visemes": {"sil": 0.8, "aa": 0.1}},
	{"visemes": {"aa": 0.6, "oh": 0.3}}
  ],
  "frame_time": 0.01,
  "id": "utterance_1",
  "callback_url": "http://192.168.1.100:8000/speak_done",
  "done": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | string | no | Base64-encoded raw float32 PCM bytes (little-endian, mono) |
| `sample_rate` | int | first chunk | Audio sample rate in Hz. Default `24000` |
| `visemes` | array | no | Array of viseme frame dicts from OVR LipSync. If omitted, realtime OVR generates them from the audio |
| `frame_time` | float | first chunk | Seconds per viseme frame. Default `0.01` (10ms) |
| `id` | string | no | Client-supplied speak ID. Auto-generated as `speak_1`, `speak_2`... if omitted. Only read from first chunk |
| `callback_url` | string | no | URL to POST when audio finishes. Only read from first chunk |
| `done` | bool | last chunk | Signal end of stream. Playback continues until buffer drains |

**Viseme frame format** (matches `tools/ovr_lipsync.py` output):
```json
{
  "visemes": {"sil": 0.8, "PP": 0.0, "aa": 0.1}
}
```

The 15 OVR viseme names: `sil`, `PP`, `FF`, `TH`, `DD`, `kk`, `CH`, `SS`, `nn`, `RR`, `aa`, `E`, `ih`, `oh`, `ou`

**Response:**
```json
{
  "speak_id": "utterance_1",
  "status": "buffering",
  "buffered_ms": 0.0,
  "playing": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `speak_id` | string | ID for this utterance |
| `status` | string | `idle`, `buffering`, `playing`, or `finished` |
| `buffered_ms` | float | Estimated unplayed audio remaining (ms) |
| `playing` | bool | Whether the stream is active |

**Chunk flow:**

```
POST /speak  {"audio": "...", "sample_rate": 24000, "visemes": [...], "id": "utt_1", "callback_url": "http://..."}
			 -> {"speak_id": "utt_1", "status": "buffering", ...}  # buffering chunk 1

POST /speak  {"audio": "...", "visemes": [...]}
			 -> {"speak_id": "utt_1", "status": "buffering", ...}  # buffering chunk 2

POST /speak  {"audio": "...", "visemes": [...]}
			 -> {"speak_id": "utt_1", "status": "playing", ...}    # chunk 3 triggers playback

POST /speak  {"audio": "...", "visemes": [...], "done": true}
			 -> {"speak_id": "utt_1", "status": "playing", ...}    # end signaled, draining

# When audio finishes, Godot POSTs to callback_url:
# -> POST http://...  {"status": "finished", "id": "utt_1"}
```

---

### `GET /speak/status`

Poll current speech playback state.

**Response:**
```json
{
  "status": "playing",
  "speak_id": "utt_1",
  "playing": true,
  "buffered_ms": 2340.5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `idle`, `buffering`, `playing`, or `finished` |
| `speak_id` | string | ID of the current/last utterance |
| `playing` | bool | Whether the audio stream is active |
| `buffered_ms` | float | Estimated unplayed audio remaining (ms) |

---

### `POST /speak/stop`

Force-stop audio playback and clear all buffers.

**Response:**
```json
{"stopped": true, "speak_id": "utt_1"}
```

---

### Speak Lifecycle Callback

When `callback_url` is provided in the first `/speak` chunk, Godot fires an async HTTP POST to that URL when audio playback finishes:

**Callback POST body:**
```json
{"status": "finished", "id": "utt_1"}
```

This is the primary mechanism for the TTS client to know when to send the next utterance without polling.

---

## Camera

### `GET /camera`

Get current camera state.

**Response:**
```json
{
  "target": [0.0, 1.05, 0.0],
  "distance": 3.0,
  "yaw": 0.0,
  "pitch": -15.0,
  "fov": 75.0,
  "transitioning": false
}
```

---

### `GET /camera/presets`

List available camera preset names.

**Response:**
```json
{"presets": ["default", "closeup", "medium", "wide", "dramatic"]}
```

---

### `POST /camera`

Set camera position. Supports named presets or direct values. Transitions are lerped by default.

**Using a preset:**
```json
{"preset": "closeup", "instant": false}
```

**Using direct values (all optional, unset fields keep current value):**
```json
{
  "target": [0.0, 1.4, 0.0],
  "distance": 1.5,
  "yaw": 25.0,
  "pitch": -5.0,
  "fov": 45.0,
  "instant": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `preset` | string | no | Named preset (`default`, `closeup`, `medium`, `wide`, `dramatic`) |
| `target` | [x,y,z] | no | Camera look-at target position |
| `distance` | float | no | Orbit distance from target |
| `yaw` | float | no | Horizontal orbit angle (degrees) |
| `pitch` | float | no | Vertical orbit angle (degrees, -89 to 89) |
| `fov` | float | no | Field of view (degrees) |
| `instant` | bool | no | Skip lerp, apply immediately. Default `false` |

**Response (preset):**
```json
{"applied_preset": "closeup", "instant": false}
```

**Response (direct):**
```json
{
  "target": [0.0, 1.4, 0.0],
  "distance": 1.5,
  "yaw": 25.0,
  "pitch": -5.0,
  "fov": 45.0,
  "instant": false
}
```

**Preset details:**

| Preset | Distance | FOV | Description |
|--------|----------|-----|-------------|
| `default` | 3.0 | 75 | Standard full-body framing |
| `closeup` | 1.0 | 50 | Face close-up, slight upward offset |
| `medium` | 2.0 | 60 | Upper body framing |
| `wide` | 5.0 | 75 | Full scene, pulled back |
| `dramatic` | 1.5 | 45 | Off-center angle, narrow FOV |

Manual mouse interaction (orbit/pan/zoom) cancels any active programmatic transition.

---

## Reaction JSON Format

Reaction presets are stored in `res://reactions/<name>.json`.

```json
{
  "name": "excited",
  "blend_shapes": {
	"eye_open": 0.5,
	"mouth_smile": 0.6,
	"brow_surprised": 0.6
  },
  "speak_mask": {
	"mouth": 0.0
  },
  "sequences": [
	{
	  "shape": "Fcl_MTH_Up",
	  "keyframes": [
		{"t": 0.0, "v": 0.0},
		{"t": 0.15, "v": 0.3},
		{"t": 0.3, "v": 0.0}
	  ],
	  "loop": true
	}
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Reaction name |
| `blend_shapes` | object | yes | Map of blend shape name to target value (-1.0 to 1.0). Only non-zero values need to be included |
| `speak_mask` | object | no | Per-category weights controlling how much the expression holds during speech |
| `sequences` | array | no | Keyframed blend shape animations that play while the reaction is active |

### Speak Mask

Controls how expression mouth shapes interact with lipsync visemes during speech.

```json
"speak_mask": {
  "mouth": 0.7
}
```

| Value | Behavior |
|-------|----------|
| `0.0` | Visemes fully override mouth shapes (default) |
| `0.5` | 50/50 blend between visemes and expression |
| `1.0` | Expression fully holds, visemes have no effect on mouth |

Mouth-category shapes: `mouth_*`, `Fcl_MTH_*`, `Vrc.v_*`, `tongue*`

Eye, brow, and cheek shapes are never masked — they always hold their expression values.

### Animation Sequences

Optional time-based keyframe animations that run alongside the reaction.

```json
"sequences": [
  {
	"shape": "Fcl_MTH_Up",
	"keyframes": [
	  {"t": 0.0, "v": 0.0},
	  {"t": 0.15, "v": 0.3},
	  {"t": 0.3, "v": 0.0}
	],
	"loop": true
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `shape` | string | Blend shape name to animate |
| `keyframes` | array | Time/value pairs. `t` = seconds, `v` = blend shape value |
| `loop` | bool | `true` to repeat, `false` for one-shot |

Keyframes interpolate linearly. Looping sequences wrap around the last keyframe's `t` value. Sequences stop when the reaction is replaced or reset.

---

## Audio Pipeline

The expected flow for the AI pipeline:

```
LLM -> TTS (float32 PCM) -> Handler (OVR lipsync) -> POST /speak -> Godot
                                                              |
                                         callback POST <------+  (when audio finishes)
                                              |
                                         send next utterance
```

1. TTS produces float32 mono PCM at its native sample rate (24kHz, 44.1kHz, etc.)
2. Handler runs `tools/ovr_lipsync.py` on 10ms chunks to generate viseme weights (or omit visemes to use Godot's realtime OVR)
3. Handler sends audio chunks + viseme frames to `/speak` with a `callback_url`
4. Godot plays audio via `AudioStreamGenerator` and drives `Vrc.v_*` blend shapes from viseme data
5. When playback finishes, Godot POSTs to `callback_url` with `{"status": "finished", "id": "..."}`
6. Handler receives callback and sends next utterance

## Notes

- Server listens on `0.0.0.0:7860` — accessible from LAN. Ensure firewall allows inbound on port 7860
- All responses include `Access-Control-Allow-Origin: *`
- Connection is closed after each response (`Connection: close`)
- Content-Length aware body parsing supports large base64 payloads across multiple TCP reads
- Pre-play buffer is 3 chunks (configurable via `SPEAK_BUFFER_CHUNKS` constant)
- If no visemes are provided in `/speak`, Godot uses realtime OVR LipSync to generate them from the audio
