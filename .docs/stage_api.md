# Stage HTTP API v0.01

HTTP server for external control of the VTuber stage. Runs on the main thread (non-blocking poll), localhost only.

**Base URL:** `http://127.0.0.1:7860`

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
{"reactions": ["happy", "sad", "angry", "surprise"]}
```

---

### `GET /status`

Current reaction state.

**Response:**
```json
{
  "current_reaction": "happy",
  "transitioning_to": "",
  "reaction_active": true
}
```

---

### `POST /react`

Apply a named reaction preset.

**Request:**
```json
{"name": "happy", "instant": false}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Reaction preset name (filename without `.json`) |
| `instant` | bool | no | Skip lerp transition, apply immediately. Default `false` |

**Response:**
```json
{"applied": "happy", "instant": false}
```

---

### `POST /react/reset`

Reset all blend shapes to default (zero).

**Response:**
```json
{"reset": true}
```

---

### `POST /speak`

Stream audio + lip sync data. Accepts chunked delivery -- send multiple requests to build up a stream.

The server buffers the first 3 chunks before starting playback to avoid stuttering. If `done` is set on an earlier chunk, playback starts immediately with whatever has been buffered.

**Request:**
```json
{
  "audio": "<base64 float32 PCM>",
  "sample_rate": 24000,
  "visemes": [
    {"visemes": {"sil": 0.8, "aa": 0.1}, "laughter": 0.0},
    {"visemes": {"aa": 0.6, "oh": 0.3}, "laughter": 0.0}
  ],
  "frame_time": 0.01,
  "done": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | string | no | Base64-encoded raw float32 PCM bytes (little-endian, mono) |
| `sample_rate` | int | first chunk | Audio sample rate in Hz. Default `24000`. Only needed on first chunk |
| `visemes` | array | no | Array of viseme frame dicts from OVR LipSync |
| `frame_time` | float | first chunk | Seconds per viseme frame. Default `0.01` (10ms). Only needed on first chunk |
| `done` | bool | last chunk | Signal end of stream. Playback continues until buffer drains |

**Response:**
```json
{"buffered_ms": 4187.3, "playing": true}
```

| Field | Type | Description |
|-------|------|-------------|
| `buffered_ms` | float | Estimated unplayed audio remaining (ms) |
| `playing` | bool | Whether the stream is active |

**Chunk flow:**

```
POST /speak  {"audio": "...", "sample_rate": 24000}
             -> {"buffered_ms": 0.0, "playing": false}     # buffering chunk 1

POST /speak  {"audio": "..."}
             -> {"buffered_ms": 0.0, "playing": false}     # buffering chunk 2

POST /speak  {"audio": "...", "done": true}
             -> {"buffered_ms": 4187.3, "playing": true}   # chunk 3 triggers playback
```

Single-chunk shortcut (done on first chunk skips buffering):
```
POST /speak  {"audio": "...", "sample_rate": 24000, "done": true}
             -> {"buffered_ms": 500.0, "playing": true}
```

---

### `POST /speak/stop`

Force-stop audio playback and clear all buffers.

**Response:**
```json
{"stopped": true}
```

---

### `GET /mirror`

Capture a screenshot of the current viewport and return it as a JPEG image.

**Response:** Binary JPEG image (`Content-Type: image/jpeg`), quality 85%.

```bash
curl http://127.0.0.1:7860/mirror --output screenshot.jpg
```

---

## Audio Pipeline

The expected flow for the AI pipeline:

```
LLM -> TTS (float32 PCM) -> POST /speak -> Godot
```

1. TTS produces float32 mono PCM at its native sample rate (24kHz)
2. Audio chunks are base64-encoded and sent to `/speak`
3. Godot plays audio via `AudioStreamGenerator` and drives blend shapes from viseme data (if provided)

## Notes

- Server is localhost-only (`127.0.0.1`), not exposed to network
- All responses include `Access-Control-Allow-Origin: *`
- Connection is closed after each response (`Connection: close`)
- Pre-play buffer is 3 chunks (configurable via `SPEAK_BUFFER_CHUNKS` constant)
- Visemes are optional -- audio-only `/speak` calls are valid
