import { w as writable, d as derived } from "./index.js";
import { t, schema, table, reducers, reducerSchema, procedures, convertToAccessorMap } from "spacetimedb";
import { c as attr } from "./index2.js";
const CreateDiarizationSegmentReducer = {
  entryId: t.u32(),
  startMs: t.u32(),
  endMs: t.u32(),
  pyannoteLabel: t.string(),
  matchedSpeaker: t.option(t.string()),
  confidence: t.option(t.f32()),
  transcriptSlice: t.option(t.string())
};
const CreateEntryReducer = {
  sessionId: t.u32(),
  entryId: t.string(),
  durationMs: t.u32(),
  transcript: t.option(t.string()),
  confidence: t.option(t.f32()),
  audioClipPath: t.option(t.string()),
  recordingStartMs: t.u32(),
  recordingEndMs: t.u32()
};
const CreateOrUpdateSpeakerReducer = {
  name: t.string(),
  embedding: t.option(t.byteArray())
};
const DeleteEntryReducer = {
  entryId: t.u32()
};
const DeleteSpeakerReducer = {
  name: t.string()
};
const DeviceHeartbeatReducer = {
  deviceIdentity: t.string()
};
const InitCaptureConfigReducer = {
  deviceId: t.u32()
};
const MarkEntryReviewedReducer = {
  entryId: t.u32(),
  reviewed: t.bool()
};
const RegisterDeviceReducer = {
  deviceIdentity: t.string(),
  name: t.string(),
  platform: t.string()
};
const ReportAudioInputsReducer = {
  deviceId: t.u32(),
  inputNames: t.array(t.string())
};
const SetAudioInputReducer = {
  deviceId: t.u32(),
  inputIndex: t.i32()
};
const SetCaptureMutedReducer = {
  deviceId: t.u32(),
  isMuted: t.bool()
};
const SetCapturePausedReducer = {
  deviceId: t.u32(),
  isPaused: t.bool()
};
const SetSilenceThresholdReducer = {
  deviceId: t.u32(),
  threshold: t.f32()
};
const StartSessionReducer = {
  deviceId: t.u32(),
  date: t.string(),
  mode: t.string(),
  audioPath: t.string()
};
const UpdateDeviceStatusReducer = {
  deviceIdentity: t.string(),
  status: t.string()
};
const UpdateDiarizationSegmentSpeakerReducer = {
  segmentId: t.u32(),
  matchedSpeaker: t.string(),
  confidence: t.f32()
};
const UpdateEntryReducer = {
  entryId: t.u32(),
  transcript: t.option(t.string()),
  speakerName: t.option(t.string()),
  sentiment: t.option(t.string()),
  notes: t.option(t.string())
};
const UpdateEntryNotesReducer = {
  entryId: t.u32(),
  notes: t.string()
};
const UpdateEntrySentimentReducer = {
  entryId: t.u32(),
  sentiment: t.string()
};
const UpdateEntrySpeakerReducer = {
  entryId: t.u32(),
  speakerName: t.string()
};
const UpdateEntryTranscriptReducer = {
  entryId: t.u32(),
  transcript: t.string()
};
const UpdateSessionDurationReducer = {
  sessionId: t.u32(),
  durationMs: t.u32()
};
const UpdateSessionStatusReducer = {
  sessionId: t.u32(),
  status: t.string()
};
const UpdateSpeakerThresholdReducer = {
  name: t.string(),
  threshold: t.f32()
};
const UpdateTranscriptionConfigReducer = {
  deviceId: t.u32(),
  minConfidence: t.f32(),
  language: t.string()
};
const UpdateVadConfigReducer = {
  deviceId: t.u32(),
  silenceThreshold: t.f32(),
  minSpeechDurationMs: t.u32(),
  maxSpeechDurationMs: t.u32(),
  silenceDurationMs: t.u32()
};
const AudioInputRow = t.row({
  id: t.u32().primaryKey().name("Id"),
  deviceId: t.u32().name("DeviceId"),
  inputIndex: t.i32().name("InputIndex"),
  name: t.string().name("Name"),
  isDefault: t.bool().name("IsDefault"),
  updatedAt: t.timestamp().name("UpdatedAt")
});
const AudioSessionRow = t.row({
  id: t.u32().primaryKey().name("Id"),
  deviceId: t.u32().name("DeviceId"),
  date: t.string().name("Date"),
  mode: t.string().name("Mode"),
  status: t.string().name("Status"),
  audioPath: t.string().name("AudioPath"),
  startedAt: t.timestamp().name("StartedAt"),
  endedAt: t.option(t.timestamp()).name("EndedAt"),
  durationMs: t.u32().name("DurationMs")
});
const CaptureConfigRow = t.row({
  deviceId: t.u32().primaryKey().name("DeviceId"),
  selectedAudioInputIndex: t.i32().name("SelectedAudioInputIndex"),
  silenceThreshold: t.f32().name("SilenceThreshold"),
  minSpeechDurationMs: t.u32().name("MinSpeechDurationMs"),
  maxSpeechDurationMs: t.u32().name("MaxSpeechDurationMs"),
  silenceDurationMs: t.u32().name("SilenceDurationMs"),
  minConfidence: t.f32().name("MinConfidence"),
  language: t.string().name("Language"),
  isPaused: t.bool().name("IsPaused"),
  isMuted: t.bool().name("IsMuted"),
  updatedAt: t.timestamp().name("UpdatedAt")
});
const CaptureDeviceRow = t.row({
  id: t.u32().primaryKey().name("Id"),
  deviceIdentity: t.string().name("DeviceIdentity"),
  name: t.string().name("Name"),
  platform: t.string().name("Platform"),
  status: t.string().name("Status"),
  lastHeartbeat: t.timestamp().name("LastHeartbeat"),
  createdAt: t.timestamp().name("CreatedAt")
});
const DiarizationSegmentRow = t.row({
  id: t.u32().primaryKey().name("Id"),
  entryId: t.u32().name("EntryId"),
  startMs: t.u32().name("StartMs"),
  endMs: t.u32().name("EndMs"),
  pyannoteLabel: t.string().name("PyannoteLabel"),
  matchedSpeaker: t.option(t.string()).name("MatchedSpeaker"),
  confidence: t.option(t.f32()).name("Confidence"),
  transcriptSlice: t.option(t.string()).name("TranscriptSlice")
});
const SpeakerRow = t.row({
  id: t.u32().primaryKey().name("Id"),
  name: t.string().name("Name"),
  similarityThreshold: t.f32().name("SimilarityThreshold"),
  embedding: t.option(t.byteArray()).name("Embedding"),
  sampleCount: t.u32().name("SampleCount"),
  createdAt: t.timestamp().name("CreatedAt"),
  updatedAt: t.timestamp().name("UpdatedAt")
});
const TranscriptEntryRow = t.row({
  id: t.u32().primaryKey().name("Id"),
  sessionId: t.u32().name("SessionId"),
  entryId: t.string().name("EntryId"),
  timestamp: t.timestamp().name("Timestamp"),
  durationMs: t.u32().name("DurationMs"),
  transcript: t.option(t.string()).name("Transcript"),
  confidence: t.option(t.f32()).name("Confidence"),
  speaker: t.option(t.string()).name("Speaker"),
  sentiment: t.option(t.string()).name("Sentiment"),
  intent: t.option(t.string()).name("Intent"),
  audioClipPath: t.option(t.string()).name("AudioClipPath"),
  recordingStartMs: t.u32().name("RecordingStartMs"),
  recordingEndMs: t.u32().name("RecordingEndMs"),
  reviewed: t.bool().name("Reviewed"),
  notes: t.option(t.string()).name("Notes"),
  createdAt: t.timestamp().name("CreatedAt"),
  updatedAt: t.timestamp().name("UpdatedAt")
});
t.object("AudioInput", {
  id: t.u32(),
  deviceId: t.u32(),
  inputIndex: t.i32(),
  name: t.string(),
  isDefault: t.bool(),
  updatedAt: t.timestamp()
});
t.object("AudioSession", {
  id: t.u32(),
  deviceId: t.u32(),
  date: t.string(),
  mode: t.string(),
  status: t.string(),
  audioPath: t.string(),
  startedAt: t.timestamp(),
  endedAt: t.option(t.timestamp()),
  durationMs: t.u32()
});
t.object("CaptureConfig", {
  deviceId: t.u32(),
  selectedAudioInputIndex: t.i32(),
  silenceThreshold: t.f32(),
  minSpeechDurationMs: t.u32(),
  maxSpeechDurationMs: t.u32(),
  silenceDurationMs: t.u32(),
  minConfidence: t.f32(),
  language: t.string(),
  isPaused: t.bool(),
  isMuted: t.bool(),
  updatedAt: t.timestamp()
});
t.object("CaptureDevice", {
  id: t.u32(),
  deviceIdentity: t.string(),
  name: t.string(),
  platform: t.string(),
  status: t.string(),
  lastHeartbeat: t.timestamp(),
  createdAt: t.timestamp()
});
t.object("DiarizationSegment", {
  id: t.u32(),
  entryId: t.u32(),
  startMs: t.u32(),
  endMs: t.u32(),
  pyannoteLabel: t.string(),
  matchedSpeaker: t.option(t.string()),
  confidence: t.option(t.f32()),
  transcriptSlice: t.option(t.string())
});
t.object("Speaker", {
  id: t.u32(),
  name: t.string(),
  similarityThreshold: t.f32(),
  embedding: t.option(t.byteArray()),
  sampleCount: t.u32(),
  createdAt: t.timestamp(),
  updatedAt: t.timestamp()
});
t.object("TranscriptEntry", {
  id: t.u32(),
  sessionId: t.u32(),
  entryId: t.string(),
  timestamp: t.timestamp(),
  durationMs: t.u32(),
  transcript: t.option(t.string()),
  confidence: t.option(t.f32()),
  speaker: t.option(t.string()),
  sentiment: t.option(t.string()),
  intent: t.option(t.string()),
  audioClipPath: t.option(t.string()),
  recordingStartMs: t.u32(),
  recordingEndMs: t.u32(),
  reviewed: t.bool(),
  notes: t.option(t.string()),
  createdAt: t.timestamp(),
  updatedAt: t.timestamp()
});
const tablesSchema = schema(
  table({
    name: "audio_input",
    indexes: [
      { name: "Id", algorithm: "btree", columns: [
        "id"
      ] }
    ],
    constraints: [
      { name: "audio_input_Id_key", constraint: "unique", columns: ["id"] }
    ]
  }, AudioInputRow),
  table({
    name: "audio_session",
    indexes: [
      { name: "Id", algorithm: "btree", columns: [
        "id"
      ] }
    ],
    constraints: [
      { name: "audio_session_Id_key", constraint: "unique", columns: ["id"] }
    ]
  }, AudioSessionRow),
  table({
    name: "capture_config",
    indexes: [
      { name: "DeviceId", algorithm: "btree", columns: [
        "deviceId"
      ] }
    ],
    constraints: [
      { name: "capture_config_DeviceId_key", constraint: "unique", columns: ["deviceId"] }
    ]
  }, CaptureConfigRow),
  table({
    name: "capture_device",
    indexes: [
      { name: "DeviceIdentity", algorithm: "btree", columns: [
        "deviceIdentity"
      ] },
      { name: "Id", algorithm: "btree", columns: [
        "id"
      ] }
    ],
    constraints: [
      { name: "capture_device_DeviceIdentity_key", constraint: "unique", columns: ["deviceIdentity"] },
      { name: "capture_device_Id_key", constraint: "unique", columns: ["id"] }
    ]
  }, CaptureDeviceRow),
  table({
    name: "diarization_segment",
    indexes: [
      { name: "Id", algorithm: "btree", columns: [
        "id"
      ] }
    ],
    constraints: [
      { name: "diarization_segment_Id_key", constraint: "unique", columns: ["id"] }
    ]
  }, DiarizationSegmentRow),
  table({
    name: "speaker",
    indexes: [
      { name: "Id", algorithm: "btree", columns: [
        "id"
      ] },
      { name: "Name", algorithm: "btree", columns: [
        "name"
      ] }
    ],
    constraints: [
      { name: "speaker_Id_key", constraint: "unique", columns: ["id"] },
      { name: "speaker_Name_key", constraint: "unique", columns: ["name"] }
    ]
  }, SpeakerRow),
  table({
    name: "transcript_entry",
    indexes: [
      { name: "Id", algorithm: "btree", columns: [
        "id"
      ] }
    ],
    constraints: [
      { name: "transcript_entry_Id_key", constraint: "unique", columns: ["id"] }
    ]
  }, TranscriptEntryRow)
);
const reducersSchema = reducers(
  reducerSchema("CreateDiarizationSegment", CreateDiarizationSegmentReducer),
  reducerSchema("CreateEntry", CreateEntryReducer),
  reducerSchema("CreateOrUpdateSpeaker", CreateOrUpdateSpeakerReducer),
  reducerSchema("DeleteEntry", DeleteEntryReducer),
  reducerSchema("DeleteSpeaker", DeleteSpeakerReducer),
  reducerSchema("DeviceHeartbeat", DeviceHeartbeatReducer),
  reducerSchema("InitCaptureConfig", InitCaptureConfigReducer),
  reducerSchema("MarkEntryReviewed", MarkEntryReviewedReducer),
  reducerSchema("RegisterDevice", RegisterDeviceReducer),
  reducerSchema("ReportAudioInputs", ReportAudioInputsReducer),
  reducerSchema("SetAudioInput", SetAudioInputReducer),
  reducerSchema("SetCaptureMuted", SetCaptureMutedReducer),
  reducerSchema("SetCapturePaused", SetCapturePausedReducer),
  reducerSchema("SetSilenceThreshold", SetSilenceThresholdReducer),
  reducerSchema("StartSession", StartSessionReducer),
  reducerSchema("UpdateDeviceStatus", UpdateDeviceStatusReducer),
  reducerSchema("UpdateDiarizationSegmentSpeaker", UpdateDiarizationSegmentSpeakerReducer),
  reducerSchema("UpdateEntry", UpdateEntryReducer),
  reducerSchema("UpdateEntryNotes", UpdateEntryNotesReducer),
  reducerSchema("UpdateEntrySentiment", UpdateEntrySentimentReducer),
  reducerSchema("UpdateEntrySpeaker", UpdateEntrySpeakerReducer),
  reducerSchema("UpdateEntryTranscript", UpdateEntryTranscriptReducer),
  reducerSchema("UpdateSessionDuration", UpdateSessionDurationReducer),
  reducerSchema("UpdateSessionStatus", UpdateSessionStatusReducer),
  reducerSchema("UpdateSpeakerThreshold", UpdateSpeakerThresholdReducer),
  reducerSchema("UpdateTranscriptionConfig", UpdateTranscriptionConfigReducer),
  reducerSchema("UpdateVadConfig", UpdateVadConfigReducer)
);
const proceduresSchema = procedures();
({
  tables: tablesSchema.schemaType.tables,
  reducers: reducersSchema.reducersType.reducers,
  ...proceduresSchema
});
convertToAccessorMap(tablesSchema.schemaType.tables);
convertToAccessorMap(reducersSchema.reducersType.reducers);
const isConnected = writable(false);
const isConnecting = writable(false);
const connectionError = writable(null);
const entries = writable([]);
const sessions = writable([]);
const audioInputs = writable([]);
const devices = writable([]);
const speakers = writable([]);
const configs = writable([]);
const selectedDate = writable((/* @__PURE__ */ new Date()).toISOString().split("T")[0]);
const selectedEntryId = writable(null);
const drawerOpen = writable(false);
function timestampToDate(timestamp) {
  if (typeof timestamp.toDate === "function") {
    return timestamp.toDate();
  }
  return new Date(Number(timestamp));
}
const selectedEntry = derived(
  [entries, selectedEntryId],
  ([$entries, $selectedEntryId]) => {
    if ($selectedEntryId === null) return null;
    return $entries.find((e) => e.id === $selectedEntryId) ?? null;
  }
);
const entriesForDate = derived(
  [entries, selectedDate],
  ([$entries, $selectedDate]) => {
    return $entries.filter((e) => {
      const entryDate = timestampToDate(e.timestamp).toISOString().split("T")[0];
      return entryDate === $selectedDate;
    }).sort((a, b) => timestampToDate(b.timestamp).getTime() - timestampToDate(a.timestamp).getTime());
  }
);
const currentDevice = derived(devices, ($devices) => {
  return $devices.length > 0 ? $devices[0] : null;
});
const currentConfig = derived(
  [configs, currentDevice],
  ([$configs, $currentDevice]) => {
    if (!$currentDevice) return null;
    return $configs.find((c) => c.deviceId === $currentDevice.id) ?? null;
  }
);
const currentSession = derived(
  [sessions, currentDevice],
  ([$sessions, $currentDevice]) => {
    if (!$currentDevice) return null;
    return $sessions.find((s) => s.deviceId === $currentDevice.id && s.status === "recording") ?? null;
  }
);
const inputsForDevice = derived(
  [audioInputs, currentDevice],
  ([$audioInputs, $currentDevice]) => {
    if (!$currentDevice) return [];
    return $audioInputs.filter((i) => i.deviceId === $currentDevice.id);
  }
);
const availableDates = derived(entries, ($entries) => {
  const dates = /* @__PURE__ */ new Set();
  $entries.forEach((e) => {
    const date = timestampToDate(e.timestamp).toISOString().split("T")[0];
    dates.add(date);
  });
  return Array.from(dates).sort().reverse();
});
function updateEntry(entryId, updates) {
  {
    console.error("updateEntry: no connection");
    return;
  }
}
function setAudioInput(deviceId, inputIndex) {
  return;
}
function Icon($$renderer, $$props) {
  let { name, size = 20 } = $$props;
  $$renderer.push(`<svg${attr("width", size)}${attr("height", size)} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">`);
  if (name === "menu") {
    $$renderer.push("<!--[-->");
    $$renderer.push(`<line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line>`);
  } else {
    $$renderer.push("<!--[!-->");
    if (name === "close") {
      $$renderer.push("<!--[-->");
      $$renderer.push(`<line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line>`);
    } else {
      $$renderer.push("<!--[!-->");
      if (name === "play") {
        $$renderer.push("<!--[-->");
        $$renderer.push(`<polygon points="5 3 19 12 5 21 5 3" fill="currentColor"></polygon>`);
      } else {
        $$renderer.push("<!--[!-->");
        if (name === "pause") {
          $$renderer.push("<!--[-->");
          $$renderer.push(`<rect x="6" y="4" width="4" height="16" fill="currentColor"></rect><rect x="14" y="4" width="4" height="16" fill="currentColor"></rect>`);
        } else {
          $$renderer.push("<!--[!-->");
          if (name === "check") {
            $$renderer.push("<!--[-->");
            $$renderer.push(`<polyline points="20 6 9 17 4 12"></polyline>`);
          } else {
            $$renderer.push("<!--[!-->");
            if (name === "trash") {
              $$renderer.push("<!--[-->");
              $$renderer.push(`<polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>`);
            } else {
              $$renderer.push("<!--[!-->");
              if (name === "chevron-left") {
                $$renderer.push("<!--[-->");
                $$renderer.push(`<polyline points="15 18 9 12 15 6"></polyline>`);
              } else {
                $$renderer.push("<!--[!-->");
                if (name === "chevron-right") {
                  $$renderer.push("<!--[-->");
                  $$renderer.push(`<polyline points="9 18 15 12 9 6"></polyline>`);
                } else {
                  $$renderer.push("<!--[!-->");
                  if (name === "mic") {
                    $$renderer.push("<!--[-->");
                    $$renderer.push(`<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line>`);
                  } else {
                    $$renderer.push("<!--[!-->");
                    if (name === "mic-off") {
                      $$renderer.push("<!--[-->");
                      $$renderer.push(`<line x1="1" y1="1" x2="23" y2="23"></line><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"></path><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line>`);
                    } else {
                      $$renderer.push("<!--[!-->");
                      if (name === "volume") {
                        $$renderer.push("<!--[-->");
                        $$renderer.push(`<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>`);
                      } else {
                        $$renderer.push("<!--[!-->");
                        if (name === "volume-off") {
                          $$renderer.push("<!--[-->");
                          $$renderer.push(`<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line>`);
                        } else {
                          $$renderer.push("<!--[!-->");
                          if (name === "clock") {
                            $$renderer.push("<!--[-->");
                            $$renderer.push(`<circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline>`);
                          } else {
                            $$renderer.push("<!--[!-->");
                            if (name === "user") {
                              $$renderer.push("<!--[-->");
                              $$renderer.push(`<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle>`);
                            } else {
                              $$renderer.push("<!--[!-->");
                              if (name === "settings") {
                                $$renderer.push("<!--[-->");
                                $$renderer.push(`<circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>`);
                              } else {
                                $$renderer.push("<!--[!-->");
                                if (name === "alert") {
                                  $$renderer.push("<!--[-->");
                                  $$renderer.push(`<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>`);
                                } else {
                                  $$renderer.push("<!--[!-->");
                                }
                                $$renderer.push(`<!--]-->`);
                              }
                              $$renderer.push(`<!--]-->`);
                            }
                            $$renderer.push(`<!--]-->`);
                          }
                          $$renderer.push(`<!--]-->`);
                        }
                        $$renderer.push(`<!--]-->`);
                      }
                      $$renderer.push(`<!--]-->`);
                    }
                    $$renderer.push(`<!--]-->`);
                  }
                  $$renderer.push(`<!--]-->`);
                }
                $$renderer.push(`<!--]-->`);
              }
              $$renderer.push(`<!--]-->`);
            }
            $$renderer.push(`<!--]-->`);
          }
          $$renderer.push(`<!--]-->`);
        }
        $$renderer.push(`<!--]-->`);
      }
      $$renderer.push(`<!--]-->`);
    }
    $$renderer.push(`<!--]-->`);
  }
  $$renderer.push(`<!--]--></svg>`);
}
export {
  Icon as I,
  isConnected as a,
  currentConfig as b,
  currentDevice as c,
  currentSession as d,
  drawerOpen as e,
  inputsForDevice as f,
  connectionError as g,
  selectedDate as h,
  isConnecting as i,
  availableDates as j,
  entriesForDate as k,
  selectedEntryId as l,
  selectedEntry as m,
  speakers as n,
  setAudioInput as s,
  timestampToDate as t,
  updateEntry as u
};
