//! SpacetimeDB module for Ambient Listener — Rust port of the C# module.
//!
//! Tables store capture device state, audio sessions, transcripts, diarization,
//! speaker profiles, vision frames, assistant responses, and executive context.
//!
//! Reducer names use PascalCase to match the original C# module so that
//! Python (SpacetimeClient) and Svelte API endpoints work without changes.

#![allow(non_snake_case)]

use spacetimedb::{table, reducer, ReducerContext, Table, Timestamp};

// ============================================================================
// TABLES
// ============================================================================

#[table(name = capture_device, public)]
pub struct CaptureDevice {
    #[primary_key]
    #[auto_inc]
    pub id: u32,

    #[unique]
    pub device_identity: String,

    pub name: String,
    pub platform: String,
    pub status: String,
    pub last_heartbeat: Timestamp,
    pub created_at: Timestamp,
}

#[table(name = audio_session, public)]
pub struct AudioSession {
    #[primary_key]
    #[auto_inc]
    pub id: u32,

    pub device_id: u32,
    pub date: String,
    pub mode: String,
    pub status: String,
    pub audio_path: String,
    pub started_at: Timestamp,
    pub ended_at: Option<Timestamp>,
    pub duration_ms: u32,
}

#[table(name = transcript_entry, public)]
pub struct TranscriptEntry {
    #[primary_key]
    #[auto_inc]
    pub id: u32,

    pub session_id: u32,
    pub entry_id: String,
    pub timestamp: Timestamp,
    pub duration_ms: u32,

    // Transcription
    pub transcript: Option<String>,
    pub confidence: Option<f32>,

    // Labels
    pub speaker: Option<String>,
    pub sentiment: Option<String>,
    pub intent: Option<String>,

    // Audio reference
    pub audio_clip_path: Option<String>,
    pub recording_start_ms: u32,
    pub recording_end_ms: u32,

    // State
    pub reviewed: bool,
    pub notes: Option<String>,

    // Timestamps
    pub created_at: Timestamp,
    pub updated_at: Timestamp,

    // Enrichment tracking
    pub enriched_at: Option<Timestamp>,
}

#[table(name = speaker, public)]
pub struct Speaker {
    #[primary_key]
    #[auto_inc]
    pub id: u32,

    #[unique]
    pub name: String,

    pub similarity_threshold: f32,
    pub embedding: Option<Vec<u8>>,
    pub sample_count: u32,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

#[table(name = diarization_segment, public)]
pub struct DiarizationSegment {
    #[primary_key]
    #[auto_inc]
    pub id: u32,

    pub entry_id: u32,
    pub start_ms: u32,
    pub end_ms: u32,
    pub pyannote_label: String,
    pub matched_speaker: Option<String>,
    pub confidence: Option<f32>,
    pub transcript_slice: Option<String>,
    pub embedding: Option<Vec<u8>>,
}

#[table(name = assistant_response, public)]
pub struct AssistantResponse {
    #[primary_key]
    #[auto_inc]
    pub id: u32,

    pub entry_id: u32,
    pub quick_reply: Option<String>,
    pub full_reply: Option<String>,
    pub reaction: Option<String>,
    pub thinking_log: Option<String>,
    pub created_at: Timestamp,
}

#[table(name = frame, public)]
pub struct Frame {
    #[primary_key]
    #[auto_inc]
    pub id: u32,

    pub session_id: u32,
    pub timestamp: Timestamp,
    pub frame_type: String,
    pub image_path: String,
    pub detections: String,
    pub reviewed: bool,
    pub notes: Option<String>,
    pub created_at: Timestamp,
}

#[table(name = audio_input, public)]
pub struct AudioInput {
    #[primary_key]
    #[auto_inc]
    pub id: u32,

    pub device_id: u32,
    pub input_index: i32,
    pub name: String,
    pub is_default: bool,
    pub updated_at: Timestamp,
}

#[table(name = capture_config, public)]
pub struct CaptureConfig {
    #[primary_key]
    pub device_id: u32,

    pub selected_audio_input_index: i32,

    // VAD settings
    pub silence_threshold: f32,
    pub min_speech_duration_ms: u32,
    pub max_speech_duration_ms: u32,
    pub silence_duration_ms: u32,

    // Transcription settings
    pub min_confidence: f32,
    pub language: String,

    // Recording control
    pub is_paused: bool,
    pub is_muted: bool,

    // Speaker embedding settings
    pub speaker_embedding_enabled: bool,
    pub speaker_embedding_model: String,
    pub speaker_match_threshold: f32,
    pub create_unknown_speakers: bool,

    pub updated_at: Timestamp,
}

#[table(name = executive_context, public)]
pub struct ExecutiveContext {
    #[primary_key]
    pub id: u32, // Always 1 (singleton)

    pub last_updated: Timestamp,

    pub recent_visual: String,
    pub recent_audio: String,

    // Baseline tracking for delta detection
    pub baseline_visual: String,
    pub baseline_audio: String,

    pub user_state: String,
    pub agent_state: String,

    pub next_check_in: Option<i64>,
    pub notes: String,

    pub created_at: Timestamp,
}

// ============================================================================
// DEVICE REDUCERS
// ============================================================================

#[reducer]
pub fn RegisterDevice(
    ctx: &ReducerContext,
    device_identity: String,
    name: String,
    platform: String,
) {
    if let Some(mut existing) = ctx.db.capture_device().device_identity().find(&device_identity) {
        existing.status = "online".to_string();
        existing.last_heartbeat = ctx.timestamp;
        ctx.db.capture_device().device_identity().update(existing);
    } else {
        ctx.db.capture_device().insert(CaptureDevice {
            id: 0,
            device_identity,
            name,
            platform,
            status: "online".to_string(),
            last_heartbeat: ctx.timestamp,
            created_at: ctx.timestamp,
        });
    }
}

#[reducer]
pub fn DeviceHeartbeat(ctx: &ReducerContext, device_identity: String) {
    if let Some(mut device) = ctx.db.capture_device().device_identity().find(&device_identity) {
        device.last_heartbeat = ctx.timestamp;
        ctx.db.capture_device().device_identity().update(device);
    }
}

#[reducer]
pub fn UpdateDeviceStatus(ctx: &ReducerContext, device_identity: String, status: String) {
    if let Some(mut device) = ctx.db.capture_device().device_identity().find(&device_identity) {
        device.status = status;
        device.last_heartbeat = ctx.timestamp;
        ctx.db.capture_device().device_identity().update(device);
    }
}

// ============================================================================
// SESSION REDUCERS
// ============================================================================

#[reducer]
pub fn StartSession(
    ctx: &ReducerContext,
    device_id: u32,
    date: String,
    mode: String,
    audio_path: String,
) {
    ctx.db.audio_session().insert(AudioSession {
        id: 0,
        device_id,
        date,
        mode,
        status: "recording".to_string(),
        audio_path,
        started_at: ctx.timestamp,
        ended_at: None,
        duration_ms: 0,
    });
}

#[reducer]
pub fn UpdateSessionStatus(ctx: &ReducerContext, session_id: u32, status: String) {
    if let Some(mut session) = ctx.db.audio_session().id().find(&session_id) {
        session.status = status.clone();
        if status == "completed" {
            session.ended_at = Some(ctx.timestamp);
        }
        ctx.db.audio_session().id().update(session);
    }
}

#[reducer]
pub fn UpdateSessionDuration(ctx: &ReducerContext, session_id: u32, duration_ms: u32) {
    if let Some(mut session) = ctx.db.audio_session().id().find(&session_id) {
        session.duration_ms = duration_ms;
        ctx.db.audio_session().id().update(session);
    }
}

// ============================================================================
// TRANSCRIPT ENTRY REDUCERS
// ============================================================================

#[reducer]
pub fn CreateEntry(
    ctx: &ReducerContext,
    session_id: u32,
    entry_id: String,
    duration_ms: u32,
    transcript: Option<String>,
    confidence: Option<f32>,
    audio_clip_path: Option<String>,
    recording_start_ms: u32,
    recording_end_ms: u32,
) {
    ctx.db.transcript_entry().insert(TranscriptEntry {
        id: 0,
        session_id,
        entry_id,
        timestamp: ctx.timestamp,
        duration_ms,
        transcript,
        confidence,
        speaker: None,
        sentiment: None,
        intent: None,
        audio_clip_path,
        recording_start_ms,
        recording_end_ms,
        reviewed: false,
        notes: None,
        created_at: ctx.timestamp,
        updated_at: ctx.timestamp,
        enriched_at: None,
    });
}

#[reducer]
pub fn UpdateEntryTranscript(ctx: &ReducerContext, entry_id: u32, transcript: String) {
    if let Some(mut entry) = ctx.db.transcript_entry().id().find(&entry_id) {
        entry.transcript = Some(transcript);
        entry.updated_at = ctx.timestamp;
        ctx.db.transcript_entry().id().update(entry);
    }
}

#[reducer]
pub fn UpdateEntrySpeaker(ctx: &ReducerContext, entry_id: u32, speaker_name: String) {
    if let Some(mut entry) = ctx.db.transcript_entry().id().find(&entry_id) {
        entry.speaker = Some(speaker_name);
        entry.updated_at = ctx.timestamp;
        ctx.db.transcript_entry().id().update(entry);
    }
}

#[reducer]
pub fn UpdateEntrySpeakerByEntryId(
    ctx: &ReducerContext,
    entry_id: String,
    speaker_name: String,
) {
    // Find entry by string entry_id field (not the u32 primary key)
    for entry in ctx.db.transcript_entry().iter() {
        if entry.entry_id == entry_id {
            let mut updated = entry;
            updated.speaker = Some(speaker_name);
            updated.updated_at = ctx.timestamp;
            ctx.db.transcript_entry().id().update(updated);
            return;
        }
    }
}

#[reducer]
pub fn UpdateEntrySentiment(ctx: &ReducerContext, entry_id: u32, sentiment: String) {
    if let Some(mut entry) = ctx.db.transcript_entry().id().find(&entry_id) {
        entry.sentiment = Some(sentiment);
        entry.updated_at = ctx.timestamp;
        ctx.db.transcript_entry().id().update(entry);
    }
}

#[reducer]
pub fn UpdateEntryNotes(ctx: &ReducerContext, entry_id: u32, notes: String) {
    if let Some(mut entry) = ctx.db.transcript_entry().id().find(&entry_id) {
        entry.notes = Some(notes);
        entry.updated_at = ctx.timestamp;
        ctx.db.transcript_entry().id().update(entry);
    }
}

#[reducer]
pub fn MarkEntryReviewed(ctx: &ReducerContext, entry_id: u32, reviewed: bool) {
    if let Some(mut entry) = ctx.db.transcript_entry().id().find(&entry_id) {
        entry.reviewed = reviewed;
        entry.updated_at = ctx.timestamp;
        ctx.db.transcript_entry().id().update(entry);
    }
}

#[reducer]
pub fn MarkEntryEnriched(ctx: &ReducerContext, entry_id: u32) {
    if let Some(mut entry) = ctx.db.transcript_entry().id().find(&entry_id) {
        entry.enriched_at = Some(ctx.timestamp);
        entry.updated_at = ctx.timestamp;
        ctx.db.transcript_entry().id().update(entry);
    }
}

#[reducer]
pub fn UpdateEntry(
    ctx: &ReducerContext,
    entry_id: u32,
    transcript: Option<String>,
    speaker_name: Option<String>,
    sentiment: Option<String>,
    notes: Option<String>,
) {
    if let Some(mut entry) = ctx.db.transcript_entry().id().find(&entry_id) {
        if let Some(t) = transcript {
            entry.transcript = Some(t);
        }
        if let Some(s) = speaker_name {
            entry.speaker = Some(s);
        }
        if let Some(s) = sentiment {
            entry.sentiment = Some(s);
        }
        if let Some(n) = notes {
            entry.notes = Some(n);
        }
        entry.reviewed = true;
        entry.updated_at = ctx.timestamp;
        ctx.db.transcript_entry().id().update(entry);
    }
}

#[reducer]
pub fn DeleteEntry(ctx: &ReducerContext, entry_id: u32) {
    // Delete associated diarization segments first
    let segment_ids: Vec<u32> = ctx
        .db
        .diarization_segment()
        .iter()
        .filter(|s| s.entry_id == entry_id)
        .map(|s| s.id)
        .collect();
    for sid in segment_ids {
        ctx.db.diarization_segment().id().delete(&sid);
    }

    ctx.db.transcript_entry().id().delete(&entry_id);
}

// ============================================================================
// SPEAKER REDUCERS
// ============================================================================

#[reducer]
pub fn CreateOrUpdateSpeaker(
    ctx: &ReducerContext,
    name: String,
    embedding: Option<Vec<u8>>,
) {
    if let Some(mut existing) = ctx.db.speaker().name().find(&name) {
        if embedding.is_some() {
            existing.embedding = embedding;
            existing.sample_count += 1;
        }
        existing.updated_at = ctx.timestamp;
        ctx.db.speaker().name().update(existing);
    } else {
        let has_embedding = embedding.is_some();
        ctx.db.speaker().insert(Speaker {
            id: 0,
            name,
            similarity_threshold: 0.75,
            embedding,
            sample_count: if has_embedding { 1 } else { 0 },
            created_at: ctx.timestamp,
            updated_at: ctx.timestamp,
        });
    }
}

#[reducer]
pub fn UpdateSpeakerThreshold(ctx: &ReducerContext, name: String, threshold: f32) {
    if let Some(mut existing) = ctx.db.speaker().name().find(&name) {
        existing.similarity_threshold = threshold;
        existing.updated_at = ctx.timestamp;
        ctx.db.speaker().name().update(existing);
    }
}

#[reducer]
pub fn DeleteSpeaker(ctx: &ReducerContext, name: String) {
    ctx.db.speaker().name().delete(&name);
}

// ============================================================================
// DIARIZATION SEGMENT REDUCERS
// ============================================================================

#[reducer]
pub fn CreateDiarizationSegment(
    ctx: &ReducerContext,
    entry_id: u32,
    start_ms: u32,
    end_ms: u32,
    pyannote_label: String,
    matched_speaker: Option<String>,
    confidence: Option<f32>,
    transcript_slice: Option<String>,
    embedding: Option<Vec<u8>>,
) {
    ctx.db.diarization_segment().insert(DiarizationSegment {
        id: 0,
        entry_id,
        start_ms,
        end_ms,
        pyannote_label,
        matched_speaker,
        confidence,
        transcript_slice,
        embedding,
    });
}

#[reducer]
pub fn UpdateDiarizationSegmentSpeaker(
    ctx: &ReducerContext,
    segment_id: u32,
    matched_speaker: String,
    confidence: f32,
) {
    if let Some(mut segment) = ctx.db.diarization_segment().id().find(&segment_id) {
        segment.matched_speaker = Some(matched_speaker);
        segment.confidence = Some(confidence);
        ctx.db.diarization_segment().id().update(segment);
    }
}

// ============================================================================
// ASSISTANT RESPONSE REDUCERS
// ============================================================================

#[reducer]
pub fn CreateAssistantResponse(
    ctx: &ReducerContext,
    entry_id: u32,
    quick_reply: Option<String>,
    full_reply: Option<String>,
    reaction: Option<String>,
    thinking_log: Option<String>,
) {
    ctx.db.assistant_response().insert(AssistantResponse {
        id: 0,
        entry_id,
        quick_reply,
        full_reply,
        reaction,
        thinking_log,
        created_at: ctx.timestamp,
    });
}

// ============================================================================
// FRAME REDUCERS (Vision)
// ============================================================================

#[reducer]
pub fn CreateFrame(
    ctx: &ReducerContext,
    session_id: u32,
    frame_type: String,
    image_path: String,
    detections: String,
    reviewed: bool,
    notes: Option<String>,
) {
    ctx.db.frame().insert(Frame {
        id: 0,
        session_id,
        timestamp: ctx.timestamp,
        frame_type,
        image_path,
        detections,
        reviewed,
        notes,
        created_at: ctx.timestamp,
    });
}

#[reducer]
pub fn UpdateFrameDetections(ctx: &ReducerContext, frame_id: u32, detections: String) {
    if let Some(mut frame) = ctx.db.frame().id().find(&frame_id) {
        frame.detections = detections;
        frame.reviewed = true;
        ctx.db.frame().id().update(frame);
    }
}

#[reducer]
pub fn UpdateFrameNotes(ctx: &ReducerContext, frame_id: u32, notes: String) {
    if let Some(mut frame) = ctx.db.frame().id().find(&frame_id) {
        frame.notes = Some(notes);
        ctx.db.frame().id().update(frame);
    }
}

#[reducer]
pub fn MarkFrameReviewed(ctx: &ReducerContext, frame_id: u32, reviewed: bool) {
    if let Some(mut frame) = ctx.db.frame().id().find(&frame_id) {
        frame.reviewed = reviewed;
        ctx.db.frame().id().update(frame);
    }
}

#[reducer]
pub fn DeleteFrame(ctx: &ReducerContext, frame_id: u32) {
    ctx.db.frame().id().delete(&frame_id);
}

// ============================================================================
// CAPTURE CONFIG REDUCERS
// ============================================================================

#[reducer]
pub fn InitCaptureConfig(ctx: &ReducerContext, device_id: u32) {
    // Don't overwrite if already exists
    if ctx.db.capture_config().device_id().find(&device_id).is_some() {
        return;
    }

    ctx.db.capture_config().insert(CaptureConfig {
        device_id,
        selected_audio_input_index: -1,
        silence_threshold: 0.01,
        min_speech_duration_ms: 500,
        max_speech_duration_ms: 30000,
        silence_duration_ms: 1000,
        min_confidence: 0.0,
        language: "en".to_string(),
        is_paused: false,
        is_muted: false,
        speaker_embedding_enabled: false,
        speaker_embedding_model: "3dspeaker_speech_eres2net_base".to_string(),
        speaker_match_threshold: 0.7,
        create_unknown_speakers: true,
        updated_at: ctx.timestamp,
    });
}

#[reducer]
pub fn ReportAudioInputs(ctx: &ReducerContext, device_id: u32, input_names: Vec<String>) {
    // Remove old inputs for this device
    let to_remove: Vec<u32> = ctx
        .db
        .audio_input()
        .iter()
        .filter(|i| i.device_id == device_id)
        .map(|i| i.id)
        .collect();
    for id in to_remove {
        ctx.db.audio_input().id().delete(&id);
    }

    // Insert new inputs
    for (i, name) in input_names.into_iter().enumerate() {
        ctx.db.audio_input().insert(AudioInput {
            id: 0,
            device_id,
            input_index: i as i32,
            name,
            is_default: i == 0,
            updated_at: ctx.timestamp,
        });
    }
}

#[reducer]
pub fn SetAudioInput(ctx: &ReducerContext, device_id: u32, input_index: i32) {
    if let Some(mut config) = ctx.db.capture_config().device_id().find(&device_id) {
        config.selected_audio_input_index = input_index;
        config.updated_at = ctx.timestamp;
        ctx.db.capture_config().device_id().update(config);
    }
}

#[reducer]
pub fn UpdateVadConfig(
    ctx: &ReducerContext,
    device_id: u32,
    silence_threshold: f32,
    min_speech_duration_ms: u32,
    max_speech_duration_ms: u32,
    silence_duration_ms: u32,
) {
    if let Some(mut config) = ctx.db.capture_config().device_id().find(&device_id) {
        config.silence_threshold = silence_threshold;
        config.min_speech_duration_ms = min_speech_duration_ms;
        config.max_speech_duration_ms = max_speech_duration_ms;
        config.silence_duration_ms = silence_duration_ms;
        config.updated_at = ctx.timestamp;
        ctx.db.capture_config().device_id().update(config);
    }
}

#[reducer]
pub fn SetSilenceThreshold(ctx: &ReducerContext, device_id: u32, threshold: f32) {
    if let Some(mut config) = ctx.db.capture_config().device_id().find(&device_id) {
        config.silence_threshold = threshold;
        config.updated_at = ctx.timestamp;
        ctx.db.capture_config().device_id().update(config);
    }
}

#[reducer]
pub fn UpdateTranscriptionConfig(
    ctx: &ReducerContext,
    device_id: u32,
    min_confidence: f32,
    language: String,
) {
    if let Some(mut config) = ctx.db.capture_config().device_id().find(&device_id) {
        config.min_confidence = min_confidence;
        config.language = language;
        config.updated_at = ctx.timestamp;
        ctx.db.capture_config().device_id().update(config);
    }
}

#[reducer]
pub fn SetCapturePaused(ctx: &ReducerContext, device_id: u32, is_paused: bool) {
    if let Some(mut config) = ctx.db.capture_config().device_id().find(&device_id) {
        config.is_paused = is_paused;
        config.updated_at = ctx.timestamp;
        ctx.db.capture_config().device_id().update(config);
    }
}

#[reducer]
pub fn SetCaptureMuted(ctx: &ReducerContext, device_id: u32, is_muted: bool) {
    if let Some(mut config) = ctx.db.capture_config().device_id().find(&device_id) {
        config.is_muted = is_muted;
        config.updated_at = ctx.timestamp;
        ctx.db.capture_config().device_id().update(config);
    }
}

#[reducer]
pub fn UpdateSpeakerEmbeddingConfig(
    ctx: &ReducerContext,
    device_id: u32,
    enabled: bool,
    model: String,
    match_threshold: f32,
    create_unknown: bool,
) {
    if let Some(mut config) = ctx.db.capture_config().device_id().find(&device_id) {
        config.speaker_embedding_enabled = enabled;
        config.speaker_embedding_model = model;
        config.speaker_match_threshold = match_threshold;
        config.create_unknown_speakers = create_unknown;
        config.updated_at = ctx.timestamp;
        ctx.db.capture_config().device_id().update(config);
    }
}

// ============================================================================
// EXECUTIVE CONTEXT REDUCER
// ============================================================================

#[reducer]
pub fn UpdateExecutiveContext(
    ctx: &ReducerContext,
    recent_visual: Option<String>,
    recent_audio: Option<String>,
    baseline_visual: Option<String>,
    baseline_audio: Option<String>,
    user_state: Option<String>,
    agent_state: Option<String>,
    next_check_in: Option<i64>,
    notes: Option<String>,
) {
    if let Some(mut existing) = ctx.db.executive_context().id().find(&1) {
        existing.last_updated = ctx.timestamp;

        if let Some(v) = recent_visual {
            existing.recent_visual = v;
        }
        if let Some(a) = recent_audio {
            existing.recent_audio = a;
        }
        if let Some(v) = baseline_visual {
            existing.baseline_visual = v;
        }
        if let Some(a) = baseline_audio {
            existing.baseline_audio = a;
        }
        if let Some(s) = user_state {
            existing.user_state = s;
        }
        if let Some(s) = agent_state {
            existing.agent_state = s;
        }
        if next_check_in.is_some() {
            existing.next_check_in = next_check_in;
        }
        if let Some(n) = notes {
            existing.notes = n;
        }

        ctx.db.executive_context().id().update(existing);
    } else {
        // Initialize on first update
        ctx.db.executive_context().insert(ExecutiveContext {
            id: 1,
            last_updated: ctx.timestamp,
            recent_visual: recent_visual.unwrap_or_else(|| "[]".to_string()),
            recent_audio: recent_audio.unwrap_or_else(|| "[]".to_string()),
            baseline_visual: baseline_visual.unwrap_or_else(|| "{}".to_string()),
            baseline_audio: baseline_audio.unwrap_or_else(|| "{}".to_string()),
            user_state: user_state.unwrap_or_else(|| "idle".to_string()),
            agent_state: agent_state.unwrap_or_else(|| "active".to_string()),
            next_check_in,
            notes: notes.unwrap_or_else(|| "[]".to_string()),
            created_at: ctx.timestamp,
        });
    }
}
