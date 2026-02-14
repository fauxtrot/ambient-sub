using SpacetimeDB;

namespace AmbientListener.SpacetimeDb;

// ============================================================================
// TABLES
// ============================================================================

[Table(Name = "capture_device", Public = true)]
public partial class CaptureDevice
{
    [PrimaryKey, AutoInc]
    public uint Id;

    [Unique]
    public string DeviceIdentity = "";  // Machine GUID or similar

    public string Name = "";
    public string Platform = "";        // "Windows", "Linux", "macOS"
    public string Status = "";          // "online", "offline", "recording"
    public Timestamp LastHeartbeat;
    public Timestamp CreatedAt;
}

[Table(Name = "audio_session", Public = true)]
public partial class AudioSession
{
    [PrimaryKey, AutoInc]
    public uint Id;

    public uint DeviceId;
    public string Date = "";            // "2026-01-23"
    public string Mode = "";            // "ambient", "meeting", "focus", "phone"
    public string Status = "";          // "recording", "paused", "completed"
    public string AudioPath = "";       // Path to continuous recording
    public Timestamp StartedAt;
    public Timestamp? EndedAt;
    public uint DurationMs;
}

[Table(Name = "transcript_entry", Public = true)]
public partial class TranscriptEntry
{
    [PrimaryKey, AutoInc]
    public uint Id;

    public uint SessionId;
    public string EntryId = "";         // "2026-01-23_14-30-45_a1b2" format
    public Timestamp Timestamp;
    public uint DurationMs;

    // Transcription
    public string? Transcript;
    public float? Confidence;

    // Labels
    public string? Speaker;
    public string? Sentiment;
    public string? Intent;

    // Audio reference
    public string? AudioClipPath;
    public uint RecordingStartMs;  // Offset into continuous recording
    public uint RecordingEndMs;

    // State
    public bool Reviewed;
    public string? Notes;

    // Timestamps
    public Timestamp CreatedAt;
    public Timestamp UpdatedAt;

    // Enrichment tracking (added at end to avoid migration)
    public Timestamp? EnrichedAt;        // Set when Claude enriches this entry
}

[Table(Name = "speaker", Public = true)]
public partial class Speaker
{
    [PrimaryKey, AutoInc]
    public uint Id;

    [Unique]
    public string Name = "";

    public float SimilarityThreshold;  // Default 0.75
    public byte[]? Embedding;          // Speaker voice embedding
    public uint SampleCount;           // Number of samples used
    public Timestamp CreatedAt;
    public Timestamp UpdatedAt;
}

[Table(Name = "diarization_segment", Public = true)]
public partial class DiarizationSegment
{
    [PrimaryKey, AutoInc]
    public uint Id;

    public uint EntryId;
    public uint StartMs;           // Relative to entry start
    public uint EndMs;
    public string PyannoteLabel = "";   // "SPEAKER_00", etc. (or "REALTIME" for C# capture)
    public string? MatchedSpeaker;
    public float? Confidence;
    public string? TranscriptSlice;
    public byte[]? Embedding;      // Speaker embedding vector (for real-time capture)
}

/// <summary>
/// Stores AI assistant responses linked to triggering utterances.
/// </summary>
[Table(Name = "assistant_response", Public = true)]
public partial class AssistantResponse
{
    [PrimaryKey, AutoInc]
    public uint Id;

    public uint EntryId;              // Links to transcript_entry that triggered this
    public string? QuickReply;        // Filler phrase ("Sure!", "Let me think...")
    public string? FullReply;         // The complete spoken response
    public string? Reaction;          // Avatar reaction set (confused, happy, etc.)
    public string? ThinkingLog;       // Optional: LLM reasoning (for debugging)
    public Timestamp CreatedAt;
}

/// <summary>
/// Visual frames captured from webcam or screen with YOLO object detection.
/// </summary>
[Table(Name = "frame", Public = true)]
public partial class Frame
{
    [PrimaryKey, AutoInc]
    public uint Id;

    public uint SessionId;             // Foreign key to audio_session
    public Timestamp Timestamp;
    public string FrameType = "";      // "webcam" or "screen"
    public string ImagePath = "";      // Path to saved frame image
    public string Detections = "";     // JSON: [{"class": "person", "bbox": [x,y,x2,y2], "confidence": 0.95}]
    public bool Reviewed;
    public string? Notes;              // Optional user notes or inferred context
    public Timestamp CreatedAt;
}

/// <summary>
/// Available audio input devices on a capture device (microphones, virtual cables, etc.)
/// </summary>
[Table(Name = "audio_input", Public = true)]
public partial class AudioInput
{
    [PrimaryKey, AutoInc]
    public uint Id;

    public uint DeviceId;              // Foreign key to capture_device
    public int InputIndex;             // NAudio device index
    public string Name = "";           // Device name (e.g., "Microphone (Realtek)")
    public bool IsDefault;             // Is this the system default?
    public Timestamp UpdatedAt;
}

/// <summary>
/// Runtime configuration for capture devices. Changes are applied in real-time.
/// </summary>
[Table(Name = "capture_config", Public = true)]
public partial class CaptureConfig
{
    [PrimaryKey]
    public uint DeviceId;              // Foreign key to capture_device

    // Audio input selection
    public int SelectedAudioInputIndex; // NAudio device index (-1 = default)

    // VAD settings
    public float SilenceThreshold;     // RMS energy threshold (default 0.01)
    public uint MinSpeechDurationMs;   // Minimum speech segment length (default 500)
    public uint MaxSpeechDurationMs;   // Maximum speech segment length (default 30000)
    public uint SilenceDurationMs;     // Silence duration to end segment (default 1000)

    // Transcription settings
    public float MinConfidence;        // Minimum confidence to keep entry (default 0.0)
    public string Language = "en";     // Whisper language code (default "en")

    // Recording control
    public bool IsPaused;              // Pause capture without stopping session
    public bool IsMuted;               // Capture but don't transcribe

    // Speaker embedding settings (real-time speaker identification)
    public bool SpeakerEmbeddingEnabled;           // Feature toggle (default false)
    public string SpeakerEmbeddingModel = "";      // Model name (e.g., "3dspeaker_speech_eres2net_base")
    public float SpeakerMatchThreshold;            // Cosine similarity threshold (default 0.7)
    public bool CreateUnknownSpeakers;             // Auto-create profiles for unmatched speakers

    public Timestamp UpdatedAt;
}

/// <summary>
/// Aggregated context for executive agent. Single row (ID=1) updated by Observer.
/// </summary>
[Table(Name = "executive_context", Public = true)]
public partial class ExecutiveContext
{
    [PrimaryKey]
    public uint Id;  // Always 1 (singleton table)

    public Timestamp LastUpdated;

    // Aggregated visual context from recent frames
    public string RecentVisual = "";  // JSON: ["laptop", "person", "coffee", "monitor", "phone"]

    // Aggregated audio context from recent entries
    public string RecentAudio = "";   // JSON: [{"speaker": "User", "text": "..."}, ...]

    // NEW: Baseline tracking for delta detection
    public string BaselineVisual = "";     // JSON: {"avg_objects": 5.2, "common_objects": ["laptop", "monitor"]}
    public string BaselineAudio = "";      // JSON: {"avg_entries_per_min": 0.5}

    // Inferred user state (updated by Observer or Executive)
    public string UserState = "";     // "focused", "idle", "in_meeting", etc.

    // Agent state
    public string AgentState = "";    // "active", "idle", "thinking", "wake" (manual trigger)

    // NEW: Scheduled check-in (Unix timestamp in seconds)
    public long? NextCheckIn = null;  // Next time Observer should send update (even without delta)

    // Agent notes (appended by Executive for memory)
    public string Notes = "";         // JSON array of timestamped notes

    public Timestamp CreatedAt;
}

// ============================================================================
// REDUCERS
// ============================================================================

public static partial class Reducers
{
    [Reducer]
    public static void RegisterDevice(ReducerContext ctx, string deviceIdentity, string name, string platform)
    {
        var existing = ctx.Db.capture_device.DeviceIdentity.Find(deviceIdentity);
        if (existing != null)
        {
            existing.Status = "online";
            existing.LastHeartbeat = ctx.Timestamp;
            ctx.Db.capture_device.DeviceIdentity.Update(existing);
        }
        else
        {
            ctx.Db.capture_device.Insert(new CaptureDevice
            {
                DeviceIdentity = deviceIdentity,
                Name = name,
                Platform = platform,
                Status = "online",
                LastHeartbeat = ctx.Timestamp,
                CreatedAt = ctx.Timestamp
            });
        }
    }

    [Reducer]
    public static void DeviceHeartbeat(ReducerContext ctx, string deviceIdentity)
    {
        var device = ctx.Db.capture_device.DeviceIdentity.Find(deviceIdentity);
        if (device != null)
        {
            device.LastHeartbeat = ctx.Timestamp;
            ctx.Db.capture_device.DeviceIdentity.Update(device);
        }
    }

    [Reducer]
    public static void UpdateDeviceStatus(ReducerContext ctx, string deviceIdentity, string status)
    {
        var device = ctx.Db.capture_device.DeviceIdentity.Find(deviceIdentity);
        if (device != null)
        {
            device.Status = status;
            device.LastHeartbeat = ctx.Timestamp;
            ctx.Db.capture_device.DeviceIdentity.Update(device);
        }
    }

    [Reducer]
    public static void StartSession(ReducerContext ctx, uint deviceId, string date, string mode, string audioPath)
    {
        ctx.Db.audio_session.Insert(new AudioSession
        {
            DeviceId = deviceId,
            Date = date,
            Mode = mode,
            Status = "recording",
            AudioPath = audioPath,
            StartedAt = ctx.Timestamp,
            DurationMs = 0
        });
    }

    [Reducer]
    public static void UpdateSessionStatus(ReducerContext ctx, uint sessionId, string status)
    {
        var session = ctx.Db.audio_session.Id.Find(sessionId);
        if (session != null)
        {
            session.Status = status;
            if (status == "completed")
            {
                session.EndedAt = ctx.Timestamp;
            }
            ctx.Db.audio_session.Id.Update(session);
        }
    }

    [Reducer]
    public static void UpdateSessionDuration(ReducerContext ctx, uint sessionId, uint durationMs)
    {
        var session = ctx.Db.audio_session.Id.Find(sessionId);
        if (session != null)
        {
            session.DurationMs = durationMs;
            ctx.Db.audio_session.Id.Update(session);
        }
    }

    [Reducer]
    public static void CreateEntry(ReducerContext ctx,
        uint sessionId,
        string entryId,
        uint durationMs,
        string? transcript,
        float? confidence,
        string? audioClipPath,
        uint recordingStartMs,
        uint recordingEndMs)
    {
        ctx.Db.transcript_entry.Insert(new TranscriptEntry
        {
            SessionId = sessionId,
            EntryId = entryId,
            Timestamp = ctx.Timestamp,
            DurationMs = durationMs,
            Transcript = transcript,
            Confidence = confidence,
            AudioClipPath = audioClipPath,
            RecordingStartMs = recordingStartMs,
            RecordingEndMs = recordingEndMs,
            Reviewed = false,
            CreatedAt = ctx.Timestamp,
            UpdatedAt = ctx.Timestamp
        });
    }

    [Reducer]
    public static void UpdateEntryTranscript(ReducerContext ctx, uint entryId, string transcript)
    {
        var entry = ctx.Db.transcript_entry.Id.Find(entryId);
        if (entry != null)
        {
            entry.Transcript = transcript;
            entry.UpdatedAt = ctx.Timestamp;
            ctx.Db.transcript_entry.Id.Update(entry);
        }
    }

    [Reducer]
    public static void UpdateEntrySpeaker(ReducerContext ctx, uint entryId, string speakerName)
    {
        var entry = ctx.Db.transcript_entry.Id.Find(entryId);
        if (entry != null)
        {
            entry.Speaker = speakerName;
            entry.UpdatedAt = ctx.Timestamp;
            ctx.Db.transcript_entry.Id.Update(entry);
        }
    }

    [Reducer]
    public static void UpdateEntrySpeakerByEntryId(ReducerContext ctx, string entryId, string speakerName)
    {
        // Find entry by EntryId string field (not Id uint)
        foreach (var entry in ctx.Db.transcript_entry.Iter())
        {
            if (entry.EntryId == entryId)
            {
                entry.Speaker = speakerName;
                entry.UpdatedAt = ctx.Timestamp;
                ctx.Db.transcript_entry.Id.Update(entry);
                return;
            }
        }
    }

    [Reducer]
    public static void UpdateEntrySentiment(ReducerContext ctx, uint entryId, string sentiment)
    {
        var entry = ctx.Db.transcript_entry.Id.Find(entryId);
        if (entry != null)
        {
            entry.Sentiment = sentiment;
            entry.UpdatedAt = ctx.Timestamp;
            ctx.Db.transcript_entry.Id.Update(entry);
        }
    }

    [Reducer]
    public static void UpdateEntryNotes(ReducerContext ctx, uint entryId, string notes)
    {
        var entry = ctx.Db.transcript_entry.Id.Find(entryId);
        if (entry != null)
        {
            entry.Notes = notes;
            entry.UpdatedAt = ctx.Timestamp;
            ctx.Db.transcript_entry.Id.Update(entry);
        }
    }

    [Reducer]
    public static void MarkEntryReviewed(ReducerContext ctx, uint entryId, bool reviewed)
    {
        var entry = ctx.Db.transcript_entry.Id.Find(entryId);
        if (entry != null)
        {
            entry.Reviewed = reviewed;
            entry.UpdatedAt = ctx.Timestamp;
            ctx.Db.transcript_entry.Id.Update(entry);
        }
    }

    [Reducer]
    public static void MarkEntryEnriched(ReducerContext ctx, uint entryId)
    {
        var entry = ctx.Db.transcript_entry.Id.Find(entryId);
        if (entry != null)
        {
            entry.EnrichedAt = ctx.Timestamp;
            entry.UpdatedAt = ctx.Timestamp;
            ctx.Db.transcript_entry.Id.Update(entry);
        }
    }

    [Reducer]
    public static void UpdateEntry(ReducerContext ctx,
        uint entryId,
        string? transcript,
        string? speakerName,
        string? sentiment,
        string? notes)
    {
        var entry = ctx.Db.transcript_entry.Id.Find(entryId);
        if (entry == null) return;

        if (transcript != null) entry.Transcript = transcript;
        if (speakerName != null) entry.Speaker = speakerName;
        if (sentiment != null) entry.Sentiment = sentiment;
        if (notes != null) entry.Notes = notes;

        entry.Reviewed = true;
        entry.UpdatedAt = ctx.Timestamp;

        ctx.Db.transcript_entry.Id.Update(entry);
    }

    [Reducer]
    public static void DeleteEntry(ReducerContext ctx, uint entryId)
    {
        var entry = ctx.Db.transcript_entry.Id.Find(entryId);
        if (entry != null)
        {
            // Delete associated diarization segments first
            foreach (var segment in ctx.Db.diarization_segment.Iter())
            {
                if (segment.EntryId == entryId)
                {
                    ctx.Db.diarization_segment.Id.Delete(segment.Id);
                }
            }
            ctx.Db.transcript_entry.Id.Delete(entryId);
        }
    }

    [Reducer]
    public static void CreateOrUpdateSpeaker(ReducerContext ctx, string name, byte[]? embedding)
    {
        var existing = ctx.Db.speaker.Name.Find(name);
        if (existing != null)
        {
            if (embedding != null)
            {
                existing.Embedding = embedding;
                existing.SampleCount += 1;
            }
            existing.UpdatedAt = ctx.Timestamp;
            ctx.Db.speaker.Name.Update(existing);
        }
        else
        {
            ctx.Db.speaker.Insert(new Speaker
            {
                Name = name,
                SimilarityThreshold = 0.75f,
                Embedding = embedding,
                SampleCount = embedding != null ? 1u : 0u,
                CreatedAt = ctx.Timestamp,
                UpdatedAt = ctx.Timestamp
            });
        }
    }

    [Reducer]
    public static void UpdateSpeakerThreshold(ReducerContext ctx, string name, float threshold)
    {
        var existing = ctx.Db.speaker.Name.Find(name);
        if (existing != null)
        {
            existing.SimilarityThreshold = threshold;
            existing.UpdatedAt = ctx.Timestamp;
            ctx.Db.speaker.Name.Update(existing);
        }
    }

    [Reducer]
    public static void DeleteSpeaker(ReducerContext ctx, string name)
    {
        var existing = ctx.Db.speaker.Name.Find(name);
        if (existing != null)
        {
            ctx.Db.speaker.Name.Delete(name);
        }
    }

    [Reducer]
    public static void CreateDiarizationSegment(ReducerContext ctx,
        uint entryId,
        uint startMs,
        uint endMs,
        string pyannoteLabel,
        string? matchedSpeaker,
        float? confidence,
        string? transcriptSlice,
        byte[]? embedding)
    {
        ctx.Db.diarization_segment.Insert(new DiarizationSegment
        {
            EntryId = entryId,
            StartMs = startMs,
            EndMs = endMs,
            PyannoteLabel = pyannoteLabel,
            MatchedSpeaker = matchedSpeaker,
            Confidence = confidence,
            TranscriptSlice = transcriptSlice,
            Embedding = embedding
        });
    }

    [Reducer]
    public static void UpdateDiarizationSegmentSpeaker(ReducerContext ctx, uint segmentId, string matchedSpeaker, float confidence)
    {
        var segment = ctx.Db.diarization_segment.Id.Find(segmentId);
        if (segment != null)
        {
            segment.MatchedSpeaker = matchedSpeaker;
            segment.Confidence = confidence;
            ctx.Db.diarization_segment.Id.Update(segment);
        }
    }

    // ============================================================================
    // ASSISTANT RESPONSE REDUCERS
    // ============================================================================

    [Reducer]
    public static void CreateAssistantResponse(
        ReducerContext ctx,
        uint entryId,
        string? quickReply,
        string? fullReply,
        string? reaction,
        string? thinkingLog)
    {
        ctx.Db.assistant_response.Insert(new AssistantResponse
        {
            EntryId = entryId,
            QuickReply = quickReply,
            FullReply = fullReply,
            Reaction = reaction,
            ThinkingLog = thinkingLog,
            CreatedAt = ctx.Timestamp
        });
    }

    // ============================================================================
    // FRAME REDUCERS (Vision)
    // ============================================================================

    /// <summary>
    /// Create a new frame record from webcam or screen capture.
    /// </summary>
    [Reducer]
    public static void CreateFrame(
        ReducerContext ctx,
        uint sessionId,
        string frameType,
        string imagePath,
        string detections,
        bool reviewed,
        string? notes)
    {
        ctx.Db.frame.Insert(new Frame
        {
            SessionId = sessionId,
            Timestamp = ctx.Timestamp,
            FrameType = frameType,
            ImagePath = imagePath,
            Detections = detections,
            Reviewed = reviewed,
            Notes = notes,
            CreatedAt = ctx.Timestamp
        });
    }

    /// <summary>
    /// Update YOLO detections for a frame (user corrections).
    /// </summary>
    [Reducer]
    public static void UpdateFrameDetections(ReducerContext ctx, uint frameId, string detections)
    {
        var frame = ctx.Db.frame.Id.Find(frameId);
        if (frame != null)
        {
            frame.Detections = detections;
            frame.Reviewed = true;
            ctx.Db.frame.Id.Update(frame);
        }
    }

    /// <summary>
    /// Update notes for a frame.
    /// </summary>
    [Reducer]
    public static void UpdateFrameNotes(ReducerContext ctx, uint frameId, string notes)
    {
        var frame = ctx.Db.frame.Id.Find(frameId);
        if (frame != null)
        {
            frame.Notes = notes;
            ctx.Db.frame.Id.Update(frame);
        }
    }

    /// <summary>
    /// Mark frame as reviewed by user.
    /// </summary>
    [Reducer]
    public static void MarkFrameReviewed(ReducerContext ctx, uint frameId, bool reviewed)
    {
        var frame = ctx.Db.frame.Id.Find(frameId);
        if (frame != null)
        {
            frame.Reviewed = reviewed;
            ctx.Db.frame.Id.Update(frame);
        }
    }

    /// <summary>
    /// Delete a frame record.
    /// </summary>
    [Reducer]
    public static void DeleteFrame(ReducerContext ctx, uint frameId)
    {
        var frame = ctx.Db.frame.Id.Find(frameId);
        if (frame != null)
        {
            ctx.Db.frame.Id.Delete(frameId);
        }
    }

    // ============================================================================
    // CAPTURE CONFIG REDUCERS
    // ============================================================================

    /// <summary>
    /// Initialize config for a device with default values. Called when device registers.
    /// </summary>
    [Reducer]
    public static void InitCaptureConfig(ReducerContext ctx, uint deviceId)
    {
        var existing = ctx.Db.capture_config.DeviceId.Find(deviceId);
        if (existing != null) return;  // Already exists

        ctx.Db.capture_config.Insert(new CaptureConfig
        {
            DeviceId = deviceId,
            SelectedAudioInputIndex = -1,  // -1 = use default
            SilenceThreshold = 0.01f,
            MinSpeechDurationMs = 500,
            MaxSpeechDurationMs = 30000,
            SilenceDurationMs = 1000,
            MinConfidence = 0.0f,
            Language = "en",
            IsPaused = false,
            IsMuted = false,
            SpeakerEmbeddingEnabled = false,
            SpeakerEmbeddingModel = "3dspeaker_speech_eres2net_base",
            SpeakerMatchThreshold = 0.7f,
            CreateUnknownSpeakers = true,
            UpdatedAt = ctx.Timestamp
        });
    }

    /// <summary>
    /// Report available audio input devices from capture client.
    /// </summary>
    [Reducer]
    public static void ReportAudioInputs(ReducerContext ctx, uint deviceId, List<string> inputNames)
    {
        // Remove old inputs for this device
        var toRemove = new List<uint>();
        foreach (var input in ctx.Db.audio_input.Iter())
        {
            if (input.DeviceId == deviceId)
            {
                toRemove.Add(input.Id);
            }
        }
        foreach (var id in toRemove)
        {
            ctx.Db.audio_input.Id.Delete(id);
        }

        // Insert new inputs
        for (int i = 0; i < inputNames.Count; i++)
        {
            ctx.Db.audio_input.Insert(new AudioInput
            {
                DeviceId = deviceId,
                InputIndex = i,
                Name = inputNames[i],
                IsDefault = i == 0,
                UpdatedAt = ctx.Timestamp
            });
        }
    }

    /// <summary>
    /// Select which audio input device to use for capture.
    /// </summary>
    [Reducer]
    public static void SetAudioInput(ReducerContext ctx, uint deviceId, int inputIndex)
    {
        var config = ctx.Db.capture_config.DeviceId.Find(deviceId);
        if (config == null) return;

        config.SelectedAudioInputIndex = inputIndex;
        config.UpdatedAt = ctx.Timestamp;
        ctx.Db.capture_config.DeviceId.Update(config);
    }

    /// <summary>
    /// Update VAD settings for a device.
    /// </summary>
    [Reducer]
    public static void UpdateVadConfig(ReducerContext ctx,
        uint deviceId,
        float silenceThreshold,
        uint minSpeechDurationMs,
        uint maxSpeechDurationMs,
        uint silenceDurationMs)
    {
        var config = ctx.Db.capture_config.DeviceId.Find(deviceId);
        if (config == null) return;

        config.SilenceThreshold = silenceThreshold;
        config.MinSpeechDurationMs = minSpeechDurationMs;
        config.MaxSpeechDurationMs = maxSpeechDurationMs;
        config.SilenceDurationMs = silenceDurationMs;
        config.UpdatedAt = ctx.Timestamp;
        ctx.Db.capture_config.DeviceId.Update(config);
    }

    /// <summary>
    /// Update just the silence threshold.
    /// </summary>
    [Reducer]
    public static void SetSilenceThreshold(ReducerContext ctx, uint deviceId, float threshold)
    {
        var config = ctx.Db.capture_config.DeviceId.Find(deviceId);
        if (config == null) return;

        config.SilenceThreshold = threshold;
        config.UpdatedAt = ctx.Timestamp;
        ctx.Db.capture_config.DeviceId.Update(config);
    }

    /// <summary>
    /// Update transcription settings.
    /// </summary>
    [Reducer]
    public static void UpdateTranscriptionConfig(ReducerContext ctx, uint deviceId, float minConfidence, string language)
    {
        var config = ctx.Db.capture_config.DeviceId.Find(deviceId);
        if (config == null) return;

        config.MinConfidence = minConfidence;
        config.Language = language;
        config.UpdatedAt = ctx.Timestamp;
        ctx.Db.capture_config.DeviceId.Update(config);
    }

    /// <summary>
    /// Pause or resume capture.
    /// </summary>
    [Reducer]
    public static void SetCapturePaused(ReducerContext ctx, uint deviceId, bool isPaused)
    {
        var config = ctx.Db.capture_config.DeviceId.Find(deviceId);
        if (config == null) return;

        config.IsPaused = isPaused;
        config.UpdatedAt = ctx.Timestamp;
        ctx.Db.capture_config.DeviceId.Update(config);
    }

    /// <summary>
    /// Mute or unmute transcription (still captures audio).
    /// </summary>
    [Reducer]
    public static void SetCaptureMuted(ReducerContext ctx, uint deviceId, bool isMuted)
    {
        var config = ctx.Db.capture_config.DeviceId.Find(deviceId);
        if (config == null) return;

        config.IsMuted = isMuted;
        config.UpdatedAt = ctx.Timestamp;
        ctx.Db.capture_config.DeviceId.Update(config);
    }

    /// <summary>
    /// Update speaker embedding configuration for real-time speaker identification.
    /// </summary>
    [Reducer]
    public static void UpdateSpeakerEmbeddingConfig(
        ReducerContext ctx,
        uint deviceId,
        bool enabled,
        string model,
        float matchThreshold,
        bool createUnknown)
    {
        var config = ctx.Db.capture_config.DeviceId.Find(deviceId);
        if (config == null) return;

        config.SpeakerEmbeddingEnabled = enabled;
        config.SpeakerEmbeddingModel = model;
        config.SpeakerMatchThreshold = matchThreshold;
        config.CreateUnknownSpeakers = createUnknown;
        config.UpdatedAt = ctx.Timestamp;
        ctx.Db.capture_config.DeviceId.Update(config);
    }

    /// <summary>
    /// Update executive context (called by Observer or Executive process).
    /// All parameters except ctx are nullable - only non-null fields are updated.
    /// </summary>
    [Reducer]
    public static void UpdateExecutiveContext(
        ReducerContext ctx,
        string? recentVisual,
        string? recentAudio,
        string? baselineVisual,      // NEW
        string? baselineAudio,       // NEW
        string? userState,
        string? agentState,
        long? nextCheckIn,           // NEW
        string? notes)
    {
        var existing = ctx.Db.executive_context.Id.Find(1);
        if (existing != null)
        {
            existing.LastUpdated = ctx.Timestamp;

            // Update only non-null fields (partial updates)
            if (recentVisual != null) existing.RecentVisual = recentVisual;
            if (recentAudio != null) existing.RecentAudio = recentAudio;
            if (baselineVisual != null) existing.BaselineVisual = baselineVisual;
            if (baselineAudio != null) existing.BaselineAudio = baselineAudio;
            if (userState != null) existing.UserState = userState;
            if (agentState != null) existing.AgentState = agentState;
            if (nextCheckIn.HasValue) existing.NextCheckIn = nextCheckIn;
            if (notes != null) existing.Notes = notes;

            ctx.Db.executive_context.Id.Update(existing);
        }
        else
        {
            // Initialize on first update
            ctx.Db.executive_context.Insert(new ExecutiveContext
            {
                Id = 1,
                LastUpdated = ctx.Timestamp,
                RecentVisual = recentVisual ?? "[]",
                RecentAudio = recentAudio ?? "[]",
                BaselineVisual = baselineVisual ?? "{}",
                BaselineAudio = baselineAudio ?? "{}",
                UserState = userState ?? "idle",
                AgentState = agentState ?? "active",
                NextCheckIn = nextCheckIn,
                Notes = notes ?? "[]",
                CreatedAt = ctx.Timestamp
            });
        }
    }
}
