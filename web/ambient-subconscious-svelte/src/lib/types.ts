// Re-export generated types from SpacetimeDB bindings
import type { Infer } from 'spacetimedb';
import TranscriptEntryType from '../generated/transcript_entry_type';
import AudioSessionType from '../generated/audio_session_type';
import AudioInputType from '../generated/audio_input_type';
import CaptureDeviceType from '../generated/capture_device_type';
import SpeakerType from '../generated/speaker_type';
import DiarizationSegmentType from '../generated/diarization_segment_type';
import CaptureConfigType from '../generated/capture_config_type';
import FrameType from '../generated/frame_type';

// Inferred row types from the generated table types
export type TranscriptEntry = Infer<typeof TranscriptEntryType>;
export type AudioSession = Infer<typeof AudioSessionType>;
export type AudioInput = Infer<typeof AudioInputType>;
export type CaptureDevice = Infer<typeof CaptureDeviceType>;
export type Speaker = Infer<typeof SpeakerType>;
export type DiarizationSegment = Infer<typeof DiarizationSegmentType>;
export type CaptureConfig = Infer<typeof CaptureConfigType>;
export type Frame = Infer<typeof FrameType>;

// Helper types for frame detections
export interface Detection {
  class: string;
  bbox: number[];
  confidence: number;
}
