//! Structured prompts and session plans — the `Nomas-PE` output contract.
//!
//! An interactive session is a sequence of segments; each segment carries a
//! free-form text prompt, zero or more **timed actions**
//! (`contained_actions: [{start_time, end_time, action_description}]`) and an
//! optional camera trajectory. This mirrors the upstream prompt structure and
//! the multi-turn `reactor` runtime.

use serde::{Deserialize, Serialize};

use crate::actions::ActionSpec;
use crate::camera::CameraTrajectory;

/// One segment's generation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentPlan {
    /// Free-form scene description for this turn.
    pub prompt: String,
    /// Timed actions inside the segment.
    pub actions: Vec<ActionSpec>,
    /// Camera trajectory (one pose per latent frame). `None` keeps the
    /// default (static) camera.
    pub camera: Option<CameraTrajectory>,
    /// Latent frames to generate.
    pub latent_frames: usize,
    /// Optional text-prompt token budget hint for the encoder.
    pub max_text_tokens: usize,
}

impl SegmentPlan {
    /// A plain text-only segment.
    pub fn text(prompt: impl Into<String>, latent_frames: usize) -> Self {
        Self {
            prompt: prompt.into(),
            actions: Vec::new(),
            camera: None,
            latent_frames,
            max_text_tokens: 32,
        }
    }

    /// A camera-only segment (empty prompt).
    pub fn camera(traj: CameraTrajectory, latent_frames: usize) -> Self {
        Self {
            prompt: String::new(),
            actions: Vec::new(),
            camera: Some(traj),
            latent_frames,
            max_text_tokens: 32,
        }
    }
}

/// A full interactive session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionPlan {
    /// Segments, generated in order.
    pub segments: Vec<SegmentPlan>,
    /// Latent resolution `(H, W)` of every segment.
    pub latent_hw: (usize, usize),
    /// Latent frames per second (for action time windows).
    pub latent_fps: f64,
    /// Optional global seed override.
    pub seed: Option<u64>,
}

impl SessionPlan {
    /// A session of text-only segments.
    pub fn text_session(
        prompts: &[&str],
        latent_frames: usize,
        latent_hw: (usize, usize),
        latent_fps: f64,
    ) -> Self {
        Self {
            segments: prompts
                .iter()
                .map(|p| SegmentPlan::text(*p, latent_frames))
                .collect(),
            latent_hw,
            latent_fps,
            seed: None,
        }
    }
}

/// What the pipeline reports after each segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentReport {
    /// Segment index.
    pub index: usize,
    /// Latent frames generated.
    pub latent_frames: usize,
    /// Denoising steps executed.
    pub steps: usize,
    /// Video tokens of the segment.
    pub video_tokens: usize,
    /// Text tokens of the segment.
    pub text_tokens: usize,
    /// Cache working-set tokens after the commit.
    pub cache_tokens: usize,
    /// Total archived segments in the lifetime cache.
    pub archived_segments: usize,
    /// Wall-clock duration of the segment, milliseconds.
    pub duration_ms: f64,
    /// Mean |velocity| of the final step (a health indicator).
    pub final_velocity_mean: f32,
}

/// Aggregate session report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionReport {
    /// Per-segment reports.
    pub segments: Vec<SegmentReport>,
    /// Total latent frames.
    pub total_frames: usize,
    /// Total wall-clock duration, milliseconds.
    pub total_ms: f64,
    /// Cache metrics snapshot at the end.
    pub cache: crate::cache::CacheMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_plan_serde() {
        let plan = SessionPlan::text_session(&["a", "b"], 3, (16, 16), 4.0);
        let s = serde_json::to_string(&plan).unwrap();
        let back: SessionPlan = serde_json::from_str(&s).unwrap();
        assert_eq!(plan, back);
        assert_eq!(back.segments.len(), 2);
    }
}
