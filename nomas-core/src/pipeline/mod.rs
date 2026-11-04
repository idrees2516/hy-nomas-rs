//! The segmented autoregressive inference pipeline.

pub mod causal_ar;
pub mod guidance;
pub mod prompt;
pub mod solver;

pub use causal_ar::{CausalArPipeline, OnSegment};
pub use guidance::Guidance;
pub use prompt::{SegmentPlan, SegmentReport, SessionPlan, SessionReport};
pub use solver::FlowSolver;
