//! # nomas-core
//!
//! Rust port of the core of **Hy-Nomas** (Tencent Hunyuan's free-form interactive
//! video world model, 8B DiT): segmented autoregressive world generation driven by
//! multi-turn text prompts, timed *action* tokens and camera trajectories.
//!
//! The port keeps the reference architecture 1:1 where it matters algorithmically
//! and re-engineers the execution for Rust:
//!
//! * [`rope`] — 3D RoPE (temporal / height / width axes) with temporal offsets for
//!   causal segment generation, theta rescaling, packed multi-sample sequences and
//!   LaCT-style teacher-forcing positions.
//! * [`actions`] — `contained_actions` scheduling: `(start_time, end_time, text)`
//!   resolved to per-frame token visibility windows.
//! * [`camera`] — Plücker ray conditioning and per-frame camera trajectories.
//! * [`modules`] — the DiT backbone: patchifier, QK-RMSNorm, AdaLN-Zero blocks,
//!   SwiGLU MLPs, optional MoE, final adaLN layer.
//! * [`masks`] — action-aware causal attention masks: intra-segment bidirectional,
//!   inter-segment causal, chunk-grouped causality and action time windows.
//! * [`cache`] — the **lifelong KV cache** family: sliding-window attention with
//!   frame-aligned eviction, archived uniform-history sampling
//!   (`uniform` / `recent_mix_uniform` / `temporal_context`), INT8 quantized KV
//!   with per-token-per-head scales, a QVG residual-vector-quantization codec and
//!   phase-swapped CPU offload — plus interactive rollback.
//! * [`model`] — [`model::NomasDiT`], the single-stream DiT with per-segment text
//!   modulation and KV capture for cache commits.
//! * [`pipeline`] — the segmented autoregressive engine (`seg_ar` / `chunk_ar`),
//!   flow-matching Euler/distilled solvers and two-axis classifier-free guidance
//!   (text + camera).
//! * [`vae`] — the DCAE-V 32-channel latent spec and `PixelUMM` decoder interface
//!   (pluggable; a passthrough codec ships for tests and demos).
//! * [`weights`] — safetensors checkpoint loading with key diagnostics.
//!
//! Everything runs on CPU out of the box through `candle`; enable the `cuda`
//! feature to build against GPU tensors.

// The mask/codec math is written with explicit index loops on purpose: the
// token/geometry indices ARE the semantics being tested.
#![allow(clippy::needless_range_loop)]
// Several pipeline/model entry points intentionally take the full conditioning
// tuple; collapsing it would hurt API clarity.
#![allow(clippy::too_many_arguments)]

pub mod actions;
pub mod cache;
pub mod camera;
pub mod config;
pub mod error;
pub mod masks;
pub mod model;
pub mod modules;
pub mod pipeline;
pub mod rope;
pub mod tensor_utils;
pub mod vae;
pub mod weights;

pub use config::{
    CacheStorage, HistorySampling, MoeConfig, NomasConfig, SolverConfig, StoragePreset,
};
pub use error::{Error, Result};
