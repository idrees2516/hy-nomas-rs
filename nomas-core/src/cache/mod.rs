//! The **lifelong KV cache** family — Rust port of
//! `hymm/models/dit/modules/{kv_cache_storage,swa_cache,uniform_history_cache,
//! int8_kv_cache,qvg_kv_cache}.py`.
//!
//! # Model of the world
//!
//! Every DiT layer owns one cache instance ([`KvBank`] fans commits out to all
//! layers). A *segment* of interactive generation contributes:
//!
//! ```text
//! [ video tokens (frame_count x tokens_per_frame) | text tokens (incl. actions) ]
//! ```
//!
//! with `action_start` locating the action tokens inside the text block — the
//! same packing the upstream `ArchivedSegment` uses. Keys and values are
//! stored **already rotated**: they were produced with the RoPE positions of
//! their original frames, so history rebuilds never re-rotate anything.
//!
//! # Policies
//!
//! * [`HistorySampling::Recent`] — sliding window with **frame-aligned
//!   eviction** (`swa_cache.py`): the newest `capacity` tokens are kept and
//!   cuts always land on frame boundaries; a segment's text is dropped once
//!   its video is fully evicted.
//! * [`HistorySampling::Uniform`], [`HistorySampling::RecentMixUniform`],
//!   [`HistorySampling::TemporalContext`] — archived history sampling
//!   (`uniform_history_cache.py`): every completed segment is archived on the
//!   host (optionally under INT8/QVG compression) and the working set is
//!   rebuilt from `past_frames` sampled frames plus the text blocks of every
//!   represented segment. The full archive lets future segments reselect
//!   frames that are not currently resident.
//! * Storage formats: floating point, **INT8** with per-token-per-head FP32
//!   scales, and **QVG** residual vector quantization (see [`qvg`]).
//! * Interactive **rollback**: drop the last segment and rebuild the working
//!   set from the remaining archive without re-encoding any KV.
//!
//! # A note on tensors
//!
//! Upstream preallocates a `(B, capacity, H, D)` arena and writes into it in
//! place. `candle` tensors are immutable, so the working set here is
//! *materialized* by concatenating archived slices on every rebuild; the
//! token-geometry bookkeeping (`SegmentLayout`) is preserved exactly. With the
//! `cuda` feature the phase-swap wrapper performs real device moves.

pub mod history;
pub mod int8;
pub mod phase_swap;
pub mod qvg;
pub mod swa;
pub mod uniform_history;

use serde::{Deserialize, Serialize};

use crate::config::{CacheConfig, CacheStorage, HistorySampling};
use crate::Result;
pub use history::HistoryCache;

/// Token geometry of one packed segment inside a working set — mirrors the
/// upstream `swa_uniform_layout` tuples
/// `(segment, video_start, video_end, text_start, text_end, action_start)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentLayout {
    /// Global segment index (generation order).
    pub segment_index: usize,
    /// First video token of the segment inside the packed working set.
    pub video_start: usize,
    /// One-past-the-last video token.
    pub video_end: usize,
    /// First text token (always `== video_end` in the packed layout).
    pub text_start: usize,
    /// One-past-the-last text token.
    pub text_end: usize,
    /// Offset of the first action token inside the text block, or `-1`.
    pub action_start: isize,
}

impl SegmentLayout {
    /// Total tokens covered by this segment.
    pub fn len(&self) -> usize {
        (self.video_end - self.video_start) + (self.text_end - self.text_start)
    }

    /// Whether the segment occupies any token at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `(video_start, video_end)` range.
    pub fn video_range(&self) -> (usize, usize) {
        (self.video_start, self.video_end)
    }

    /// `(text_start, text_end)` range.
    pub fn text_range(&self) -> (usize, usize) {
        (self.text_start, self.text_end)
    }
}

/// A completed segment's KV payload, ready to commit into a cache.
#[derive(Debug, Clone)]
pub struct SegmentKv {
    /// Global segment index.
    pub segment_index: usize,
    /// Global latent-frame index of the segment's first frame.
    pub frame_start: usize,
    /// Number of latent frames in the segment.
    pub frame_count: usize,
    /// Video token count (`frame_count * tokens_per_frame`).
    pub image_tokens: usize,
    /// Text token count (prompt + actions + separator).
    pub text_tokens: usize,
    /// Offset of the first action token inside the text block, `-1` if none.
    pub action_start: isize,
    /// Keys `(B, image_tokens + text_tokens, H, D)` — **already rotated**.
    pub k: candle_core::Tensor,
    /// Values, same layout as `k`.
    pub v: candle_core::Tensor,
}

impl SegmentKv {
    /// Total token count of the packed segment.
    pub fn len(&self) -> usize {
        self.image_tokens + self.text_tokens
    }

    /// Whether the segment carries no tokens.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Counters mirroring `kv_codec_stats` / `inference_metrics` upstream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Completed segments archived so far.
    pub archived_segments: usize,
    /// Host bytes held by the archive (payload + scales).
    pub archived_bytes: usize,
    /// INT8/QVG encode invocations.
    pub encode_calls: usize,
    /// Device→host bytes moved by commits / phase parking.
    pub d2h_bytes: usize,
    /// Host→device bytes moved by working-set rebuilds / unparking.
    pub h2d_bytes: usize,
    /// Working-set rebuilds (commits + rollbacks).
    pub rebuilds: usize,
    /// Tokens evicted by the recent/SWA policy.
    pub evicted_tokens: usize,
    /// Successful rollbacks.
    pub rollbacks: usize,
}

/// The interface shared by every lifelong-cache implementation.
pub trait LifetimeKvCache {
    /// Storage format in use.
    fn storage_format(&self) -> CacheStorage;

    /// History sampling policy in use.
    fn sampling(&self) -> HistorySampling;

    /// Commit one completed segment (archives it, then rebuilds the working
    /// set according to the sampling policy).
    fn commit_segment(&mut self, seg: SegmentKv) -> Result<()>;

    /// Materialize the current working set as `(k, v)` tensors of the model
    /// dtype, shape `(B, valid_len, H, D)`. INT8/QVG payloads are decoded on
    /// the fly; already-rotated keys are returned untouched.
    fn working_set(&self) -> Result<(candle_core::Tensor, candle_core::Tensor)>;

    /// Token layout of the working set.
    fn working_layout(&self) -> Vec<SegmentLayout>;

    /// Number of valid tokens in the working set.
    fn valid_len(&self) -> usize;

    /// Token capacity of the working set (`usize::MAX` when unbounded).
    fn capacity(&self) -> usize;

    /// Number of archived segments (including evicted ones).
    fn n_segments(&self) -> usize;

    /// Total latent frames generated over the session.
    fn total_frames(&self) -> usize;

    /// Tokens per latent frame (constant for a fixed resolution).
    fn tokens_per_frame(&self) -> usize;

    /// Drop the last committed segment and rebuild the working set from the
    /// remaining archive. Requires an archived (non-`recent`) policy.
    fn rollback_last_segment(&mut self) -> Result<()>;

    /// Snapshot of the cache counters.
    fn metrics(&self) -> CacheMetrics;

    /// Park the working set on the host (phase-swap offload). Default no-op.
    fn park(&mut self) -> Result<()> {
        Ok(())
    }

    /// Unpark the working set onto the device. Default no-op.
    fn unpark(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Build the concrete cache described by `cfg` — the factory used by
/// [`crate::pipeline`]. Returns the phase-swap wrapper when configured.
pub fn build_cache(
    cfg: &CacheConfig,
    batch: usize,
    heads: usize,
    head_dim: usize,
    dtype: candle_core::DType,
    dev: &candle_core::Device,
) -> Result<Box<dyn LifetimeKvCache>> {
    let inner = HistoryCache::new(cfg, batch, heads, head_dim, dtype, dev)?;
    if cfg.phase_swap {
        Ok(Box::new(phase_swap::PhaseSwappedCache::new(inner)))
    } else {
        Ok(Box::new(inner))
    }
}

/// One [`LifetimeKvCache`] per DiT layer, fanning out commits and rollbacks.
pub struct KvBank {
    layers: Vec<Box<dyn LifetimeKvCache>>,
}

impl KvBank {
    /// Build a bank of `n_layers` caches.
    pub fn new(
        cfg: &CacheConfig,
        n_layers: usize,
        batch: usize,
        heads: usize,
        head_dim: usize,
        dtype: candle_core::DType,
        dev: &candle_core::Device,
    ) -> Result<Self> {
        let layers = (0..n_layers)
            .map(|_| build_cache(cfg, batch, heads, head_dim, dtype, dev))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    /// Number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the bank is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Commit one segment's per-layer KV (index-aligned with the layers).
    pub fn commit(&mut self, per_layer: Vec<SegmentKv>) -> Result<()> {
        crate::ensure!(
            per_layer.len() == self.layers.len(),
            "KvBank::commit got {} segments for {} layers",
            per_layer.len(),
            self.layers.len()
        );
        for (cache, seg) in self.layers.iter_mut().zip(per_layer) {
            cache.commit_segment(seg)?;
        }
        Ok(())
    }

    /// Access the cache of layer `i`.
    pub fn layer(&self, i: usize) -> &dyn LifetimeKvCache {
        self.layers[i].as_ref()
    }

    /// Rollback every layer.
    pub fn rollback(&mut self) -> Result<()> {
        for cache in self.layers.iter_mut() {
            cache.rollback_last_segment()?;
        }
        Ok(())
    }

    /// Park every layer (phase-swap offload).
    pub fn park_all(&mut self) -> Result<()> {
        for cache in self.layers.iter_mut() {
            cache.park()?;
        }
        Ok(())
    }

    /// Unpark every layer.
    pub fn unpark_all(&mut self) -> Result<()> {
        for cache in self.layers.iter_mut() {
            cache.unpark()?;
        }
        Ok(())
    }

    /// Borrow all layers as trait objects (model forward input).
    pub fn as_refs(&self) -> Vec<&dyn LifetimeKvCache> {
        self.layers.iter().map(|c| c.as_ref()).collect()
    }

    /// Aggregate metrics over layers.
    pub fn metrics(&self) -> CacheMetrics {
        let mut m = CacheMetrics::default();
        for cache in &self.layers {
            let c = cache.metrics();
            m.archived_segments = m.archived_segments.max(c.archived_segments);
            m.archived_bytes += c.archived_bytes;
            m.encode_calls += c.encode_calls;
            m.d2h_bytes += c.d2h_bytes;
            m.h2d_bytes += c.h2d_bytes;
            m.rebuilds = m.rebuilds.max(c.rebuilds);
            m.evicted_tokens = m.evicted_tokens.max(c.evicted_tokens);
            m.rollbacks = m.rollbacks.max(c.rollbacks);
        }
        m
    }
}
