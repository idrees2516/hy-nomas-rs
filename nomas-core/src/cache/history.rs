//! The unified lifelong history cache.
//!
//! One implementation covers every upstream policy by combining:
//!
//! * an **archive** of completed segments (host-resident, optionally
//!   INT8/QVG-compressed) — `uniform_history_cache.py`'s CPU archive;
//! * a **working set** materialized on demand from the archive following the
//!   sampling policy — `swa_cache.py`'s recent window (frame-aligned
//!   eviction) or the archived samplers of `uniform_history_cache.py`;
//! * **rollback** by dropping the last archive record and rebuilding.

use candle_core::{Device, Tensor};

use crate::config::{CacheConfig, CacheStorage, HistorySampling};
use crate::error::Error;
use crate::Result;

use super::int8::{dequantize, quantize_kv_pair, Int8KvBlock};
use super::qvg::{QvgCodec, QvgPayload};
use super::swa::{apply_trim, plan_recent_trim, DenseRecord};
use super::uniform_history::history_frame_indices;
use super::{CacheMetrics, LifetimeKvCache, SegmentKv, SegmentLayout};

/// Compressed payload of one archived segment.
pub enum ArchivedPayload {
    /// Floating-point KV `(B, L, H, D)` in the model dtype.
    Float { k: Tensor, v: Tensor },
    /// INT8-quantized whole segment.
    Int8 { block: Int8KvBlock },
    /// QVG RVQ video tokens + (optionally) INT8 text tokens.
    Qvg {
        k: QvgPayload,
        v: QvgPayload,
        text: Option<Int8KvBlock>,
        video_tokens: usize,
    },
}

/// One archived segment (host side).
pub struct ArchivedSegment {
    /// Global segment index.
    pub segment_index: usize,
    /// Global latent-frame index of the first frame.
    pub frame_start: usize,
    /// Number of latent frames.
    pub frame_count: usize,
    /// Video tokens.
    pub image_tokens: usize,
    /// Text tokens.
    pub text_tokens: usize,
    /// First action token inside the text block, `-1` if none.
    pub action_start: isize,
    /// Compressed KV.
    pub payload: ArchivedPayload,
}

impl ArchivedSegment {
    fn dense(&self) -> DenseRecord {
        DenseRecord {
            segment_index: self.segment_index,
            frame_start: self.frame_start,
            frame_count: self.frame_count,
            image_tokens: self.image_tokens,
            text_tokens: self.text_tokens,
        }
    }
}

/// Host bytes of an archived payload.
fn payload_bytes(payload: &ArchivedPayload) -> usize {
    match payload {
        ArchivedPayload::Float { k, v: _ } => {
            let per = k.dims().iter().product::<usize>() * k.dtype().size_in_bytes();
            2 * per
        }
        ArchivedPayload::Int8 { block } => block.nbytes(),
        ArchivedPayload::Qvg { k, v, text, .. } => {
            k.nbytes() + v.nbytes() + text.as_ref().map(|t| t.nbytes()).unwrap_or(0)
        }
    }
}

/// The lifelong KV cache. See [`crate::cache`] for the design overview.
pub struct HistoryCache {
    cfg: CacheConfig,
    batch: usize,
    heads: usize,
    head_dim: usize,
    dtype: candle_core::DType,
    dev: Device,
    archive: Vec<ArchivedSegment>,
    ws_k: Option<Tensor>,
    ws_v: Option<Tensor>,
    ws_layout: Vec<SegmentLayout>,
    valid_len: usize,
    total_frames: usize,
    tokens_per_frame: usize,
    metrics: CacheMetrics,
    qvg: Option<QvgCodec>,
}

impl HistoryCache {
    /// Build a cache from configuration.
    pub fn new(
        cfg: &CacheConfig,
        batch: usize,
        heads: usize,
        head_dim: usize,
        dtype: candle_core::DType,
        dev: &Device,
    ) -> Result<Self> {
        if let CacheStorage::Qvg(q) = &cfg.storage {
            if batch != 1 {
                return Err(Error::InvalidConfig(
                    "QVG storage requires batch=1 (interactive world-model sessions)".into(),
                ));
            }
            if q.bits_per_stage == 0 || q.bits_per_stage > 8 || q.stages == 0 {
                return Err(Error::InvalidConfig("invalid QVG stages/bits".into()));
            }
        }
        if let HistorySampling::TemporalContext { gamma } = cfg.sampling {
            if !gamma.is_finite() || gamma <= 0.0 {
                return Err(Error::InvalidConfig(format!(
                    "invalid temporal_context gamma {gamma}"
                )));
            }
        }
        if cfg.capacity == Some(0) {
            return Err(Error::InvalidConfig("cache capacity must be > 0".into()));
        }
        if cfg.past_frames == 0 {
            return Err(Error::InvalidConfig("past_frames must be > 0".into()));
        }
        let qvg = match &cfg.storage {
            CacheStorage::Qvg(q) => Some(QvgCodec::new(q.clone())),
            _ => None,
        };
        Ok(Self {
            cfg: cfg.clone(),
            batch,
            heads,
            head_dim,
            dtype,
            dev: dev.clone(),
            archive: Vec::new(),
            ws_k: None,
            ws_v: None,
            ws_layout: Vec::new(),
            valid_len: 0,
            total_frames: 0,
            tokens_per_frame: 0,
            metrics: CacheMetrics::default(),
            qvg,
        })
    }

    /// The cache's device.
    pub fn device(&self) -> &Device {
        &self.dev
    }

    fn capacity_or_max(&self) -> usize {
        self.cfg.capacity.unwrap_or(usize::MAX)
    }

    /// Validate the incoming segment and learn/verify the frame geometry.
    fn validate_segment(&mut self, seg: &SegmentKv) -> Result<()> {
        if seg.frame_count == 0 || seg.image_tokens == 0 {
            return Err(Error::Geometry(
                "segment must contain at least one frame".into(),
            ));
        }
        let tpf = seg.image_tokens / seg.frame_count;
        if !seg.image_tokens.is_multiple_of(seg.frame_count) || tpf == 0 {
            return Err(Error::Geometry(format!(
                "image_tokens {} not divisible by frame_count {}",
                seg.image_tokens, seg.frame_count
            )));
        }
        if self.tokens_per_frame == 0 {
            self.tokens_per_frame = tpf;
        } else if self.tokens_per_frame != tpf {
            return Err(Error::Geometry(format!(
                "tokens_per_frame changed mid-session: {} -> {}",
                self.tokens_per_frame, tpf
            )));
        }
        if let Some(last) = self.archive.last() {
            if seg.segment_index <= last.segment_index {
                return Err(Error::CacheState(
                    "history requires one chronological write per segment".into(),
                ));
            }
            if seg.frame_start != self.total_frames {
                return Err(Error::Geometry(format!(
                    "segment frame_start {} does not continue the timeline at {}",
                    seg.frame_start, self.total_frames
                )));
            }
        } else if seg.frame_start != 0 && self.total_frames == 0 {
            return Err(Error::Geometry(format!(
                "first segment must start at frame 0, got {}",
                seg.frame_start
            )));
        }
        if !(-1..=seg.text_tokens as isize).contains(&seg.action_start) {
            return Err(Error::Geometry(format!(
                "action_start {} outside text block of {}",
                seg.action_start, seg.text_tokens
            )));
        }
        for (name, t) in [("k", &seg.k), ("v", &seg.v)] {
            let expected = [self.batch, seg.len(), self.heads, self.head_dim];
            if t.dims() != expected {
                return Err(Error::Geometry(format!(
                    "segment {name} shape {:?} != {:?}",
                    t.dims(),
                    expected
                )));
            }
            if !matches!(
                t.dtype(),
                candle_core::DType::F32 | candle_core::DType::BF16 | candle_core::DType::F16
            ) {
                return Err(Error::Geometry(format!(
                    "segment {name} must be float, got {:?}",
                    t.dtype()
                )));
            }
        }
        if seg.k.device().location() != self.dev.location() {
            return Err(Error::Geometry(
                "segment KV must live on the cache device".into(),
            ));
        }
        Ok(())
    }

    /// Archive the segment under the configured storage format.
    fn archive_segment(&mut self, seg: &SegmentKv) -> Result<()> {
        let payload = match &self.cfg.storage {
            CacheStorage::Floating => {
                let k = seg.k.to_dtype(self.dtype)?.contiguous()?;
                let v = seg.v.to_dtype(self.dtype)?.contiguous()?;
                ArchivedPayload::Float { k, v }
            }
            CacheStorage::Int8 => {
                let k = seg.k.to_dtype(candle_core::DType::BF16)?;
                let v = seg.v.to_dtype(candle_core::DType::BF16)?;
                ArchivedPayload::Int8 {
                    block: quantize_kv_pair(&k, &v)?,
                }
            }
            CacheStorage::Qvg(_) => {
                let codec = self.qvg.as_mut().expect("qvg codec");
                if !codec.is_fitted() {
                    // Fit once from the first segment; frozen afterwards.
                    let kf = seg.k.to_dtype(candle_core::DType::F32)?;
                    let vf = seg.v.to_dtype(candle_core::DType::F32)?;
                    codec.fit(&kf, &vf)?;
                }
                let kf = seg.k.to_dtype(candle_core::DType::F32)?;
                let vf = seg.v.to_dtype(candle_core::DType::F32)?;
                let video = seg.image_tokens;
                let kq = codec.encode(0, &kf, video)?;
                let vq = codec.encode(1, &vf, video)?;
                let text = if seg.text_tokens > 0 {
                    let kt = kf
                        .narrow(1usize, video, seg.text_tokens)?
                        .to_dtype(candle_core::DType::BF16)?;
                    let vt = vf
                        .narrow(1usize, video, seg.text_tokens)?
                        .to_dtype(candle_core::DType::BF16)?;
                    Some(quantize_kv_pair(&kt, &vt)?)
                } else {
                    None
                };
                ArchivedPayload::Qvg {
                    k: kq,
                    v: vq,
                    text,
                    video_tokens: video,
                }
            }
        };
        let nbytes = payload_bytes(&payload);
        self.metrics.encode_calls += match &self.cfg.storage {
            CacheStorage::Floating => 0,
            _ => 1,
        };
        self.archive.push(ArchivedSegment {
            segment_index: seg.segment_index,
            frame_start: seg.frame_start,
            frame_count: seg.frame_count,
            image_tokens: seg.image_tokens,
            text_tokens: seg.text_tokens,
            action_start: seg.action_start,
            payload,
        });
        self.metrics.archived_segments += 1;
        self.metrics.archived_bytes += nbytes;
        self.metrics.d2h_bytes += nbytes;
        Ok(())
    }

    /// Decode one archived segment back to model-dtype `(k, v)`.
    fn decode_segment(&self, seg: &ArchivedSegment) -> Result<(Tensor, Tensor)> {
        match &seg.payload {
            ArchivedPayload::Float { k, v } => Ok((k.clone(), v.clone())),
            ArchivedPayload::Int8 { block } => {
                let len = seg.image_tokens + seg.text_tokens;
                let k = dequantize(&block.k, &block.k_scale, len, self.dtype)?;
                let v = dequantize(&block.v, &block.v_scale, len, self.dtype)?;
                Ok((k, v))
            }
            ArchivedPayload::Qvg {
                k,
                v,
                text,
                video_tokens,
            } => {
                let codec = self.qvg.as_ref().expect("qvg codec");
                let mut k = codec.decode(0, k, &self.dev)?;
                let mut v = codec.decode(1, v, &self.dev)?;
                if let Some(tb) = text {
                    let tk =
                        dequantize(&tb.k, &tb.k_scale, seg.text_tokens, candle_core::DType::F32)?;
                    let tv =
                        dequantize(&tb.v, &tb.v_scale, seg.text_tokens, candle_core::DType::F32)?;
                    k = Tensor::cat(&[&k, &tk], 1)?;
                    v = Tensor::cat(&[&v, &tv], 1)?;
                }
                let _ = video_tokens;
                Ok((k.to_dtype(self.dtype)?, v.to_dtype(self.dtype)?))
            }
        }
    }

    /// Rebuild the working set from the archive under the sampling policy.
    fn rebuild(&mut self) -> Result<()> {
        let tpf = self.tokens_per_frame.max(1);
        // 1) decide, per archived record, which frames survive.
        let selection: Vec<(usize, Vec<usize>)> = match self.cfg.sampling {
            HistorySampling::Recent => {
                let records: Vec<DenseRecord> = self.archive.iter().map(|a| a.dense()).collect();
                let plan = plan_recent_trim(&records, self.capacity_or_max(), tpf);
                self.metrics.evicted_tokens += plan.evicted_tokens;
                let kept = apply_trim(&records, &plan, tpf);
                kept.into_iter()
                    .map(|r| {
                        let idx = self
                            .archive
                            .iter()
                            .position(|a| a.segment_index == r.segment_index)
                            .unwrap_or(0);
                        (idx, (0..r.frame_count).collect())
                    })
                    .collect()
            }
            HistorySampling::Uniform
            | HistorySampling::RecentMixUniform
            | HistorySampling::TemporalContext { .. } => {
                let ids = history_frame_indices(
                    self.total_frames,
                    self.cfg.past_frames,
                    &self.cfg.sampling,
                )?;
                self.archive
                    .iter()
                    .enumerate()
                    .map(|(idx, a)| {
                        let sel: Vec<usize> = ids
                            .iter()
                            .filter(|&&f| a.frame_start <= f && f < a.frame_start + a.frame_count)
                            .map(|&f| f - a.frame_start)
                            .collect();
                        (idx, sel)
                    })
                    .filter(|(_, sel)| !sel.is_empty())
                    .collect()
            }
        };

        // 2) pack: [selected video frames | text block] per segment.
        let mut layout: Vec<SegmentLayout> = Vec::with_capacity(selection.len());
        let mut k_parts: Vec<Tensor> = Vec::new();
        let mut v_parts: Vec<Tensor> = Vec::new();
        let mut cursor = 0usize;
        for (idx, sel) in &selection {
            let a = &self.archive[*idx];
            let mut indices: Vec<usize> = Vec::with_capacity(sel.len() * tpf + a.text_tokens);
            for &f in sel {
                let base = f * tpf;
                indices.extend(base..base + tpf);
            }
            indices.extend(a.image_tokens..a.image_tokens + a.text_tokens);
            let (k, v) = self.decode_segment(a)?;
            let idx_t = crate::tensor_utils::idx_tensor(&indices, &self.dev)?;
            let k = k.index_select(&idx_t, 1)?;
            let v = v.index_select(&idx_t, 1)?;
            let video_end = cursor + sel.len() * tpf;
            let text_end = video_end + a.text_tokens;
            layout.push(SegmentLayout {
                segment_index: a.segment_index,
                video_start: cursor,
                video_end,
                text_start: video_end,
                text_end,
                action_start: a.action_start,
            });
            cursor = text_end;
            k_parts.push(k);
            v_parts.push(v);
        }
        if cursor > self.capacity_or_max() {
            return Err(Error::Geometry(format!(
                "history needs {cursor} tokens; capacity is {}",
                self.capacity_or_max()
            )));
        }
        let (k, v) = if k_parts.is_empty() {
            (
                Tensor::zeros(
                    (self.batch, 0usize, self.heads, self.head_dim),
                    self.dtype,
                    &self.dev,
                )?,
                Tensor::zeros(
                    (self.batch, 0usize, self.heads, self.head_dim),
                    self.dtype,
                    &self.dev,
                )?,
            )
        } else {
            (Tensor::cat(&k_parts, 1)?, Tensor::cat(&v_parts, 1)?)
        };
        self.metrics.h2d_bytes +=
            cursor * self.batch * self.heads * self.head_dim * self.dtype.size_in_bytes() * 2;
        self.ws_k = Some(k);
        self.ws_v = Some(v);
        self.ws_layout = layout;
        self.valid_len = cursor;
        self.metrics.rebuilds += 1;
        Ok(())
    }
}

impl LifetimeKvCache for HistoryCache {
    fn storage_format(&self) -> CacheStorage {
        self.cfg.storage.clone()
    }

    fn sampling(&self) -> HistorySampling {
        self.cfg.sampling
    }

    fn commit_segment(&mut self, seg: SegmentKv) -> Result<()> {
        self.validate_segment(&seg)?;
        self.archive_segment(&seg)?;
        self.total_frames += seg.frame_count;
        self.rebuild()
    }

    fn working_set(&self) -> Result<(Tensor, Tensor)> {
        match (&self.ws_k, &self.ws_v) {
            (Some(k), Some(v)) => Ok((k.clone(), v.clone())),
            _ => Ok((
                Tensor::zeros(
                    (self.batch, 0usize, self.heads, self.head_dim),
                    self.dtype,
                    &self.dev,
                )?,
                Tensor::zeros(
                    (self.batch, 0usize, self.heads, self.head_dim),
                    self.dtype,
                    &self.dev,
                )?,
            )),
        }
    }

    fn working_layout(&self) -> Vec<SegmentLayout> {
        self.ws_layout.clone()
    }

    fn valid_len(&self) -> usize {
        self.valid_len
    }

    fn capacity(&self) -> usize {
        self.capacity_or_max()
    }

    fn n_segments(&self) -> usize {
        self.archive.len()
    }

    fn total_frames(&self) -> usize {
        self.total_frames
    }

    fn tokens_per_frame(&self) -> usize {
        self.tokens_per_frame
    }

    fn rollback_last_segment(&mut self) -> Result<()> {
        if matches!(self.cfg.sampling, HistorySampling::Recent) {
            return Err(Error::CacheState(
                "recent-mode SWA does not support streaming rollback (use an archived sampling)"
                    .into(),
            ));
        }
        let last = self
            .archive
            .pop()
            .ok_or_else(|| Error::CacheState("nothing to roll back".into()))?;
        if last.frame_start + last.frame_count != self.total_frames {
            return Err(Error::CacheState(
                "archive does not match the frame timeline".into(),
            ));
        }
        self.total_frames -= last.frame_count;
        self.metrics.archived_segments = self.metrics.archived_segments.saturating_sub(1);
        self.rebuild()?;
        self.metrics.rollbacks += 1;
        Ok(())
    }

    fn metrics(&self) -> CacheMetrics {
        self.metrics
    }
}

impl HistoryCache {
    // ---- phase-swap hooks (used by `phase_swap::PhaseSwappedCache`) ----

    /// Drop the device-side working set (the phase-swap wrapper parked it).
    pub(crate) fn release_working_set(&mut self) {
        self.ws_k = None;
        self.ws_v = None;
        self.ws_layout = Vec::new();
        self.valid_len = 0;
    }

    /// Restore a parked working set.
    pub(crate) fn restore_working_set(
        &mut self,
        k: Tensor,
        v: Tensor,
        layout: Vec<SegmentLayout>,
        valid_len: usize,
    ) {
        self.ws_k = Some(k);
        self.ws_v = Some(v);
        self.ws_layout = layout;
        self.valid_len = valid_len;
    }

    /// Account host→device bytes moved by an unpark.
    pub(crate) fn note_h2d(&mut self, bytes: usize) {
        self.metrics.h2d_bytes += bytes;
    }

    /// Account device→host bytes moved by a park.
    pub(crate) fn note_d2h(&mut self, bytes: usize) {
        self.metrics.d2h_bytes += bytes;
    }
}
