//! The segmented autoregressive engine — `hymm/pipeline/causal_ar.py`
//! (`CPTCausalInferencePipeline`) in Rust.
//!
//! # Rollout modes (upstream `inference_chunk_size`)
//!
//! * `1` — **seg_ar, full bidirectional** (default): the whole segment
//!   denoises jointly; intra-segment attention is bidirectional (modulo
//!   action windows).
//! * `0` — **seg_ar, frame-causal**: the segment denoises jointly but frames
//!   only attend to earlier frames of the same segment.
//! * `k > 1` — **chunk_ar**: the segment is split into chunks of `k` latent
//!   frames, denoised left-to-right; each chunk attends to the lifelong
//!   history plus the *already denoised* (clean) chunks of its segment.
//!
//! # Per-turn flow
//!
//! ```text
//! encode text (prompt + action descriptions) ──► layout with action windows
//! sample noise ──► Euler/distilled denoise loop with dual CFG
//!              └─► capture per-layer KV at `cache_at_step`
//! commit SegmentKv into every layer's lifelong cache
//! (optional) park cache ──► decode segment ──► unpark cache
//! ```

use std::time::Instant;

use candle_core::{Device, Tensor};

use crate::actions::TextBlockLayout;
use crate::cache::{KvBank, SegmentKv};
use crate::camera::{trajectory_embeddings, CameraTrajectory, Intrinsics};
use crate::config::CacheConfig;
use crate::error::{Error, Result};
use crate::masks::IntraCausality;
use crate::model::{HistoryView, LayerKv, NomasDiT};
use crate::modules::text_embed::TextEncoder;
use crate::pipeline::guidance::Guidance;
use crate::pipeline::prompt::{SegmentPlan, SegmentReport, SessionPlan, SessionReport};
use crate::pipeline::solver::FlowSolver;
use crate::vae::VideoCodec;

/// Streaming callback: `(segment_report, decoded_frames)` after each segment.
pub type OnSegment<'a> = dyn FnMut(&SegmentReport, &Tensor) + 'a;

/// The interactive world-generation engine.
pub struct CausalArPipeline {
    dit: NomasDiT,
    text_encoder: Box<dyn TextEncoder>,
    codec: Box<dyn VideoCodec>,
    solver: FlowSolver,
    guidance: Guidance,
    cache_cfg: CacheConfig,
    chunk_mode: usize,
    dtype: candle_core::DType,
    dev: Device,
    // session state
    bank: Option<KvBank>,
    frame_offset: usize,
    segment_index: usize,
    reports: Vec<SegmentReport>,
    started: Option<Instant>,
}

impl CausalArPipeline {
    /// Assemble a pipeline from its parts.
    pub fn new(
        dit: NomasDiT,
        text_encoder: Box<dyn TextEncoder>,
        codec: Box<dyn VideoCodec>,
        solver: FlowSolver,
        guidance: Guidance,
        chunk_mode: usize,
    ) -> Result<Self> {
        let cfg = dit.config().clone();
        let dev = dit.device().clone();
        let dtype = dit.dtype();
        Ok(Self {
            cache_cfg: cfg.cache.clone(),
            dtype,
            dev,
            dit,
            text_encoder,
            codec,
            solver,
            guidance,
            chunk_mode,
            bank: None,
            frame_offset: 0,
            segment_index: 0,
            reports: Vec::new(),
            started: None,
        })
    }

    /// Device to run on (the model's tensors already live there).
    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Access the DiT.
    pub fn dit(&self) -> &NomasDiT {
        &self.dit
    }

    /// Debug/inspection snapshot: `(archived_segments, total_frames,
    /// working_set_tokens)`.
    pub fn debug_cache_snapshot(&self) -> (usize, usize, usize) {
        match &self.bank {
            Some(bank) => {
                let l = bank.layer(0);
                (l.n_segments(), l.total_frames(), l.valid_len())
            }
            None => (0, 0, 0),
        }
    }

    /// Begin an interactive session (lazily builds the per-layer caches).
    pub fn begin(&mut self, latent_hw: (usize, usize)) -> Result<()> {
        let cfg = self.dit.config().clone();
        crate::ensure!(
            latent_hw.0.is_multiple_of(cfg.patch_size.1)
                && latent_hw.1.is_multiple_of(cfg.patch_size.2),
            "latent resolution {:?} not divisible by patch {:?}",
            latent_hw,
            cfg.patch_size
        );
        let dev = self.dev.clone();
        self.bank = Some(KvBank::new(
            &self.cache_cfg,
            cfg.n_layers,
            1,
            cfg.n_heads,
            cfg.head_dim(),
            self.dtype,
            &dev,
        )?);
        self.frame_offset = 0;
        self.segment_index = 0;
        self.reports.clear();
        self.started = Some(Instant::now());
        Ok(())
    }

    /// Generate one interactive turn. Returns `(report, clean_latent)`.
    pub fn turn(
        &mut self,
        plan: &SegmentPlan,
        latent_fps: f64,
        latent_hw: (usize, usize),
    ) -> Result<(SegmentReport, Tensor)> {
        let bank = self
            .bank
            .as_mut()
            .ok_or_else(|| Error::CacheState("session not begun; call begin() first".into()))?;
        let t0 = Instant::now();
        let cfg = self.dit.config().clone();
        let dev = self.dev.clone();

        // ---- text conditioning ------------------------------------------
        let base = self.text_encoder.encode(&plan.prompt)?;
        let mut act_feats = Vec::new();
        let mut act_tokens = Vec::new();
        for a in &plan.actions {
            let f = self.text_encoder.encode(&a.description)?;
            act_tokens.push(f.dims()[1]);
            act_feats.push(f);
        }
        let mut text = base.clone();
        if !act_feats.is_empty() {
            let parts: Vec<Tensor> = std::iter::once(base).chain(act_feats).collect();
            text = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?;
        }
        let base_tokens = text.dims()[1] - act_tokens.iter().sum::<usize>();

        // ---- camera ------------------------------------------------------
        let grid_h = latent_hw.0 / cfg.patch_size.1;
        let grid_w = latent_hw.1 / cfg.patch_size.2;
        let cam_feats = plan
            .camera
            .as_ref()
            .map(|traj| trajectory_embeddings(traj, grid_h.clamp(1, 4), grid_w.clamp(1, 4), &dev))
            .transpose()?;

        // ---- chunking ------------------------------------------------------
        let frames = plan.latent_frames;
        let causality = IntraCausality::from_chunk_size(self.chunk_mode);
        // chunk size in frames for chunk_ar: k (last chunk may be shorter)
        let chunk_len = match self.chunk_mode {
            0 | 1 => frames,
            k => k.max(1),
        };
        let mut chunk_starts = Vec::new();
        let mut f = 0usize;
        while f < frames {
            chunk_starts.push(f);
            f += chunk_len;
        }

        let capture_step = if self.cache_cfg.cache_at_step < 0 {
            self.solver.steps.saturating_sub(1)
        } else {
            (self.cache_cfg.cache_at_step as usize).min(self.solver.steps - 1)
        };
        let sigma = self.cache_cfg.cache_noise_sigma;

        // full captured KV per chunk: (video_token_count, per-layer KV)
        let mut chunk_kvs: Vec<(usize, Vec<LayerKv>)> = Vec::new();
        let mut final_latent: Option<Tensor> = None;
        let mut final_velocity_mean = 0f32;
        let mut steps_run = 0usize;

        for &chunk_start in &chunk_starts {
            let chunk_frames_n = chunk_len.min(frames - chunk_start);
            let shape = (
                1usize,
                cfg.in_channels,
                chunk_frames_n,
                latent_hw.0,
                latent_hw.1,
            );
            let mut x = self.solver.initial_noise(shape, &dev)?;
            x = x.to_dtype(self.dtype)?;
            // chunk-local action windows (offset the global frame axis)
            let layout = TextBlockLayout::new(
                base_tokens,
                &plan.actions,
                &act_tokens,
                chunk_frames_n,
                latent_fps,
                cfg.text_separator,
            );
            let chunk_cam = cam_feats.as_ref().map(|c| {
                c.narrow(0usize, chunk_start, chunk_frames_n)
                    .unwrap()
                    .contiguous()
                    .unwrap()
            });

            // video-only pending KV for this chunk's history (chunk_ar)
            let (pending, pending_chunk_tokens) = build_pending(&chunk_kvs)?;
            let cache_refs = bank.as_refs();
            let ts = self.solver.timesteps();
            let mut captured: Option<Vec<LayerKv>> = None;
            for (si, (&t, &tn)) in ts[..ts.len() - 1].iter().zip(ts[1..].iter()).enumerate() {
                let is_capture = si == capture_step;
                let view = HistoryView {
                    caches: &cache_refs,
                    pending: &pending,
                    pending_chunk_tokens: &pending_chunk_tokens,
                };
                // observation noise for the capture pass (cache_noise_sigma)
                let x_obs = if is_capture && sigma > 0.0 {
                    let noise = crate::tensor_utils::deterministic_randn(
                        x.dims().to_vec(),
                        self.solver.seed ^ 0xCAC4E,
                        &dev,
                    )?
                    .to_dtype(self.dtype)?;
                    (x.clone() + noise.affine(sigma, 0.0)?)?
                } else {
                    x.clone()
                };
                let out = self.dit.forward_segment(
                    &x_obs,
                    t,
                    &text,
                    &layout,
                    chunk_cam.as_ref(),
                    self.frame_offset + chunk_start,
                    causality,
                    &view,
                    is_capture,
                )?;
                let v_cond = out.velocity;
                if is_capture {
                    captured = out.kv;
                }

                let v = if self.guidance.is_raw() {
                    v_cond
                } else {
                    // text-only pass (default camera)
                    let view = HistoryView {
                        caches: &cache_refs,
                        pending: &pending,
                        pending_chunk_tokens: &pending_chunk_tokens,
                    };
                    let tnc = self.dit.forward_segment(
                        &x,
                        t,
                        &text,
                        &layout,
                        None,
                        self.frame_offset + chunk_start,
                        causality,
                        &view,
                        false,
                    )?;
                    let uc = if self.guidance.text != 1.0 {
                        let view = HistoryView {
                            caches: &cache_refs,
                            pending: &pending,
                            pending_chunk_tokens: &pending_chunk_tokens,
                        };
                        // Token-count-preserving null ablation: the null
                        // features tile over the conditional prompt's
                        // positions so the mask/rope geometry is identical.
                        let null_text = {
                            let null = self.text_encoder.encode_null()?;
                            let target = text.dims()[1];
                            let ln = null.dims()[1];
                            if ln >= target {
                                null.narrow(1usize, 0, target)?.contiguous()?
                            } else {
                                let reps = target.div_ceil(ln);
                                let parts: Vec<Tensor> = (0..reps).map(|_| null.clone()).collect();
                                Tensor::cat(&parts.iter().collect::<Vec<_>>(), 1)?
                                    .narrow(1usize, 0, target)?
                                    .contiguous()?
                            }
                        };
                        Some(self.dit.forward_segment(
                            &x,
                            t,
                            &null_text,
                            &layout,
                            None,
                            self.frame_offset + chunk_start,
                            causality,
                            &view,
                            false,
                        )?)
                    } else {
                        None
                    };
                    self.guidance.combine(
                        &v_cond,
                        Some(&tnc.velocity),
                        uc.as_ref().map(|u| &u.velocity),
                    )?
                };
                final_velocity_mean = v
                    .to_dtype(candle_core::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|x| x.abs())
                    .sum::<f32>()
                    / v.dims().iter().product::<usize>() as f32;
                x = self.solver.step(&x, &v, t, tn, si, &dev)?;
                steps_run += 1;
            }
            final_latent = Some(x.clone());
            drop(cache_refs);

            // stash the chunk's full captured KV [video | text]
            if let Some(kv) = captured {
                let video_tokens = (chunk_frames_n / cfg.patch_size.0) * grid_h * grid_w;
                chunk_kvs.push((video_tokens, kv));
            }
        }

        let latent = final_latent.ok_or_else(|| Error::Msg("no chunks generated".into()))?;

        // ---- commit the whole segment into the lifelong cache ------------
        let video_tokens_total = (frames / cfg.patch_size.0) * grid_h * grid_w;
        let text_tokens = text.dims()[1] + usize::from(cfg.text_separator);
        let action_start = if plan.actions.is_empty() {
            -1
        } else {
            base_tokens as isize
        };
        crate::ensure!(
            !chunk_kvs.is_empty(),
            "no KV captured; check cache_at_step and solver steps"
        );
        let n_layers = chunk_kvs[0].1.len();
        let last = chunk_kvs.last().expect("non-empty");
        let last_video = last.0;
        let last_kv = &last.1;
        let mut per_layer = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let mut k_parts: Vec<Tensor> = Vec::with_capacity(chunk_kvs.len() + 1);
            let mut v_parts: Vec<Tensor> = Vec::with_capacity(chunk_kvs.len() + 1);
            for (vt, kv) in &chunk_kvs {
                k_parts.push(kv[li].k.narrow(1usize, 0, *vt)?.contiguous()?);
                v_parts.push(kv[li].v.narrow(1usize, 0, *vt)?.contiguous()?);
            }
            // text tokens come from the last chunk's capture (freshest state)
            let k_text = last_kv[li]
                .k
                .narrow(1usize, last_video, last_kv[li].k.dims()[1] - last_video)?
                .contiguous()?;
            let v_text = last_kv[li]
                .v
                .narrow(1usize, last_video, last_kv[li].v.dims()[1] - last_video)?
                .contiguous()?;
            k_parts.push(k_text);
            v_parts.push(v_text);
            let k = Tensor::cat(&k_parts.iter().collect::<Vec<_>>(), 1)?;
            let v = Tensor::cat(&v_parts.iter().collect::<Vec<_>>(), 1)?;
            per_layer.push((k, v));
        }

        let seg_index = self.segment_index;
        let frame_start = self.frame_offset;
        let mut seg_kvs = Vec::with_capacity(per_layer.len());
        for (k, v) in per_layer.into_iter() {
            seg_kvs.push(SegmentKv {
                segment_index: seg_index,
                frame_start,
                frame_count: frames,
                image_tokens: video_tokens_total,
                text_tokens,
                action_start,
                k,
                v,
            });
        }
        bank.commit(seg_kvs)?;

        // ---- report --------------------------------------------------------
        let cache_tokens = bank.layer(0).valid_len();
        let report = SegmentReport {
            index: seg_index,
            latent_frames: frames,
            steps: steps_run,
            video_tokens: video_tokens_total,
            text_tokens,
            cache_tokens,
            archived_segments: bank.layer(0).n_segments(),
            duration_ms: t0.elapsed().as_secs_f64() * 1000.0,
            final_velocity_mean,
        };
        self.reports.push(report.clone());
        self.segment_index += 1;
        self.frame_offset += frames;
        Ok((report, latent))
    }

    /// Decode a segment's latents through the codec (with phase-swap parking
    /// when configured).
    pub fn decode(&mut self, latent: &Tensor) -> Result<Tensor> {
        let bank = self
            .bank
            .as_mut()
            .ok_or_else(|| Error::CacheState("session not begun".into()))?;
        bank.park_all()?;
        let frames = self.codec.decode_segment(latent);
        bank.unpark_all()?;
        frames
    }

    /// Roll back the last turn (interactive re-roll).
    pub fn rollback(&mut self) -> Result<()> {
        let bank = self
            .bank
            .as_mut()
            .ok_or_else(|| Error::CacheState("session not begun".into()))?;
        bank.rollback()?;
        if self.segment_index > 0 {
            self.segment_index -= 1;
        }
        if let Some(last) = self.reports.last() {
            self.frame_offset = self.frame_offset.saturating_sub(last.latent_frames);
        }
        self.reports.pop();
        Ok(())
    }

    /// Finish the session and produce the aggregate report.
    pub fn finish(&mut self) -> Result<SessionReport> {
        let total_frames = self.frame_offset;
        let total_ms = self
            .started
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let cache = self.bank.as_ref().map(|b| b.metrics()).unwrap_or_default();
        Ok(SessionReport {
            segments: std::mem::take(&mut self.reports),
            total_frames,
            total_ms,
            cache,
        })
    }

    /// One-shot convenience: run a whole [`SessionPlan`], streaming decoded
    /// frames to `on_segment` when provided.
    pub fn generate_session(
        &mut self,
        plan: &SessionPlan,
        mut on_segment: Option<&mut OnSegment<'_>>,
    ) -> Result<SessionReport> {
        self.begin(plan.latent_hw)?;
        let hw = plan.latent_hw;
        for seg in &plan.segments {
            let (report, latent) = self.turn(seg, plan.latent_fps, hw)?;
            if let Some(cb) = on_segment.as_mut() {
                let frames = self.decode(&latent)?;
                cb(&report, &frames);
            }
        }
        self.finish()
    }
}

/// Build a camera trajectory with `frames` poses from a shorter one by
/// holding the last pose (convenience for interactive turns).
pub fn fit_trajectory(traj: &CameraTrajectory, frames: usize) -> CameraTrajectory {
    let mut poses = Vec::with_capacity(frames);
    for i in 0..frames {
        let idx = i.min(traj.poses.len().saturating_sub(1));
        poses.push(traj.poses[idx].clone());
    }
    CameraTrajectory {
        poses,
        intrinsics: traj.intrinsics,
    }
}

/// Build a default intrinsics helper for CLI use.
pub fn default_intrinsics(w: usize, h: usize) -> Intrinsics {
    Intrinsics::for_resolution(w, h)
}

/// Video-only pending history for chunk-level AR: per-layer concatenated KV
/// of the already-denoised chunks plus their token counts.
fn build_pending(chunk_kvs: &[(usize, Vec<LayerKv>)]) -> Result<(Vec<LayerKv>, Vec<usize>)> {
    if chunk_kvs.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let n_layers = chunk_kvs[0].1.len();
    let mut pending = Vec::with_capacity(n_layers);
    for li in 0..n_layers {
        let mut k_parts = Vec::with_capacity(chunk_kvs.len());
        let mut v_parts = Vec::with_capacity(chunk_kvs.len());
        for (vt, kv) in chunk_kvs {
            k_parts.push(kv[li].k.narrow(1usize, 0, *vt)?.contiguous()?);
            v_parts.push(kv[li].v.narrow(1usize, 0, *vt)?.contiguous()?);
        }
        let k = Tensor::cat(&k_parts.iter().collect::<Vec<_>>(), 1)?;
        let v = Tensor::cat(&v_parts.iter().collect::<Vec<_>>(), 1)?;
        pending.push(LayerKv { k, v });
    }
    let tokens = chunk_kvs.iter().map(|(vt, _)| *vt).collect();
    Ok((pending, tokens))
}
