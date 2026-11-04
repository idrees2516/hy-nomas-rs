//! `NomasDiT` — the single-stream interactive video DiT (`nomas_cpt.py`).
//!
//! One forward pass denoises the *current* tokens (a segment or a chunk of
//! latent frames) while attending to the **lifelong KV cache** of all previous
//! segments, with per-segment text modulation, timed action windows and
//! per-frame Plücker camera conditioning.
//!
//! Token layout of a forward (mirrors the cache packing):
//!
//! ```text
//! [ history (from cache) | current video tokens | current text tokens ]
//! ```

use candle_core::{Device, Tensor};

use crate::actions::TextBlockLayout;
use crate::cache::{LifetimeKvCache, SegmentLayout};
use crate::config::NomasConfig;
use crate::masks::{build_attention_mask, IntraCausality, MaskSpec};
use crate::modules::block::DiTBlock;
use crate::modules::final_layer::FinalLayer;
use crate::modules::patch_embed::PatchEmbed;
use crate::modules::text_embed::TextProjector;
use crate::modules::timestep::TimestepEmbedder;
use crate::modules::{Lin, Weights};
use crate::rope::{rotary_table_from_coords, RotaryTable};
use crate::Result;

/// Per-layer captured KV of the current tokens (rotated keys).
#[derive(Debug, Clone)]
pub struct LayerKv {
    /// Keys `(B, L, H, D)`.
    pub k: Tensor,
    /// Values `(B, L, H, D)`.
    pub v: Tensor,
}

/// Read-only view of everything attendable *before* the current tokens:
/// the lifelong caches plus the video-only KV of chunks already denoised
/// inside the current segment (chunk-level AR).
pub struct HistoryView<'a> {
    /// One cache per block.
    pub caches: &'a [&'a dyn LifetimeKvCache],
    /// Video-only per-layer KV of previously denoised chunks of this segment
    /// (each entry = one layer, tokens of all chunks concatenated).
    pub pending: &'a [LayerKv],
    /// Video-token count of each denoised chunk, in order.
    pub pending_chunk_tokens: &'a [usize],
}

impl<'a> HistoryView<'a> {
    /// Caches only, no pending chunks.
    pub fn from_caches(caches: &'a [&'a dyn LifetimeKvCache]) -> Self {
        Self {
            caches,
            pending: &[],
            pending_chunk_tokens: &[],
        }
    }

    /// Per-layer history K/V, materialized.
    pub fn layer_history(
        &self,
        i: usize,
        dtype: candle_core::DType,
        dev: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let _ = dev;
        let (mut k, mut v) = self.caches[i].working_set()?;
        if let Some(p) = self.pending.get(i) {
            k = Tensor::cat(&[&k, &p.k.to_dtype(dtype)?], 1)?;
            v = Tensor::cat(&[&v, &p.v.to_dtype(dtype)?], 1)?;
        }
        Ok((k, v))
    }

    /// Combined token layout (cache + pending) for the mask builder.
    pub fn layout(&self) -> Vec<SegmentLayout> {
        let mut layout = self
            .caches
            .first()
            .map(|c| c.working_layout())
            .unwrap_or_default();
        let mut cursor = self.caches.first().map(|c| c.valid_len()).unwrap_or(0);
        for (ci, &tokens) in self.pending_chunk_tokens.iter().enumerate() {
            layout.push(SegmentLayout {
                segment_index: usize::MAX - ci,
                video_start: cursor,
                video_end: cursor + tokens,
                text_start: cursor + tokens,
                text_end: cursor + tokens,
                action_start: -1,
            });
            cursor += tokens;
        }
        layout
    }

    /// Total history token count (cache + pending).
    pub fn len(&self) -> usize {
        let base = self.caches.first().map(|c| c.valid_len()).unwrap_or(0);
        base + self.pending_chunk_tokens.iter().sum::<usize>()
    }

    /// Whether there is no history at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Output of one segment forward.
pub struct ForwardOut {
    /// Flow-matching velocity in latent space `(B, C, T, H, W)`.
    pub velocity: Tensor,
    /// Captured per-layer KV when requested (for cache commits).
    pub kv: Option<Vec<LayerKv>>,
}

/// The full model.
pub struct NomasDiT {
    cfg: NomasConfig,
    patch_embed: PatchEmbed,
    text_proj: TextProjector,
    time_embed: TimestepEmbedder,
    cam_mlp1: Lin,
    cam_mlp2: Lin,
    blocks: Vec<DiTBlock>,
    final_layer: FinalLayer,
}

impl NomasDiT {
    /// Build from a weight source (checkpoint or deterministic random).
    pub fn from_source(src: &Weights<'_>, cfg: &NomasConfig) -> Result<Self> {
        cfg.validate()?;
        let freq_dim = cfg.dim.max(32);
        let blocks = (0..cfg.n_layers)
            .map(|i| DiTBlock::from_source(src, cfg, i))
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed: PatchEmbed::from_source(src, cfg.in_channels, cfg.patch_size, cfg.dim)?,
            text_proj: TextProjector::from_source(
                src,
                cfg.text_embed_dim,
                cfg.dim,
                cfg.text_separator,
            )?,
            time_embed: TimestepEmbedder::from_source(src, "time_embed", cfg.dim, freq_dim)?,
            cam_mlp1: Lin::from_source(src, "cam_embed.mlp1", 6, cfg.dim, true)?,
            cam_mlp2: Lin::from_source(src, "cam_embed.mlp2", cfg.dim, cfg.dim, true)?,
            blocks,
            final_layer: FinalLayer::from_source(
                src,
                cfg.dim,
                cfg.patch_in_features() / cfg.patch_size.0, // per-frame patch features
            )?,
            cfg: cfg.clone(),
        })
    }

    /// Configuration reference.
    pub fn config(&self) -> &NomasConfig {
        &self.cfg
    }

    /// The device holding the parameters.
    pub fn device(&self) -> &Device {
        self.patch_embed.proj.weight.device()
    }

    /// The parameter dtype.
    pub fn dtype(&self) -> candle_core::DType {
        self.patch_embed.proj.weight.dtype()
    }

    /// Total parameter count.
    pub fn n_params(&self) -> usize {
        let mut n = 0usize;
        n += self.patch_embed.proj.n_params();
        n += self.text_proj.proj.n_params();
        n += self.time_embed.mlp1.n_params() + self.time_embed.mlp2.n_params();
        n += self.cam_mlp1.n_params() + self.cam_mlp2.n_params();
        n += self.final_layer.ada.n_params() + self.final_layer.linear.n_params();
        for b in &self.blocks {
            n += b.n_params();
        }
        n
    }

    /// Camera embedding: `(T, 6)` Plücker features -> `(T, D)` (or `(1, D)`
    /// for `None` — a neutral default camera).
    fn camera_embed(
        &self,
        camera: Option<&Tensor>,
        frames: usize,
        dev: &Device,
        dtype: candle_core::DType,
    ) -> Result<Tensor> {
        let feats = match camera {
            Some(c) => c.to_dtype(candle_core::DType::F32)?,
            None => Tensor::zeros((frames, 6usize), candle_core::DType::F32, dev)?,
        };
        let h = self.cam_mlp1.fwd(&feats)?.silu()?;
        let out = self.cam_mlp2.fwd(&h)?.silu()?;
        Ok(out.to_dtype(dtype)?)
    }

    /// Build the RoPE table covering `[video tokens | text tokens]`.
    ///
    /// Video tokens use their **global** temporal coordinates (so cached,
    /// already-rotated keys line up); text tokens sit at the chunk's temporal
    /// start with cycling spatial coordinates.
    fn segment_rope(
        &self,
        grid: (usize, usize, usize),
        frame_offset: usize,
        text_tokens: usize,
        dev: &Device,
    ) -> Result<RotaryTable> {
        let (tt, th, tw) = grid;
        let tpf = th * tw;
        let n_video = tt * tpf;
        let mut t_coords = Vec::with_capacity(n_video + text_tokens);
        let mut h_coords = Vec::with_capacity(n_video + text_tokens);
        let mut w_coords = Vec::with_capacity(n_video + text_tokens);
        for f in 0..tt {
            for h in 0..th {
                for w in 0..tw {
                    t_coords.push((frame_offset + f) as f32);
                    h_coords.push(h as f32);
                    w_coords.push(w as f32);
                }
            }
        }
        for i in 0..text_tokens {
            t_coords.push(frame_offset as f32);
            h_coords.push((i % th.max(1)) as f32);
            w_coords.push((i / th.max(1) % tw.max(1)) as f32);
        }
        let mk = |v: Vec<f32>| -> Result<Tensor> {
            let n = v.len();
            Ok(Tensor::from_vec(v, (n,), dev)?)
        };
        let rope_dims = self.cfg.rope_dims();
        let dims = [rope_dims[0], rope_dims[1], rope_dims[2]];
        let t_t = mk(t_coords)?;
        let t_h = mk(h_coords)?;
        let t_w = mk(w_coords)?;
        rotary_table_from_coords(
            &dims,
            [&t_t, &t_h, &t_w],
            self.cfg.rope_theta,
            self.cfg.rope_theta_rescale,
            self.cfg.rope_interpolation,
            dev,
        )
        .map_err(Into::into)
    }

    /// Denoise one forward for the current tokens.
    ///
    /// * `latent` — `(B, C, T, H, W)` noisy latent (the whole segment or the
    ///   active chunk).
    /// * `timestep` — flow-matching time in `[0, 1]`.
    /// * `text` — `(B, Lt, D_text)` encoded text features of this segment.
    /// * `text_layout` — action windows resolved **for this forward's frame
    ///   range** (chunk-local frame indices).
    /// * `camera` — optional `(T, 6)` pooled Plücker features per latent frame.
    /// * `frame_offset` — global latent-frame index of the first frame.
    /// * `causality` — intra-segment causality mode.
    /// * `history` — the caches (+ optional pending chunks) to attend to.
    /// * `capture` — also return per-layer K/V of the current tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_segment(
        &self,
        latent: &Tensor,
        timestep: f64,
        text: &Tensor,
        text_layout: &TextBlockLayout,
        camera: Option<&Tensor>,
        frame_offset: usize,
        causality: IntraCausality,
        history: &HistoryView<'_>,
        capture: bool,
    ) -> Result<ForwardOut> {
        crate::ensure!(
            history.caches.len() == self.blocks.len(),
            "forward_segment needs {} caches, got {}",
            self.blocks.len(),
            history.caches.len()
        );
        let dev = latent.device();
        let b = latent.dims()[0];
        let (t, h, w) = match latent.dims() {
            [_, _, t, h, w] => (*t, *h, *w),
            s => crate::bail!("latent must be (B,C,T,H,W), got {s:?}"),
        };
        let grid = self.patch_embed.token_grid(t, h, w);
        let (tt, th, tw) = grid;
        let tpf = th * tw;
        let video_tokens = tt * tpf;

        // ---- tokens ----------------------------------------------------
        let x = self.patch_embed.forward(latent)?; // (B, Nv, D)
        let txt = self.text_proj.forward(text)?; // (B, Nt, D)
        let nt = txt.dims()[1];
        crate::ensure!(
            nt == text_layout.total_tokens,
            "text token count {nt} != layout total {}",
            text_layout.total_tokens
        );

        // ---- per-frame camera conditioning ------------------------------
        let cam = self.camera_embed(camera, tt, dev, x.dtype())?; // (tt, D)
        let cam = cam.unsqueeze(0)?; // (1, tt, D)
        let xv = x.reshape((b, tt, tpf, ()))?;
        let xv = xv.broadcast_add(&cam.unsqueeze(2)?)?;
        let x_video = xv.reshape((b, video_tokens, ()))?;

        // ---- packed sequence: [video tokens | text tokens] ---------------
        // The cache and mask both use this packing (video first, then the
        // text block with its action tokens).
        let mut x = Tensor::cat(&[&x_video, &txt], 1)?;

        // ---- pooled condition (per-chunk text modulation) ---------------
        let pooled_text = txt.mean(1)?; // (B, D)
        let time = self.time_embed.forward(&vec![timestep; b], dev)?; // (B, D)
        let cond = (time + pooled_text)?;

        // ---- mask + rope -------------------------------------------------
        let history_layout = history.layout();
        let history_len = history.len();
        for (i, c) in history.caches.iter().enumerate() {
            crate::ensure!(
                c.valid_len() == history.caches[0].valid_len(),
                "cache layer {i} diverged: {} vs {}",
                c.valid_len(),
                history.caches[0].valid_len()
            );
        }
        let spec = MaskSpec {
            history_layout: &history_layout,
            history_len,
            video_tokens,
            frames: tt,
            tokens_per_frame: tpf,
            text_layout,
            causality,
        };
        let mask = build_attention_mask(&spec, dev)?;
        let rope = self.segment_rope(grid, frame_offset, nt, dev)?;

        // ---- blocks -------------------------------------------------------
        let mut captured: Vec<LayerKv> = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let (k_hist, v_hist) = history.layer_history(i, x.dtype(), dev)?;
            let (x_new, k, v) = block.forward(&x, &cond, &rope, &k_hist, &v_hist, &mask)?;
            x = x_new;
            if capture {
                captured.push(LayerKv { k, v });
            }
        }

        // ---- head ---------------------------------------------------------
        // Only the video tokens carry a velocity: text tokens are pure
        // conditioning, so they are dropped before the final layer.
        let x_video_out = x.narrow(1usize, 0, video_tokens)?;
        let y = self.final_layer.forward(&x_video_out, &cond)?;
        let velocity = self.patch_embed.unpatchify(&y, grid)?;
        Ok(ForwardOut {
            velocity,
            kv: if capture { Some(captured) } else { None },
        })
    }
}
