//! Model, cache and solver configuration.
//!
//! Presets encode the *documented defaults* of the upstream Hy-Nomas release
//! (`configs/config.yaml`): RoPE theta `256`, 4/6/8-step distilled inference,
//! INT8 / QVG KV cache presets (`num_bits 2`, 4 stages, 256 centroids, block
//! size 64, text stored as INT8), camera chunk of 5 latent frames and
//! chunk-mode `1` (full bidirectional attention inside a segment).
//!
//! > **Note:** the transformer dimensions of the 8B checkpoint are not part of the
//! > public config; [`NomasConfig::nomas_8b`] encodes a faithful preset that is
//! > overridden automatically when loading a real checkpoint via
//! > [`crate::weights`]. Treat them as placeholders for the released geometry.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Activation used inside the DiT MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Activation {
    /// `gelu(x)` with the exact erf definition (HunyuanVideo-style FFN).
    #[default]
    GeluErf,
    /// SwiGLU gated MLP (`silu(gate(x)) * up(x)`).
    Silu,
}

/// Numeric dtype used for model weights and activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DataType {
    /// 32-bit floats (portable CPU default).
    #[default]
    F32,
    /// Brain floats — the upstream inference dtype; requires CUDA builds for
    /// fast kernels but works on CPU too.
    Bf16,
    /// 16-bit floats.
    F16,
}

impl From<DataType> for candle_core::DType {
    fn from(v: DataType) -> Self {
        match v {
            DataType::F32 => candle_core::DType::F32,
            DataType::Bf16 => candle_core::DType::BF16,
            DataType::F16 => candle_core::DType::F16,
        }
    }
}

/// How old segments are sampled into the GPU working set of the lifelong cache.
///
/// Mirrors `HISTORY_SAMPLINGS` in `hymm/models/dit/modules/uniform_history_cache.py`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum HistorySampling {
    /// Keep the most recent frames only (plain SWA).
    Recent,
    /// Endpoint-inclusive uniform grid over the complete archived history.
    #[default]
    Uniform,
    /// Half budget of contiguous recent frames + uniform grid over the older
    /// prefix.
    RecentMixUniform,
    /// Power-spaced distances from the newest frame (`gamma > 1` biases towards
    /// recent frames, `gamma = 1` degenerates to [`HistorySampling::Uniform`]).
    TemporalContext { gamma: f64 },
}

impl fmt::Display for HistorySampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Recent => write!(f, "recent"),
            Self::Uniform => write!(f, "uniform"),
            Self::RecentMixUniform => write!(f, "recent_mix_uniform"),
            Self::TemporalContext { gamma } => write!(f, "temporal_context(gamma={gamma})"),
        }
    }
}

/// QVG (residual vector quantization) KV codec preset.
///
/// Upstream defaults (`configs/config.yaml`): `num_bits: 2`, `4` stages,
/// `256` K/V centroids, block size `64` and INT8 storage for text tokens.
/// `nomas-core` realizes this as a residual VQ: `stages` residual passes over
/// `2^bits_per_stage`-entry codebooks — the `4^4 = 256`-entry *product* code
/// of the upstream centroid count — with one scale per 64-token block. See
/// `cache::qvg` for the codec description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QvgConfig {
    /// Number of residual stages (upstream: 4).
    pub stages: usize,
    /// Code bits per stage (upstream: 2 → 4 entries per stage).
    pub bits_per_stage: usize,
    /// Tokens per quantization block, i.e. scale granularity (upstream: 64).
    pub block_size: usize,
    /// Store text-token KV as INT8 instead of RVQ (upstream: `int8`).
    pub text_as_int8: bool,
    /// Random seed used when codebooks must be synthesized instead of loaded.
    pub codebook_seed: u64,
}

impl Default for QvgConfig {
    fn default() -> Self {
        Self {
            stages: 4,
            bits_per_stage: 2,
            block_size: 64,
            text_as_int8: true,
            codebook_seed: 0x5EED_0002,
        }
    }
}

/// Storage backend for the lifelong KV cache.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum CacheStorage {
    /// Floating-point KV (BF16/F16/F32 according to [`NomasConfig::dtype`]).
    #[default]
    Floating,
    /// INT8 KV with per-token-per-head FP32 scales (`int8_kv_cache.py`).
    Int8,
    /// Residual-vector-quantized KV with the upstream QVG preset.
    Qvg(QvgConfig),
}

impl fmt::Display for CacheStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Floating => write!(f, "f32"),
            Self::Int8 => write!(f, "int8"),
            Self::Qvg(q) => write!(
                f,
                "qvg({} stages x {}b, block {})",
                q.stages, q.bits_per_stage, q.block_size
            ),
        }
    }
}

/// Lifelong-KV-cache configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Storage format of archived + working-set KV.
    pub storage: CacheStorage,
    /// History sampling policy for the working set.
    pub sampling: HistorySampling,
    /// Token capacity of the GPU working set (`swa_capacity`). `None` = grow
    /// unbounded (true *lifelong* cache; upstream `max_cache_segments: null`).
    pub capacity: Option<usize>,
    /// Frame budget (`swa_past_frames`) for archived-history samplers.
    pub past_frames: usize,
    /// Denoise step at which KV is captured for the cache (`cache_at_step`,
    /// `-1` = final step).
    pub cache_at_step: isize,
    /// Optional noise re-injection for cached latents (`cache_noise_sigma`).
    pub cache_noise_sigma: f64,
    /// Whether to park the working set on the host between denoise phases
    /// (`PhaseSwappedKVCache`).
    pub phase_swap: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            storage: CacheStorage::Floating,
            sampling: HistorySampling::Recent,
            capacity: None,
            past_frames: 64,
            cache_at_step: -1,
            cache_noise_sigma: 0.0,
            phase_swap: false,
        }
    }
}

/// Flow-matching solver configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Number of Euler steps (distilled checkpoints: 4 / 6 / 8).
    pub steps: usize,
    /// Shifted-timestep flow parameter (SD3-style `t' = s·t/(1+(s-1)·t)`); `1.0`
    /// = uniform grid.
    pub shift: f64,
    /// Additive-noise (SDE) mixing coefficient; `0.0` = pure ODE (upstream
    /// `denoise_mode: "ode"`).
    pub sigma: f64,
    /// Random seed for the initial noise and any stochasticity.
    pub seed: u64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            steps: 6,
            shift: 1.0,
            sigma: 0.0,
            seed: 0x5EED_0001,
        }
    }
}

/// Mixture-of-experts settings for the DiT FFN.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Number of experts.
    pub experts: usize,
    /// Experts activated per token (top-k routing).
    pub top_k: usize,
    /// Interpolation coefficient between dense and MoE outputs (0 = pure MoE).
    pub shared_dense: f64,
    /// Random seed for router jitter when initializing without a checkpoint.
    pub seed: u64,
}

/// Full model + runtime configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NomasConfig {
    /// DCAE-V latent channels (32 upstream).
    pub in_channels: usize,
    /// Output channels of the velocity head (matches `in_channels`).
    pub out_channels: usize,
    /// Latent patch size `(t, h, w)` — `(1, 2, 2)` upstream.
    pub patch_size: (usize, usize, usize),

    /// Transformer width.
    pub dim: usize,
    /// Number of single-stream DiT blocks.
    pub n_layers: usize,
    /// Attention heads.
    pub n_heads: usize,
    /// FFN hidden multiplier (hidden = dim * ratio).
    pub mlp_ratio: f64,
    /// FFN activation.
    pub mlp_activation: Activation,

    /// Last-hidden size of the text encoder (LLaMA-3.1-8B style encoders).
    pub text_embed_dim: usize,
    /// Learnable separator appended to every text block.
    pub text_separator: bool,

    /// QK RMSNorm before RoPE.
    pub qk_norm: bool,

    /// RoPE base theta — **256** in the upstream config (not 10000).
    pub rope_theta: f64,
    /// Per-axis RoPE dims `[t, h, w]`; defaults to `head_dim / 3` each.
    pub rope_dim_list: Option<Vec<usize>>,
    /// NTK-style theta rescale per axis.
    pub rope_theta_rescale: f64,
    /// Position interpolation per axis.
    pub rope_interpolation: f64,

    /// Optional MoE FFN.
    pub moe: Option<MoeConfig>,

    /// Weight/activation dtype.
    pub dtype: DataType,

    /// Lifelong KV cache settings.
    pub cache: CacheConfig,
    /// Solver settings.
    pub solver: SolverConfig,
}

impl NomasConfig {
    /// Small deterministic configuration used by tests, examples and the CLI
    /// demo — runs end-to-end in milliseconds on a laptop CPU.
    pub fn tiny() -> Self {
        Self {
            in_channels: 32,
            out_channels: 32,
            patch_size: (1, 2, 2),
            dim: 64,
            n_layers: 2,
            n_heads: 4,
            mlp_ratio: 4.0,
            mlp_activation: Activation::Silu,
            text_embed_dim: 32,
            text_separator: true,
            qk_norm: true,
            rope_theta: 256.0,
            rope_dim_list: Some(vec![4, 6, 6]),
            rope_theta_rescale: 1.0,
            rope_interpolation: 1.0,
            moe: None,
            dtype: DataType::F32,
            cache: CacheConfig::default(),
            solver: SolverConfig::default(),
        }
    }

    /// The 8B single-stream DiT preset.
    ///
    /// Geometry follows the HunyuanVideo-family convention for an 8B
    /// single-stream model and is *overridden from the checkpoint* when real
    /// weights are loaded — see the crate docs.
    pub fn nomas_8b() -> Self {
        Self {
            in_channels: 32,
            out_channels: 32,
            patch_size: (1, 2, 2),
            dim: 3072,
            n_layers: 48,
            n_heads: 24,
            mlp_ratio: 4.0,
            mlp_activation: Activation::Silu,
            text_embed_dim: 4096,
            text_separator: true,
            qk_norm: true,
            rope_theta: 256.0,
            rope_dim_list: Some(vec![16, 56, 56]),
            rope_theta_rescale: 1.0,
            rope_interpolation: 1.0,
            moe: None,
            dtype: DataType::Bf16,
            cache: CacheConfig {
                storage: CacheStorage::Int8,
                sampling: HistorySampling::Uniform,
                capacity: None,
                past_frames: 64,
                cache_at_step: -1,
                cache_noise_sigma: 0.0,
                phase_swap: false,
            },
            solver: SolverConfig {
                steps: 6,
                shift: 1.0,
                sigma: 0.0,
                seed: 0x5EED_0001,
            },
        }
    }

    /// Attention head dimension.
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Per-axis RoPE dims, defaulting to an even split of the head dimension.
    pub fn rope_dims(&self) -> Vec<usize> {
        match &self.rope_dim_list {
            Some(list) => list.clone(),
            None => {
                let hd = self.head_dim();
                vec![hd / 3, (hd - hd / 3) / 2, hd - hd / 3 - (hd - hd / 3) / 2]
            }
        }
    }

    /// Tokens per latent frame after patchification: `(H/ph) * (W/pw)`.
    pub fn tokens_per_frame(&self, latent_h: usize, latent_w: usize) -> usize {
        (latent_h / self.patch_size.1) * (latent_w / self.patch_size.2)
    }

    /// Patch-embedding input features: `in_channels * pt * ph * pw`.
    pub fn patch_in_features(&self) -> usize {
        self.in_channels * self.patch_size.0 * self.patch_size.1 * self.patch_size.2
    }

    /// Validate invariants (head dim divisible, rope dims sum, patch sizes...).
    pub fn validate(&self) -> crate::Result<()> {
        crate::ensure!(
            self.dim.is_multiple_of(self.n_heads),
            "dim {} not divisible by n_heads {}",
            self.dim,
            self.n_heads
        );
        let hd = self.head_dim();
        let rope = self.rope_dims();
        crate::ensure!(
            rope.iter().sum::<usize>() == hd,
            "rope_dim_list {:?} must sum to head_dim {}",
            rope,
            hd
        );
        crate::ensure!(
            rope.len() == 3,
            "rope_dim_list must have exactly 3 axes (t, h, w)"
        );
        crate::ensure!(
            rope.iter().all(|d| d % 2 == 0),
            "rope_dim_list entries must be even (each axis uses d/2 frequency pairs), got {rope:?}"
        );
        crate::ensure!(
            self.patch_size.1 >= 1 && self.patch_size.2 >= 1,
            "invalid patch size {:?}",
            self.patch_size
        );
        crate::ensure!(
            self.mlp_ratio >= 0.5,
            "mlp_ratio {} too small",
            self.mlp_ratio
        );
        if let Some(moe) = &self.moe {
            crate::ensure!(
                moe.top_k >= 1 && moe.top_k <= moe.experts,
                "invalid MoE top_k/experts"
            );
        }
        crate::ensure!(self.solver.steps >= 1, "solver needs >= 1 step");
        Ok(())
    }
}

/// Convenience alias matching the upstream config vocabulary.
pub type StoragePreset = CacheStorage;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_validate() {
        NomasConfig::tiny().validate().unwrap();
        NomasConfig::nomas_8b().validate().unwrap();
    }

    #[test]
    fn rope_dims_sum_to_head_dim() {
        let cfg = NomasConfig::nomas_8b();
        assert_eq!(cfg.rope_dims().iter().sum::<usize>(), cfg.head_dim());
        assert_eq!(cfg.head_dim(), 128);
    }

    #[test]
    fn rope_dims_fallback() {
        let mut cfg = NomasConfig::tiny();
        cfg.rope_dim_list = None;
        cfg.dim = 144;
        cfg.n_heads = 3;
        assert_eq!(cfg.rope_dims(), vec![16, 16, 16]);
    }

    #[test]
    fn serde_roundtrip() {
        let cfg = NomasConfig::tiny();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: NomasConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }
}
