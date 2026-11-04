//! Joint self-attention with QK-norm, RoPE and the lifelong cache — the
//! portable realization of `nomas_cpt.py`'s `causal_attention_with_cache_mp`.
//!
//! One call projects `q/k/v` for the *current* tokens, applies QK-RMSNorm and
//! 3D RoPE (so cached keys — already rotated at commit time — line up),
//! concatenates the history working set, applies the action-aware additive
//! mask and runs scaled dot-product attention with an F32 softmax.
//!
//! The upstream `magi`/`flex`/`triton` GPU backends all implement the same
//! math; this portable backend keeps the port dependency-free. A fused flash
//! kernel can be slotted behind the same signature on CUDA builds.

use candle_core::{DType, Tensor};

use super::norm::RmsNorm;
use super::{Lin, Weights};
use crate::rope::{apply_rotary, RotaryTable};
use crate::Result;

/// Attention projections + QK norms for one layer.
pub struct Attention {
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_heads: usize,
    head_dim: usize,
}

/// Result of one attention pass.
pub struct AttentionIo {
    /// Output activations `(B, L, D)`.
    pub x: Tensor,
    /// Rotated keys of the current tokens `(B, L, H, D)` — cache-commit
    /// material.
    pub k: Tensor,
    /// Values of the current tokens `(B, L, H, D)`.
    pub v: Tensor,
}

impl Attention {
    /// Build from a source.
    pub fn from_source(
        src: &Weights<'_>,
        path: &str,
        dim: usize,
        n_heads: usize,
        qk_norm: bool,
    ) -> Result<Self> {
        let head_dim = dim / n_heads;
        Ok(Self {
            q: Lin::from_source(src, &format!("{path}.q"), dim, dim, true)?,
            k: Lin::from_source(src, &format!("{path}.k"), dim, dim, true)?,
            v: Lin::from_source(src, &format!("{path}.v"), dim, dim, true)?,
            out: Lin::from_source(src, &format!("{path}.out"), dim, dim, true)?,
            q_norm: if qk_norm {
                Some(RmsNorm::from_source(
                    src,
                    &format!("{path}.q_norm"),
                    head_dim,
                    true,
                    1e-6,
                )?)
            } else {
                None
            },
            k_norm: if qk_norm {
                Some(RmsNorm::from_source(
                    src,
                    &format!("{path}.k_norm"),
                    head_dim,
                    true,
                    1e-6,
                )?)
            } else {
                None
            },
            n_heads,
            head_dim,
        })
    }

    /// Head count.
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Parameter count.
    pub fn n_params(&self) -> usize {
        self.q.n_params()
            + self.k.n_params()
            + self.v.n_params()
            + self.out.n_params()
            + self.q_norm.as_ref().map(|n| n.n_params()).unwrap_or(0)
            + self.k_norm.as_ref().map(|n| n.n_params()).unwrap_or(0)
    }

    /// Project `(B, L, D)` -> three `(B, L, H, D)` tensors with QK-norm
    /// applied over the head dim.
    pub fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let dims = x.dims();
        let (b, l, _) = match dims {
            [b, l, d] => (*b, *l, *d),
            s => crate::bail!("attention input must be (B,L,D), got {s:?}"),
        };
        let shape = (b, l, self.n_heads, self.head_dim);
        let mut q = self.q.fwd(x)?.reshape(shape)?;
        let mut k = self.k.fwd(x)?.reshape(shape)?;
        let v = self.v.fwd(x)?.reshape(shape)?;
        if let Some(n) = &self.q_norm {
            q = n.forward(&q)?;
        }
        if let Some(n) = &self.k_norm {
            k = n.forward(&k)?;
        }
        Ok((q, k, v))
    }

    /// Full attention over `[history | current]` keys.
    ///
    /// * `q/k/v` — current-token projections (`(B, L, H, D)`, pre-RoPE).
    /// * `rope` — RoPE table covering the current tokens.
    /// * `k_hist/v_hist` — working-set tensors (`(B, Lh, H, D)`, already
    ///   rotated keys).
    /// * `mask` — additive `(1, 1, Lq, Lh + L)`.
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        rope: &RotaryTable,
        k_hist: &Tensor,
        v_hist: &Tensor,
        mask: &Tensor,
    ) -> Result<AttentionIo> {
        let qr = apply_rotary(q, rope)?;
        let kr = apply_rotary(k, rope)?;
        let k_all = Tensor::cat(&[k_hist, &kr], 1)?;
        let v_all = Tensor::cat(&[v_hist, v], 1)?;
        let ctx = sdpa(&qr, &k_all, &v_all, mask, self.n_heads, self.head_dim)?;
        let x = ctx.reshape((ctx.dims()[0], ctx.dims()[1], self.n_heads * self.head_dim))?;
        Ok(AttentionIo {
            x: self.out.fwd(&x)?,
            k: kr,
            v: v.clone(),
        })
    }
}

/// Scaled dot-product attention with an additive mask.
///
/// * `q`: `(B, Lq, H, D)`, `k`/`v`: `(B, Lk, H, D)`, `mask`: `(1, 1, Lq, Lk)`
///   additive (0 = visible).
/// * Returns `(B, Lq, H, D)`. Softmax runs in F32.
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let dims = q.dims();
    let (b, lq, _h, d) = match dims {
        [b, lq, h, d] => (*b, *lq, *h, *d),
        s => crate::bail!("sdpa q must be (B,L,H,D), got {s:?}"),
    };
    let _ = (n_heads, head_dim);
    let qh = q.transpose(1, 2)?.contiguous()?; // (B, H, Lq, D)
    let kh = k.transpose(1, 2)?.contiguous()?; // (B, H, Lk, D)
    let vh = v.transpose(1, 2)?.contiguous()?; // (B, H, Lk, D)

    let mut scores = qh.matmul(&kh.t()?)?.to_dtype(DType::F32)?;
    let scale = 1.0 / (d as f64).sqrt();
    scores = scores.affine(scale, 0.0)?;
    scores = scores.broadcast_add(mask)?;
    let attn = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
    let ctx = attn.to_dtype(v.dtype())?.matmul(&vh)?; // (B, H, Lq, D)
    let out = ctx.transpose(1, 2)?.contiguous()?; // (B, Lq, H, D)
    let _ = b;
    let _ = lq;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masks::MASKED;
    use crate::tensor_utils::ToVecExt;
    use candle_core::{DType, Device};

    #[test]
    fn sdpa_uniform_mask_averages_values() {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 2, 2, 8), DType::F32, &dev).unwrap();
        let k = Tensor::zeros((1, 3, 2, 8), DType::F32, &dev).unwrap();
        let v = crate::tensor_utils::deterministic_randn([1, 3, 2, 8], 3, &dev).unwrap();
        let mask = Tensor::zeros((1usize, 1usize, 2usize, 3usize), DType::F32, &dev).unwrap();
        let out = sdpa(&q, &k, &v, &mask, 2, 8).unwrap();
        let vv = v.to_vec4::<f32>().unwrap();
        let ov = out.to_vec4::<f32>().unwrap();
        // zero q/k -> uniform attention -> mean of values per head
        for h in 0..2 {
            for d in 0..8 {
                let mean = (vv[0][0][h][d] + vv[0][1][h][d] + vv[0][2][h][d]) / 3.0;
                assert!((ov[0][0][h][d] - mean).abs() < 1e-5);
                assert!((ov[0][1][h][d] - mean).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn sdpa_respects_mask() {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 1, 1, 8), DType::F32, &dev).unwrap();
        let k = Tensor::zeros((1, 2, 1, 8), DType::F32, &dev).unwrap();
        let v = Tensor::from_vec(vec![5f32, 7.0], (1usize, 2usize, 1usize, 1usize), &dev).unwrap();
        // mask the second key
        let mask =
            Tensor::from_vec(vec![0f32, MASKED], (1usize, 1usize, 1usize, 2usize), &dev).unwrap();
        let out = sdpa(&q, &k, &v, &mask, 1, 8).unwrap();
        let ov = out.to_vec4::<f32>().unwrap();
        assert!(
            (ov[0][0][0][0] - 5.0).abs() < 1e-4,
            "got {}",
            ov[0][0][0][0]
        );
    }

    #[test]
    fn attention_io_shapes() {
        let dev = Device::Cpu;
        let attn =
            Attention::from_source(&Weights::random(6, DType::F32, &dev), "attn", 32, 4, true)
                .unwrap();
        let x = crate::tensor_utils::deterministic_randn([1, 6, 32], 2, &dev).unwrap();
        let (q, k, v) = attn.project(&x).unwrap();
        let rope =
            crate::rope::rotary_table_with_offset(&[2, 2, 4], (1, 2, 3), 0.0, 256.0, &dev).unwrap();
        let zeros = Tensor::zeros((1, 0usize, 4, 8), DType::F32, &dev).unwrap();
        let mask = Tensor::zeros((1usize, 1usize, 6usize, 6usize), DType::F32, &dev).unwrap();
        let io = attn
            .forward(&q, &k, &v, &rope, &zeros, &zeros, &mask)
            .unwrap();
        assert_eq!(io.x.dims(), vec![1, 6, 32]);
        assert_eq!(io.k.dims(), vec![1, 6, 4, 8]);
        assert_eq!(io.v.dims(), vec![1, 6, 4, 8]);
    }
}
