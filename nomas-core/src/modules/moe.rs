//! Mixture-of-Experts FFN — `hymm/models/dit/moe/`.
//!
//! Top-k softmax routing over SwiGLU experts. The portable evaluation applies
//! the *renormalized top-k weights* to every expert's output — numerically
//! identical to hard dispatch, and trivially correct on any backend. (A
//! gather/scatter fast path that skips unselected experts is a drop-in
//! optimization; see the README.)

use candle_core::{Result, Tensor};

use super::mlp::FeedForward;
use super::{Lin, Weights};
use crate::config::MoeConfig;

/// Router + expert bank.
pub struct MoeLayer {
    router: Lin,
    experts: Vec<FeedForward>,
    top_k: usize,
    shared_dense: f64,
    dense: Option<FeedForward>,
}

impl MoeLayer {
    /// Build from a source.
    pub fn from_source(
        src: &Weights<'_>,
        path: &str,
        dim: usize,
        hidden: usize,
        cfg: &MoeConfig,
    ) -> Result<Self> {
        let router = Lin::from_source(src, &format!("{path}.router"), dim, cfg.experts, false)?;
        let mut experts = Vec::with_capacity(cfg.experts);
        for e in 0..cfg.experts {
            experts.push(FeedForward::from_source(
                src,
                &format!("{path}.experts.{e}"),
                dim,
                hidden,
                crate::config::Activation::Silu,
            )?);
        }
        let dense = if cfg.shared_dense > 0.0 {
            Some(FeedForward::from_source(
                src,
                &format!("{path}.dense"),
                dim,
                hidden,
                crate::config::Activation::Silu,
            )?)
        } else {
            None
        };
        Ok(Self {
            router,
            experts,
            top_k: cfg.top_k,
            shared_dense: cfg.shared_dense,
            dense,
        })
    }

    /// Forward with top-k routing. `x`: `(B, S, D)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n_experts = self.experts.len();
        let logits = self.router.fwd(x)?; // (B, S, E)
        let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?;

        // Top-k mask via iterative argmax (k is small).
        let (b, s, e) = (probs.dims()[0], probs.dims()[1], probs.dims()[2]);
        let range = Tensor::arange(0u32, e as u32, probs.device())?
            .reshape((1, 1, e))?
            .broadcast_as((b, s, e))?;
        let mut remaining = probs.clone();
        let mut acc = Vec::with_capacity(self.top_k);
        for _ in 0..self.top_k {
            let best = remaining
                .argmax_keepdim(candle_core::D::Minus1)?
                .broadcast_as((b, s, e))?; // (B, S, E) one-hot positions
            let one_h = best.eq(&range)?; // (B, S, E) u8
                                          // zero out the selected expert's probability, accumulate weights
            let selected = remaining.broadcast_mul(&one_h.to_dtype(probs.dtype())?)?;
            if acc.is_empty() {
                acc.push(selected.clone());
            } else {
                let prev = acc.last().unwrap();
                acc.push((prev + &selected)?);
            }
            remaining = remaining.broadcast_sub(&selected)?;
        }
        let top_probs = acc.last().unwrap().clone(); // (B, S, E) — masked to top-k
        let norm = top_probs.sum_keepdim(candle_core::D::Minus1)?;
        let weights =
            top_probs.broadcast_div(&norm.broadcast_add(&norm.ones_like()?.affine(0.0, 1e-9)?)?)?;

        let mut out = x.zeros_like()?;
        for (e, expert) in self.experts.iter().enumerate() {
            let w = weights.narrow(2usize, e, 1)?; // (B, S, 1)
            let y = expert.forward(x)?;
            out = out.broadcast_add(&y.broadcast_mul(&w)?)?;
        }
        if let Some(dense) = &self.dense {
            let y = dense.forward(x)?;
            out = out.broadcast_add(&y.affine(self.shared_dense, 0.0)?)?;
        }
        let _ = n_experts;
        Ok(out)
    }

    /// Parameter count.
    pub fn n_params(&self) -> usize {
        self.router.n_params() + self.experts.iter().map(|e| e.n_params()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::Weights;
    use candle_core::{DType, Device};

    #[test]
    fn moe_output_shape_and_finiteness() {
        let dev = Device::Cpu;
        let cfg = MoeConfig {
            experts: 4,
            top_k: 2,
            shared_dense: 0.0,
            seed: 1,
        };
        let moe = MoeLayer::from_source(
            &Weights::random(4, DType::F32, &dev),
            "blocks.0.moe",
            32,
            64,
            &cfg,
        )
        .unwrap();
        let x = crate::tensor_utils::deterministic_randn([1, 5, 32], 3, &dev).unwrap();
        let y = moe.forward(&x).unwrap();
        assert_eq!(y.dims(), vec![1, 5, 32]);
        assert!(y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn top_k_weights_sum_to_one() -> candle_core::Result<()> {
        // sanity of the routing math: renormalized top-k weights sum to 1
        let dev = Device::Cpu;
        let probs =
            crate::tensor_utils::deterministic_uniform([1, 3, 5], 5, 0.1, 0.9, &dev).unwrap();
        let (b, s, e) = (probs.dims()[0], probs.dims()[1], probs.dims()[2]);
        let range = Tensor::arange(0u32, e as u32, probs.device())?
            .reshape((1, 1, e))?
            .broadcast_as((b, s, e))?;
        let mut remaining = probs.clone();
        let mut acc = None;
        for _ in 0..2 {
            let best = remaining
                .argmax_keepdim(candle_core::D::Minus1)?
                .broadcast_as((b, s, e))?;
            let one_h = best.eq(&range)?;
            let selected: Tensor = remaining.broadcast_mul(&one_h.to_dtype(probs.dtype())?)?;
            acc = Some(match acc {
                None => selected.clone(),
                Some(a) => (&a + &selected).unwrap(),
            });
            remaining = remaining.broadcast_sub(&selected).unwrap();
        }
        let top = acc.unwrap();
        let norm = top.sum_keepdim(candle_core::D::Minus1).unwrap();
        let w = top.broadcast_div(&norm).unwrap();
        let sums = w
            .sum_keepdim(candle_core::D::Minus1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(sums.iter().all(|&s| (s - 1.0).abs() < 1e-5));
        Ok(())
    }
}
