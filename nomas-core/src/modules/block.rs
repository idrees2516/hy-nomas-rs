//! The DiT block — AdaLN-Zero self-attention + FFN (dense or MoE).

use crate::Result;

use super::attention::Attention;
use super::mlp::FeedForward;
use super::modulate::{modulate, Modulation};
use super::moe::MoeLayer;
use super::norm::RmsNorm;
use super::Weights;
use crate::config::NomasConfig;

/// FFN variant of a block.
pub enum BlockFfn {
    /// Dense SwiGLU/GELU FFN.
    Dense(FeedForward),
    /// Mixture of experts.
    Moe(Box<MoeLayer>),
}

/// One transformer block with AdaLN-Zero modulation.
pub struct DiTBlock {
    /// Pre-attention norm.
    pub norm1: RmsNorm,
    /// Pre-FFN norm.
    pub norm2: RmsNorm,
    /// Attention.
    pub attn: Attention,
    /// Feed-forward.
    pub ffn: BlockFfn,
    modulation: Modulation,
}

impl DiTBlock {
    /// Build block `i` from a source.
    pub fn from_source(src: &Weights<'_>, cfg: &NomasConfig, i: usize) -> Result<Self> {
        let path = format!("blocks.{i}");
        let hidden = (cfg.dim as f64 * cfg.mlp_ratio).round() as usize;
        let ffn = match &cfg.moe {
            Some(moe) => BlockFfn::Moe(Box::new(MoeLayer::from_source(
                src,
                &format!("{path}.moe"),
                cfg.dim,
                hidden,
                moe,
            )?)),
            None => BlockFfn::Dense(FeedForward::from_source(
                src,
                &format!("{path}.mlp"),
                cfg.dim,
                hidden,
                cfg.mlp_activation,
            )?),
        };
        Ok(Self {
            norm1: RmsNorm::from_source(src, &format!("{path}.norm1"), cfg.dim, true, 1e-6)?,
            norm2: RmsNorm::from_source(src, &format!("{path}.norm2"), cfg.dim, true, 1e-6)?,
            attn: Attention::from_source(
                src,
                &format!("{path}.attn"),
                cfg.dim,
                cfg.n_heads,
                cfg.qk_norm,
            )?,
            ffn,
            modulation: Modulation::from_source(src, &format!("{path}.modulation"), cfg.dim)?,
        })
    }

    /// Forward one block.
    ///
    /// * `x` — `(B, L, D)` current-token activations.
    /// * `cond` — `(B, D)` pooled condition (timestep + text + camera).
    /// * `rope`, `k_hist`, `v_hist`, `mask` — see [`Attention::forward`].
    ///
    /// Returns the updated activations plus the rotated K/V of the current
    /// tokens (for cache commits).
    pub fn forward(
        &self,
        x: &candle_core::Tensor,
        cond: &candle_core::Tensor,
        rope: &crate::rope::RotaryTable,
        k_hist: &candle_core::Tensor,
        v_hist: &candle_core::Tensor,
        mask: &candle_core::Tensor,
    ) -> Result<(
        candle_core::Tensor,
        candle_core::Tensor,
        candle_core::Tensor,
    )> {
        let ada = self.modulation.forward(cond)?;
        let h = modulate(&self.norm1.forward(x)?, &ada.shift_msa, &ada.scale_msa)?;
        let (q, k, v) = self.attn.project(&h)?;
        let io = self.attn.forward(&q, &k, &v, rope, k_hist, v_hist, mask)?;
        let x = (x + io.x.broadcast_mul(&ada.gate_msa.unsqueeze(1)?)?)?;
        let h = modulate(&self.norm2.forward(&x)?, &ada.shift_mlp, &ada.scale_mlp)?;
        let y = match &self.ffn {
            BlockFfn::Dense(f) => f.forward(&h)?,
            BlockFfn::Moe(m) => m.forward(&h)?,
        };
        let x = (x + y.broadcast_mul(&ada.gate_mlp.unsqueeze(1)?)?)?;
        Ok((x, io.k, io.v))
    }

    /// Parameter count of the whole block.
    pub fn n_params(&self) -> usize {
        let ffn = match &self.ffn {
            BlockFfn::Dense(f) => f.n_params(),
            BlockFfn::Moe(m) => m.n_params(),
        };
        self.norm1.n_params()
            + self.norm2.n_params()
            + self.attn.n_params()
            + self.modulation.n_params()
            + ffn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn block_forward_shapes() {
        let dev = Device::Cpu;
        let mut cfg = NomasConfig::tiny();
        cfg.dim = 32;
        cfg.n_heads = 4;
        cfg.rope_dim_list = Some(vec![2, 2, 4]);
        let block = DiTBlock::from_source(&Weights::random(3, DType::F32, &dev), &cfg, 0).unwrap();
        let x = crate::tensor_utils::deterministic_randn([1, 6, 32], 1, &dev).unwrap();
        let cond = crate::tensor_utils::deterministic_randn([1, 32], 2, &dev).unwrap();
        let rope =
            crate::rope::rotary_table_with_offset(&[2, 2, 4], (1, 2, 3), 0.0, 256.0, &dev).unwrap();
        let zeros = Tensor::zeros((1, 0usize, 4, 8), DType::F32, &dev).unwrap();
        let mask = Tensor::zeros((1usize, 1usize, 6usize, 6usize), DType::F32, &dev).unwrap();
        let (y, k, v) = block
            .forward(&x, &cond, &rope, &zeros, &zeros, &mask)
            .unwrap();
        assert_eq!(y.dims(), vec![1, 6, 32]);
        assert_eq!(k.dims(), vec![1, 6, 4, 8]);
        assert_eq!(v.dims(), vec![1, 6, 4, 8]);
        assert!(y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|t| t.is_finite()));
    }
}
