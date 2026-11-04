//! Final adaLN layer + linear head back to patch space.

use candle_core::Result;

use super::modulate::modulate;
use super::{Lin, Weights};
use crate::modules::norm::RmsNorm;

/// `adaLN-final`: 2-parameter modulation + zero-initialized linear head.
pub struct FinalLayer {
    norm: RmsNorm,
    /// AdaLN head.
    pub ada: Lin,
    /// Linear head.
    pub linear: Lin,
}

impl FinalLayer {
    /// Build from a source.
    pub fn from_source(src: &Weights<'_>, dim: usize, out_features: usize) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::from_source(src, "final.norm", dim, true, 1e-6)?,
            ada: Lin::from_source(src, "final.ada", dim, 2 * dim, true)?,
            linear: Lin::from_source(src, "final.linear", dim, out_features, true)?,
        })
    }

    /// Forward: `linear(modulate(norm(x), shift, scale))` — `(B, S, D)` to
    /// `(B, S, out_features)`.
    pub fn forward(
        &self,
        x: &candle_core::Tensor,
        cond: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor> {
        let b = cond.dims()[0];
        let ada = self
            .ada
            .fwd(cond)?
            .silu()?
            .reshape((b, 2, ()))?
            .contiguous()?;
        let shift = ada.narrow(1usize, 0, 1)?.squeeze(1)?.contiguous()?;
        let scale = ada.narrow(1usize, 1, 1)?.squeeze(1)?.contiguous()?;
        let h = modulate(&self.norm.forward(x)?, &shift, &scale)?;
        self.linear.fwd(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn final_layer_shape() {
        let dev = Device::Cpu;
        let fin = FinalLayer::from_source(&Weights::random(8, DType::F32, &dev), 32, 128).unwrap();
        let x = crate::tensor_utils::deterministic_randn([1, 6, 32], 1, &dev).unwrap();
        let cond = crate::tensor_utils::deterministic_randn([1, 32], 2, &dev).unwrap();
        let y = fin.forward(&x, &cond).unwrap();
        assert_eq!(y.dims(), vec![1, 6, 128]);
    }
}
