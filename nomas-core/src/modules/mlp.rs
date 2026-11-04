//! Feed-forward networks — `mlp_layers.py`.

use candle_core::Result;

use super::{Lin, Weights};
use crate::config::Activation;

/// Gated/un-gated FFN switch.
pub enum FeedForward {
    /// `fc2(gelu(fc1(x)))` (HunyuanVideo-style).
    Mlp { fc1: Lin, fc2: Lin },
    /// `down(silu(gate(x)) * up(x))` — SwiGLU.
    SwiGlu { gate: Lin, up: Lin, down: Lin },
}

impl FeedForward {
    /// Build from a source: `dim -> hidden` (x2 for SwiGLU).
    pub fn from_source(
        src: &Weights<'_>,
        path: &str,
        dim: usize,
        hidden: usize,
        act: Activation,
    ) -> Result<Self> {
        Ok(match act {
            Activation::GeluErf => FeedForward::Mlp {
                fc1: Lin::from_source(src, &format!("{path}.fc1"), dim, hidden, true)?,
                fc2: Lin::from_source(src, &format!("{path}.fc2"), hidden, dim, true)?,
            },
            Activation::Silu => FeedForward::SwiGlu {
                gate: Lin::from_source(src, &format!("{path}.gate"), dim, hidden, true)?,
                up: Lin::from_source(src, &format!("{path}.up"), dim, hidden, true)?,
                down: Lin::from_source(src, &format!("{path}.down"), hidden, dim, true)?,
            },
        })
    }

    /// Apply.
    pub fn forward(&self, x: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        match self {
            FeedForward::Mlp { fc1, fc2 } => fc2.fwd(&fc1.fwd(x)?.gelu_erf()?),
            FeedForward::SwiGlu { gate, up, down } => {
                let g = gate.fwd(x)?.silu()?;
                let u = up.fwd(x)?;
                down.fwd(&g.broadcast_mul(&u)?)
            }
        }
    }

    /// Parameter count.
    pub fn n_params(&self) -> usize {
        match self {
            FeedForward::Mlp { fc1, fc2 } => fc1.n_params() + fc2.n_params(),
            FeedForward::SwiGlu { gate, up, down } => {
                gate.n_params() + up.n_params() + down.n_params()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn swiglu_shape() {
        let dev = Device::Cpu;
        let ffn = FeedForward::from_source(
            &Weights::random(2, DType::F32, &dev),
            "blocks.0.mlp",
            32,
            64,
            Activation::Silu,
        )
        .unwrap();
        let x = crate::tensor_utils::deterministic_randn([1, 7, 32], 3, &dev).unwrap();
        let y = ffn.forward(&x).unwrap();
        assert_eq!(y.dims(), vec![1, 7, 32]);
        assert!(y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn zero_weight_zero_output() {
        let dev = Device::Cpu;
        let ffn = FeedForward::from_source(
            &Weights::random(9, DType::F32, &dev),
            "blocks.0.mlp",
            8,
            16,
            Activation::GeluErf,
        )
        .unwrap();
        // zero the weights via a zero input
        let x = candle_core::Tensor::zeros((1, 3, 8), DType::F32, &dev).unwrap();
        let y = ffn.forward(&x).unwrap();
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|t| t.abs() < 1e-6));
    }
}
