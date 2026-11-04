//! AdaLN-Zero modulation — `modulate_layers.py`.
//!
//! One `Linear(dim, 6·dim)` per block produces the six DiT modulation
//! parameters `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
//! gate_mlp)` from the pooled condition vector (timestep + pooled text +
//! pooled camera). The final layer uses the 2-parameter variant.

use candle_core::Result;
use candle_nn::ops::silu;

use super::{Lin, Weights};

/// The six AdaLN parameters for one block.
pub struct AdaSix {
    /// Attention shift `(B, D)`.
    pub shift_msa: candle_core::Tensor,
    /// Attention scale `(B, D)`.
    pub scale_msa: candle_core::Tensor,
    /// Attention output gate `(B, D)`.
    pub gate_msa: candle_core::Tensor,
    /// MLP shift `(B, D)`.
    pub shift_mlp: candle_core::Tensor,
    /// MLP scale `(B, D)`.
    pub scale_mlp: candle_core::Tensor,
    /// MLP output gate `(B, D)`.
    pub gate_mlp: candle_core::Tensor,
}

/// Per-block AdaLN head.
pub struct Modulation {
    mlp: Lin,
}

impl Modulation {
    /// Build from a source.
    pub fn from_source(src: &Weights<'_>, path: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            mlp: Lin::from_source(src, path, dim, 6 * dim, true)?,
        })
    }

    /// Compute the six parameters from a pooled condition `(B, D)`.
    pub fn forward(&self, cond: &candle_core::Tensor) -> Result<AdaSix> {
        let b = cond.dims()[0];
        let out = silu(&self.mlp.fwd(cond)?)?
            .reshape((b, 6, ()))?
            .contiguous()?;
        let pick = |i: usize| -> Result<candle_core::Tensor> {
            out.narrow(1usize, i, 1)?.squeeze(1)?.contiguous()
        };
        Ok(AdaSix {
            shift_msa: pick(0)?,
            scale_msa: pick(1)?,
            gate_msa: pick(2)?,
            shift_mlp: pick(3)?,
            scale_mlp: pick(4)?,
            gate_mlp: pick(5)?,
        })
    }

    /// Parameter count.
    pub fn n_params(&self) -> usize {
        self.mlp.n_params()
    }
}

/// `modulate(x, shift, scale) = x * (1 + scale) + shift` with `(B, D)`
/// parameters broadcast over `(B, S, D)` activations.
pub fn modulate(
    x: &candle_core::Tensor,
    shift: &candle_core::Tensor,
    scale: &candle_core::Tensor,
) -> Result<candle_core::Tensor> {
    let scale = scale.unsqueeze(1)?;
    let shift = shift.unsqueeze(1)?;
    let xs = x.to_dtype(candle_core::DType::F32)?;
    let out = xs.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
    out.to_dtype(x.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn modulate_formula() {
        let dev = Device::Cpu;
        let x = Tensor::ones((1, 3, 4), DType::F32, &dev).unwrap();
        let shift = Tensor::from_vec(vec![1f32, 0.0, 0.0, 0.0], (1usize, 4usize), &dev).unwrap();
        let scale = Tensor::from_vec(vec![0f32, 1.0, 0.0, 0.0], (1usize, 4usize), &dev).unwrap();
        let y = modulate(&x, &shift, &scale).unwrap();
        let v = y.to_vec3::<f32>().unwrap();
        // token 0: x=1 -> 1*(1+0)+1 = 2 ; x=1 with scale 1 -> 2 ; others 1
        assert_eq!(v[0][0][0], 2.0);
        assert_eq!(v[0][0][1], 2.0);
        assert_eq!(v[0][0][2], 1.0);
        assert_eq!(v[0][1][0], 2.0);
    }
}
