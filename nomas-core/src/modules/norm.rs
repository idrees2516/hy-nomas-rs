//! Normalization layers — `norm_layers.py`.
//!
//! QK-norm and block norms are RMSNorm (computed in F32 like the upstream
//! `RMS_norm`), with an optional affine gain.

use candle_core::{DType, Result, Tensor};

use super::{param_from_source, Weights};

/// RMSNorm over the last dimension.
pub struct RmsNorm {
    /// Affine gain of the normalized width, when present.
    pub weight: Option<Tensor>,
    dim: usize,
    eps: f64,
}

impl RmsNorm {
    /// Build from a source; `path` names the affine gain.
    pub fn from_source(
        src: &Weights<'_>,
        path: &str,
        dim: usize,
        affine: bool,
        eps: f64,
    ) -> Result<Self> {
        let weight = if affine {
            Some(param_from_source(
                src,
                &format!("{path}.weight"),
                &[dim],
                1.0,
            )?)
        } else {
            None
        };
        Ok(Self { weight, dim, eps })
    }

    /// Non-affine constructor.
    pub fn no_affine(dim: usize, eps: f64) -> Self {
        Self {
            weight: None,
            dim,
            eps,
        }
    }

    /// Normalized width.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Apply (F32 internally, cast back to the input dtype).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let f = x.to_dtype(DType::F32)?;
        let var = f.powf(2.0)?.mean_keepdim(candle_core::D::Minus1)?;
        let inv = (var + self.eps)?.powf(-0.5)?;
        let mut out = f.broadcast_mul(&inv)?;
        if let Some(w) = &self.weight {
            out = out.broadcast_mul(&w.to_dtype(DType::F32)?)?;
        }
        out.to_dtype(dtype)
    }

    /// Parameter count (affine gain, when present).
    pub fn n_params(&self) -> usize {
        self.weight
            .as_ref()
            .map(|w| w.dims().iter().product::<usize>())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn rms_norm_unit_variance() {
        let dev = Device::Cpu;
        let norm = RmsNorm::no_affine(16, 1e-6);
        let x = crate::tensor_utils::deterministic_randn([2, 5, 16], 7, &dev).unwrap();
        let y = norm.forward(&x).unwrap();
        let yv = y.to_vec3::<f32>().unwrap();
        for row in yv.iter().flat_map(|b| b.iter()) {
            let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
            assert!((ms - 1.0).abs() < 1e-3, "rms {ms}");
        }
    }

    #[test]
    fn zero_vector_stays_finite() {
        let dev = Device::Cpu;
        let norm = RmsNorm::no_affine(8, 1e-6);
        let x = Tensor::zeros((1, 8), DType::F32, &dev).unwrap();
        let y = norm.forward(&x).unwrap();
        assert!(y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
}
