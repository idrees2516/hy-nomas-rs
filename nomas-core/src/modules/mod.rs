//! DiT building blocks — ports of `hymm/models/dit/modules/*`.

pub mod attention;
pub mod block;
pub mod final_layer;
pub mod mlp;
pub mod modulate;
pub mod moe;
pub mod norm;
pub mod patch_embed;
pub mod text_embed;
pub mod timestep;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::tensor_utils::xavier;
/// Where a module's parameters come from: a loaded checkpoint or deterministic
/// random initialization (tests, examples, smoke demos).
pub enum Weights<'a> {
    /// Checkpoint-backed variables.
    Loaded(VarBuilder<'a>),
    /// Deterministic pseudo-random parameters keyed by the parameter path —
    /// the same seed always yields the same model.
    Random {
        seed: u64,
        dtype: DType,
        dev: Device,
    },
}

impl<'a> Weights<'a> {
    /// Wrap a checkpoint [`VarBuilder`].
    pub fn loaded(vb: VarBuilder<'a>) -> Self {
        Self::Loaded(vb)
    }

    /// Deterministic random weights on `dev`.
    pub fn random(seed: u64, dtype: DType, dev: &Device) -> Self {
        Self::Random {
            seed,
            dtype,
            dev: dev.clone(),
        }
    }

    fn path_seed(seed: u64, path: &str) -> u64 {
        // FNV-1a over the path, mixed with the base seed.
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
        for b in path.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        h
    }
}

/// A dense layer (weight + optional bias) with a stable checkpoint naming
/// convention (`{path}.weight`, `{path}.bias`).
pub struct Lin {
    /// `(out, in)` weight.
    pub weight: Tensor,
    /// `(out,)` bias, when present.
    pub bias: Option<Tensor>,
}

impl Lin {
    /// Load from a checkpoint prefix.
    pub fn load(vb: &VarBuilder<'_>, in_f: usize, out_f: usize, bias: bool) -> Result<Self> {
        let weight = vb.get((out_f, in_f), "weight")?;
        let bias = if bias {
            Some(vb.get(out_f, "bias")?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    /// Deterministic Xavier init (zero bias).
    pub fn random(
        in_f: usize,
        out_f: usize,
        bias: bool,
        seed: u64,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let weight = xavier(out_f, in_f, seed, dtype, dev)?;
        let bias = if bias {
            Some(Tensor::zeros((out_f,), dtype, dev)?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    /// Build from an explicit source.
    pub fn from_source(
        src: &Weights<'_>,
        path: &str,
        in_f: usize,
        out_f: usize,
        bias: bool,
    ) -> Result<Self> {
        match src {
            Weights::Loaded(vb) => Self::load(&vb.pp(path), in_f, out_f, bias),
            Weights::Random { seed, dtype, dev } => Self::random(
                in_f,
                out_f,
                bias,
                Weights::path_seed(*seed, path),
                *dtype,
                dev,
            ),
        }
    }

    /// Apply: `x @ W^T + b` for `x: (..., in)`.
    ///
    /// Leading dimensions are folded into one gemm (candle matmul needs
    /// matching ranks, and a single large gemm beats many small ones).
    pub fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let out_f = self.weight.dims()[0];
        if dims.len() <= 1 {
            let mut out = x.reshape((1usize, ()))?.matmul(&self.weight.t()?)?;
            if let Some(b) = &self.bias {
                out = out.broadcast_add(b)?;
            }
            return out.reshape((out_f,));
        }
        let lead: Vec<usize> = dims[..dims.len() - 1].to_vec();
        let n: usize = lead.iter().product();
        let x2 = x.reshape((n, ()))?;
        let mut out = x2.matmul(&self.weight.t()?)?;
        if let Some(b) = &self.bias {
            out = out.broadcast_add(b)?;
        }
        let mut shape = lead;
        shape.push(out_f);
        out.reshape(shape)
    }

    /// Parameter count.
    pub fn n_params(&self) -> usize {
        let w = self.weight.dims().iter().product::<usize>();
        w + self
            .bias
            .as_ref()
            .map(|b| b.dims().iter().product::<usize>())
            .unwrap_or(0)
    }
}

/// A raw parameter tensor from a source (`embeddings`, learned scalars...).
pub fn param_from_source(
    src: &Weights<'_>,
    path: &str,
    shape: &[usize],
    std: f64,
) -> Result<Tensor> {
    match src {
        Weights::Loaded(vb) => Ok(vb.get(shape.to_vec(), path)?),
        Weights::Random { seed, dtype, dev } => {
            let t = crate::tensor_utils::deterministic_randn_affine(
                shape.to_vec(),
                Weights::path_seed(*seed, path),
                0.0,
                std,
                dev,
            )?
            .to_dtype(*dtype)?;
            Ok(t)
        }
    }
}
