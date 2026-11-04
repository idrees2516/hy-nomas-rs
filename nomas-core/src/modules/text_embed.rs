//! Text conditioning — projector + pluggable encoders.
//!
//! The upstream text pathway encodes prompts with a large LM (LLaMA-style
//! `text_encoder/`) and projects the last hidden states into the DiT width.
//! `nomas-core` keeps that contract via the [`TextEncoder`] trait: plug in a
//! candle Llama (or any encoder) for real checkpoints; the built-in
//! [`HashTextEncoder`] provides deterministic, dependency-free embeddings for
//! tests, examples and the interactive demo.

use candle_core::{DType, Device, Result, Tensor};

use super::{param_from_source, Lin, Weights};
use crate::tensor_utils::SplitMix64;

/// Projects text features into the DiT width and appends an optional learned
/// separator token.
pub struct TextProjector {
    /// Projection layer.
    pub proj: Lin,
    separator: Option<Tensor>,
    dim: usize,
}

impl TextProjector {
    /// Build from a source.
    pub fn from_source(
        src: &Weights<'_>,
        text_dim: usize,
        dim: usize,
        separator: bool,
    ) -> Result<Self> {
        Ok(Self {
            proj: Lin::from_source(src, "text_proj", text_dim, dim, true)?,
            separator: if separator {
                Some(param_from_source(src, "text_separator", &[dim], 0.02)?)
            } else {
                None
            },
            dim,
        })
    }

    /// Project `(B, L, D_text)` to `(B, L (+1), D)`; the separator (when
    /// enabled) is appended as the last text token.
    pub fn forward(&self, text: &Tensor) -> Result<Tensor> {
        let mut out = self.proj.fwd(text)?;
        if let Some(sep) = &self.separator {
            let b = out.dims()[0];
            let sep = sep.reshape((1, 1, self.dim))?.expand((b, 1, self.dim))?;
            out = Tensor::cat(&[&out, &sep], 1)?;
        }
        Ok(out)
    }
}

/// A text encoder producing `(1, L, text_dim)` features for a prompt.
pub trait TextEncoder {
    /// Encode one prompt.
    fn encode(&self, prompt: &str) -> Result<Tensor>;

    /// The encoder's output width.
    fn dim(&self) -> usize;

    /// Encode with the null (empty) prompt — the CFG anchor.
    fn encode_null(&self) -> Result<Tensor> {
        self.encode("")
    }
}

/// Deterministic feature encoder for tests and demos: FNV-hashed tokens index
/// rows of a fixed random matrix, and two causal mixing passes aggregate
/// context (enough structure for smoke tests, no external weights).
pub struct HashTextEncoder {
    table: Tensor,
    dim: usize,
    vocab: usize,
    mix: Lin,
}

impl HashTextEncoder {
    /// Build with a deterministic table.
    pub fn new(dim: usize, vocab: usize, seed: u64, dev: &Device) -> Result<Self> {
        let table = crate::tensor_utils::deterministic_randn([vocab, dim], seed, dev)?;
        let mix = Lin::random(dim, dim, false, seed ^ 0xA11CE, DType::F32, dev)?;
        Ok(Self {
            table,
            dim,
            vocab,
            mix,
        })
    }

    fn token_ids(&self, prompt: &str, dev: &Device) -> Result<Tensor> {
        let tokens: Vec<usize> = prompt
            .split_whitespace()
            .filter(|t| !t.is_empty())
            .map(|t| {
                let mut h = 0x811c_9dc5u64;
                for b in t.as_bytes() {
                    h ^= *b as u64;
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
                (h % self.vocab as u64) as usize
            })
            .collect();
        if tokens.is_empty() {
            // null prompt -> a single dedicated slot (0)
            return Tensor::from_vec(vec![0i64], (1usize,), dev);
        }
        let n = tokens.len();
        let ids: Vec<i64> = tokens.into_iter().map(|t| t as i64).collect();
        Tensor::from_vec(ids, (n,), dev)
    }
}

impl TextEncoder for HashTextEncoder {
    fn encode(&self, prompt: &str) -> Result<Tensor> {
        let dev = self.table.device();
        let ids = self.token_ids(prompt, dev)?;
        let mut emb = self.table.index_select(&ids, 0)?.unsqueeze(0)?; // (1, L, D)
                                                                       // causal mixing: E'_t = E_t + mean(E_0..=t)
        let (b, l, d) = (emb.dims()[0], emb.dims()[1], emb.dims()[2]);
        let mut rows = Vec::with_capacity(l);
        let mut acc = Tensor::zeros((b, 1, d), DType::F32, dev)?;
        for i in 0..l {
            acc = (acc.broadcast_add(&emb.narrow(1usize, i, 1)?))?;
            let n = (i + 1) as f64;
            rows.push(acc.affine(1.0 / n, 0.0)?.contiguous()?);
        }
        let mixed = Tensor::cat(&rows, 1)?;
        emb = self.mix.fwd(&mixed)?.tanh()?;
        Ok(emb)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Deterministic stream helper (exposed for tests).
pub fn hash_seed(prompt: &str) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64;
    for b in prompt.as_bytes() {
        h = h.wrapping_mul(31).wrapping_add(*b as u64);
    }
    h
}

/// Re-exported for callers that want the same stream type.
pub type EncoderStream = SplitMix64;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn deterministic_and_distinct() {
        let dev = Device::Cpu;
        let enc = HashTextEncoder::new(16, 128, 7, &dev).unwrap();
        let a = enc.encode("a red fox jumps").unwrap();
        let b = enc.encode("a red fox jumps").unwrap();
        let c = enc.encode("a blue whale swims").unwrap();
        assert_eq!(a.dims(), b.dims());
        assert!(
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                == b.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert!(
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                != c.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn null_prompt_is_single_token() {
        let dev = Device::Cpu;
        let enc = HashTextEncoder::new(8, 32, 3, &dev).unwrap();
        let n = enc.encode_null().unwrap();
        assert_eq!(n.dims(), vec![1, 1, 8]);
    }

    #[test]
    fn projector_appends_separator() {
        let dev = Device::Cpu;
        let proj = TextProjector::from_source(&Weights::random(5, DType::F32, &dev), 16, 32, true)
            .unwrap();
        let x = crate::tensor_utils::deterministic_randn([1, 4, 16], 2, &dev).unwrap();
        let y = proj.forward(&x).unwrap();
        assert_eq!(y.dims(), vec![1, 5, 32]);
    }
}
