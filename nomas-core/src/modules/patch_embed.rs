//! Latent patchification — the DiT token grid.
//!
//! `PatchEmbed` converts a `(B, C, T, H, W)` latent into
//! `(B, T/pt * H/ph * W/pw, C·pt·ph·pw)` tokens and projects them to the
//! model width. The patchify is a pure reshape/permute (mathematically the
//! upstream `(1, 2, 2)` Conv3d), so no conv kernels are needed.

use candle_core::Tensor;

use crate::Result;

use super::{Lin, Weights};

/// Token-grid patchifier + projection.
pub struct PatchEmbed {
    /// `C·pt·ph·pw -> dim` projection.
    pub proj: Lin,
    patch: (usize, usize, usize),
    channels: usize,
    dim: usize,
}

impl PatchEmbed {
    /// Build from a source.
    pub fn from_source(
        src: &Weights<'_>,
        channels: usize,
        patch: (usize, usize, usize),
        dim: usize,
    ) -> Result<Self> {
        let in_f = channels * patch.0 * patch.1 * patch.2;
        Ok(Self {
            proj: Lin::from_source(src, "patch_embed.proj", in_f, dim, true)?,
            patch,
            channels,
            dim,
        })
    }

    /// Model width.
    pub fn out_dim(&self) -> usize {
        self.dim
    }

    /// Patch grid of a latent `(T, H, W)`.
    pub fn token_grid(&self, t: usize, h: usize, w: usize) -> (usize, usize, usize) {
        (t / self.patch.0, h / self.patch.1, w / self.patch.2)
    }

    /// Extract patch tokens from a latent — `(B, C, T, H, W)` to
    /// `(B, tt·th·tw, C·pt·ph·pw)`.
    pub fn patchify(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = match x.dims() {
            [b, c, t, h, w] => (*b, *c, *t, *h, *w),
            s => crate::bail!("patchify expects (B,C,T,H,W), got {s:?}"),
        };
        let (pt, ph, pw) = self.patch;
        crate::ensure!(
            t % pt == 0 && h % ph == 0 && w % pw == 0,
            "latent {t}x{h}x{w} not divisible by patch {pt}x{ph}x{pw}"
        );
        let (tt, th, tw) = (t / pt, h / ph, w / pw);
        let features = c * pt * ph * pw;
        let x = x
            .reshape(vec![b, c, tt, pt, th, ph, tw, pw])?
            .permute([0, 2, 4, 6, 1, 3, 5, 7])?
            .contiguous()?
            .flatten_from(4)?
            .reshape((b, tt * th * tw, features))?;
        Ok(x)
    }

    /// Inverse of [`Self::patchify`] — `(B, tokens, C·pt·ph·pw)` to
    /// `(B, C, T, H, W)`.
    pub fn unpatchify(&self, x: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (tt, th, tw) = grid;
        let (pt, ph, pw) = self.patch;
        let b = x.dims()[0];
        let (t, h, w) = (tt * pt, th * ph, tw * pw);
        let c = self.channels;
        let x = x
            .reshape(vec![b, tt, th, tw, c, pt, ph, pw])?
            .permute([0, 4, 1, 5, 2, 6, 3, 7])?
            .contiguous()?
            .reshape((b, c, t, h, w))?;
        Ok(x)
    }

    /// Patchify + project.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let tokens = self.patchify(x)?;
        Ok(self.proj.fwd(&tokens)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn patchify_unpatchify_roundtrip() {
        let dev = Device::Cpu;
        let pe = PatchEmbed::from_source(&Weights::random(1, DType::F32, &dev), 32, (1, 2, 2), 64)
            .unwrap();
        let x = crate::tensor_utils::deterministic_randn([1, 32, 3, 8, 10], 3, &dev).unwrap();
        let tokens = pe.patchify(&x).unwrap();
        assert_eq!(tokens.dims(), vec![1, 3 * 4 * 5, 32 * 4]);
        let back = pe.unpatchify(&tokens, (3, 4, 5)).unwrap();
        // patchify+unpatchify is a pure permutation: compare sorted elements.
        let mut a = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut b = back.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(a.len(), b.len());
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-5, "{va} vs {vb}");
        }
    }

    #[test]
    fn rejects_unpatchable_latents() {
        let dev = Device::Cpu;
        let pe = PatchEmbed::from_source(&Weights::random(1, DType::F32, &dev), 32, (1, 2, 2), 8)
            .unwrap();
        let x = Tensor::zeros((1, 32, 3, 7, 8), DType::F32, &dev).unwrap();
        assert!(pe.patchify(&x).is_err());
    }
}
