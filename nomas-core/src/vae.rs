//! Video codec interface — DCAE-V latents + the PixelUMM decoder.
//!
//! Upstream, latents are produced by a finetuned **DCAE-V** (32-channel causal
//! video autoencoder, spatial /8, temporal /4 with I-frames every 4) and
//! decoded by a **PixelUMM** decoder (with an FP8 quantized variant). Those
//! networks are external checkpoints; this crate defines the contract and
//! ships a passthrough codec for tests/demos. Plug a real decoder by
//! implementing [`VideoCodec`].

use serde::{Deserialize, Serialize};

use crate::Result;

/// The DCAE-V latent specification (documented geometry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DcaeVSpec {
    /// Latent channels (32).
    pub channels: usize,
    /// Spatial compression factor (8).
    pub spatial: usize,
    /// Temporal compression factor (4): `latent_frames = (T - 1) / 4 + 1`.
    pub temporal: usize,
}

impl Default for DcaeVSpec {
    fn default() -> Self {
        Self {
            channels: 32,
            spatial: 8,
            temporal: 4,
        }
    }
}

impl DcaeVSpec {
    /// Latent frame count for a clip of `T` pixel frames.
    pub fn latent_frames(&self, t: usize) -> usize {
        if t == 0 {
            0
        } else {
            (t - 1) / self.temporal + 1
        }
    }

    /// Pixel-frame count for `l` latent frames.
    pub fn pixel_frames(&self, l: usize) -> usize {
        if l == 0 {
            0
        } else {
            (l - 1) * self.temporal + 1
        }
    }
}

/// A latent → pixel decoder.
pub trait VideoCodec {
    /// Decode one segment's latents `(B, C, T, H, W)` to frames
    /// `(B, 3, t, H·s, W·s)` with values in `[0, 1]`.
    fn decode_segment(&self, latent: &candle_core::Tensor) -> Result<candle_core::Tensor>;

    /// Spatial upscale factor.
    fn upscale(&self) -> usize;

    /// Temporal upscale factor.
    fn temporal_upscale(&self) -> usize;

    /// Codec name (for reports).
    fn name(&self) -> &str;
}

/// Identity codec for tests and demos: "decodes" latents by channel-wise
/// normalization into a previewable RGB map (first 3 latent channels,
/// `tanh`-squashed to `[0, 1]`). Useful for smoke tests and the interactive
/// demo; **not** a substitute for the PixelUMM decoder.
#[derive(Debug, Clone, Copy, Default)]
pub struct PassthroughCodec;

impl VideoCodec for PassthroughCodec {
    fn decode_segment(&self, latent: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let f = latent.to_dtype(candle_core::DType::F32)?;
        let rgb = f.narrow(1usize, 0, 3.min(f.dims()[1]))?;
        // Per-segment min/max normalization to use the full range.
        let flat = rgb.reshape((rgb.dims().iter().product::<usize>(),))?;
        let min = flat.min(0)?.to_scalar::<f32>()?;
        let max = flat.max(0)?.to_scalar::<f32>()?;
        let span = ((max - min) as f64).max(1e-6);
        let norm = flat.affine(1.0 / span, -(min as f64) / span)?.tanh()?;
        let shape = rgb.dims().to_vec();
        Ok(norm.reshape(shape)?.to_dtype(latent.dtype())?)
    }

    fn upscale(&self) -> usize {
        1
    }

    fn temporal_upscale(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "passthrough-preview"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn latent_frame_geometry() {
        let spec = DcaeVSpec::default();
        assert_eq!(spec.latent_frames(1), 1);
        assert_eq!(spec.latent_frames(33), 9);
        assert_eq!(spec.pixel_frames(9), 33);
        assert_eq!(spec.pixel_frames(spec.latent_frames(121)), 121);
    }

    #[test]
    fn passthrough_is_bounded() {
        let dev = Device::Cpu;
        let x = crate::tensor_utils::deterministic_randn([1, 32, 3, 4, 4], 3, &dev).unwrap();
        let y = PassthroughCodec.decode_segment(&x).unwrap();
        assert_eq!(y.dims(), vec![1, 3, 3, 4, 4]);
        let v = y
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(v.iter().all(|t| (0.0..=1.0).contains(t)), "{v:?}");
    }
}
