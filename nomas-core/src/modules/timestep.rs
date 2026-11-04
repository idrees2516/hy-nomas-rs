//! Timestep embedding — sinusoidal + 2-layer MLP (`embed_layers.py`).

use candle_core::{Device, Tensor};

use crate::Result;

use super::{Lin, Weights};

/// Embeds diffusion timesteps (and other scalar conditions) into the model
/// width.
pub struct TimestepEmbedder {
    freq_dim: usize,
    /// First MLP layer.
    pub mlp1: Lin,
    /// Second MLP layer.
    pub mlp2: Lin,
}

impl TimestepEmbedder {
    /// Build from a source.
    pub fn from_source(src: &Weights<'_>, path: &str, dim: usize, freq_dim: usize) -> Result<Self> {
        Ok(Self {
            freq_dim,
            mlp1: Lin::from_source(src, &format!("{path}.mlp1"), freq_dim, dim, true)?,
            mlp2: Lin::from_source(src, &format!("{path}.mlp2"), dim, dim, true)?,
        })
    }

    /// Sinusoidal features of `t` (f64 scalars): `(B, freq_dim)`.
    pub fn sinusoidal(t: &[f64], freq_dim: usize, max_period: f64, dev: &Device) -> Result<Tensor> {
        let half = freq_dim / 2;
        crate::ensure!(half >= 1, "freq_dim must be >= 2");
        let mut freqs = Vec::with_capacity(half);
        for i in 0..half {
            freqs.push((-(max_period.ln()) * i as f64 / half as f64).exp() as f32);
        }
        let mut data = Vec::with_capacity(t.len() * freq_dim);
        for &tt in t {
            let args: Vec<f32> = freqs.iter().map(|f| (tt as f32) * f).collect();
            for a in &args {
                data.push(a.cos());
            }
            for a in &args {
                data.push(a.sin());
            }
        }
        Ok(Tensor::from_vec(data, (t.len(), freq_dim), dev)?)
    }

    /// Full embedding for timesteps `t` — `(B, dim)`.
    pub fn forward(&self, t: &[f64], dev: &Device) -> Result<Tensor> {
        let s = Self::sinusoidal(t, self.freq_dim, 10_000.0, dev)?;
        let h = self.mlp1.fwd(&s)?.silu()?;
        Ok(self.mlp2.fwd(&h)?.silu()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn sinusoidal_matches_formula() {
        let dev = Device::Cpu;
        let t = TimestepEmbedder::sinusoidal(&[1.0], 8, 10000.0, &dev).unwrap();
        let v = t.to_vec2::<f32>().unwrap();
        // freqs = exp(-ln(1e4) * i / 4)
        let freq = |i: usize| (-10_000f64.ln() * i as f64 / 4.0).exp();
        for i in 0..4 {
            let f = freq(i) as f32;
            assert!((v[0][i] - f.cos()).abs() < 1e-6);
            assert!((v[0][4 + i] - f.sin()).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_time_is_half_one() {
        let dev = Device::Cpu;
        let t = TimestepEmbedder::sinusoidal(&[0.0], 6, 10000.0, &dev).unwrap();
        let v = t.to_vec2::<f32>().unwrap();
        assert!(v[0].iter().take(3).all(|x| (*x - 1.0).abs() < 1e-6));
        assert!(v[0].iter().skip(3).all(|x| x.abs() < 1e-6));
    }
}
