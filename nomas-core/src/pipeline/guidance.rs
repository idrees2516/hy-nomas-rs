//! Two-axis classifier-free guidance — text and camera
//! (`hymm/utils/camera_cfg.py` semantics).
//!
//! Three forward passes per denoise step:
//!
//! ```text
//! v_uc    : no text, no camera          (unconditional)
//! v_tnc   : text yes, camera default    (text-only)
//! v_cond  : text yes, camera yes        (full condition)
//!
//! v = v_uc + s_text · (v_tnc - v_uc) + s_cam · (v_cond - v_tnc)
//! ```
//!
//! With `s_cam = 0` this collapses to classic single-scale CFG; with both
//! scales at `1.0` it is the raw conditional velocity.

use serde::{Deserialize, Serialize};

/// Guidance scales.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Guidance {
    /// Text CFG scale (`s_text`).
    pub text: f64,
    /// Camera CFG scale (`s_cam`).
    pub camera: f64,
}

impl Default for Guidance {
    fn default() -> Self {
        Self {
            text: 6.0,
            camera: 3.0,
        }
    }
}

impl Guidance {
    /// Raw conditional velocity (both scales `1.0`).
    pub fn none() -> Self {
        Self {
            text: 1.0,
            camera: 1.0,
        }
    }

    /// Whether the unconditional pass can be skipped.
    pub fn is_raw(&self) -> bool {
        self.text == 1.0 && self.camera == 1.0
    }

    /// Whether the camera-ablated pass is needed.
    pub fn needs_camera_pass(&self) -> bool {
        self.camera != 1.0
    }

    /// Combine the (up to) three velocity predictions.
    pub fn combine(
        &self,
        v_cond: &candle_core::Tensor,
        v_text_only: Option<&candle_core::Tensor>,
        v_uncond: Option<&candle_core::Tensor>,
    ) -> candle_core::Result<candle_core::Tensor> {
        let mut v = v_cond.clone();
        if let Some(tnc) = v_text_only {
            if self.camera != 1.0 {
                let delta = (tnc - &v)?;
                v = (&v + delta.affine(self.camera, 0.0)?)?;
            }
        }
        if let Some(uc) = v_uncond {
            if self.text != 1.0 {
                let base = if self.camera != 1.0 { &v } else { v_cond };
                let delta = (base - uc)?;
                v = (uc + delta.affine(self.text, 0.0)?)?;
            }
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn raw_when_both_one() {
        let g = Guidance::none();
        assert!(g.is_raw() && !g.needs_camera_pass());
    }

    #[test]
    fn combine_matches_formula() {
        let dev = Device::Cpu;
        let mk = |v: f32| Tensor::full(v, (1usize, 2usize), &dev).unwrap();
        let cond = mk(4.0);
        let tnc = mk(2.0);
        let uc = mk(1.0);
        let g = Guidance {
            text: 3.0,
            camera: 2.0,
        };
        let v = g.combine(&cond, Some(&tnc), Some(&uc)).unwrap();
        // v = uc + 3*(base - uc), base = cond + 2*(tnc - cond) = 0
        //    = 1 + 3*(0 - 1) = -2
        let vals = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals.iter().all(|x| (x - (-2.0)).abs() < 1e-5), "{vals:?}");
    }

    #[test]
    fn single_scale_collapses_to_classic_cfg() {
        let dev = Device::Cpu;
        let mk = |v: f32| Tensor::full(v, (2usize, 2usize), &dev).unwrap();
        let cond = mk(5.0);
        let uc = mk(1.0);
        let g = Guidance {
            text: 2.0,
            camera: 1.0,
        };
        let v = g.combine(&cond, None, Some(&uc)).unwrap();
        let vals = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals.iter().all(|x| (x - 9.0).abs() < 1e-5), "{vals:?}");
    }
}
