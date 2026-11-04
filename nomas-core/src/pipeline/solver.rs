//! Flow-matching solvers — `hymm/pipeline/solvers` + the distilled few-step
//! schedules (`denoise_mode: "ode"`, 4/6/8-step inference).
//!
//! The velocity field is integrated from `t = 1` (pure noise) to `t = 0`
//! (clean latent) with Euler steps; `shift` applies the SD3-style timestep
//! warp `t' = s·t / (1 + (s-1)·t)` that concentrates steps near the noisy end,
//! and `sigma > 0` switches to the additive-noise (SDE-style) update used by
//! some distillation recipes.

use serde::{Deserialize, Serialize};

use crate::config::SolverConfig;
use crate::Result;
use candle_core::{Device, Tensor};

/// A configured solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowSolver {
    /// Number of Euler steps.
    pub steps: usize,
    /// Timestep warp (`1.0` = uniform).
    pub shift: f64,
    /// Additive-noise mixing (`0.0` = ODE).
    pub sigma: f64,
    /// Seed for stochastic updates.
    pub seed: u64,
}

impl From<&SolverConfig> for FlowSolver {
    fn from(cfg: &SolverConfig) -> Self {
        Self {
            steps: cfg.steps,
            shift: cfg.shift,
            sigma: cfg.sigma,
            seed: cfg.seed,
        }
    }
}

impl FlowSolver {
    /// Distilled preset: `steps` ∈ {4, 6, 8}.
    pub fn distilled(steps: usize) -> Self {
        Self {
            steps,
            shift: 1.0,
            sigma: 0.0,
            seed: 0x5EED_0001,
        }
    }

    /// Warp a timestep.
    pub fn warp(&self, t: f64) -> f64 {
        if self.shift == 1.0 {
            t
        } else {
            self.shift * t / (1.0 + (self.shift - 1.0) * t)
        }
    }

    /// Descending warped timesteps `t_0 = warp(1) > … > t_N = warp(0)`
    /// (endpoints included, so `steps + 1` values).
    pub fn timesteps(&self) -> Vec<f64> {
        let mut ts = Vec::with_capacity(self.steps + 1);
        for i in 0..=self.steps {
            let t = 1.0 - i as f64 / self.steps as f64;
            ts.push(self.warp(t));
        }
        ts
    }

    /// One Euler/SDE update from `t` to `t_next`.
    ///
    /// * `x` — current latent `(B, C, T, H, W)`.
    /// * `v` — velocity prediction at `t`.
    pub fn step(
        &self,
        x: &Tensor,
        v: &Tensor,
        t: f64,
        t_next: f64,
        step_index: usize,
        dev: &Device,
    ) -> Result<Tensor> {
        let dt = t_next - t; // negative (t decreases)
        let mut out = (x + v.affine(dt, 0.0)?)?;
        if self.sigma > 0.0 {
            // Additive-noise correction proportional to |dt| (scaled Wiener
            // increment approximation of the probability-flow SDE).
            let noise = crate::tensor_utils::deterministic_randn(
                x.dims().to_vec(),
                self.seed.wrapping_add((step_index as u64) * 0x9E37),
                dev,
            )?
            .to_dtype(x.dtype())?;
            let amp = (self.sigma * dt.abs()).sqrt();
            out = (out + noise.affine(amp, 0.0)?)?;
        }
        Ok(out)
    }

    /// Sample the initial noise at `t = 1`.
    pub fn initial_noise(
        &self,
        shape: (usize, usize, usize, usize, usize),
        dev: &Device,
    ) -> Result<Tensor> {
        Ok(crate::tensor_utils::deterministic_randn(
            [shape.0, shape.1, shape.2, shape.3, shape.4],
            self.seed,
            dev,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn timesteps_descend_with_endpoints() {
        let s = FlowSolver::distilled(4);
        let ts = s.timesteps();
        assert_eq!(ts.len(), 5);
        assert_eq!(ts[0], 1.0);
        assert_eq!(ts[4], 0.0);
        assert!(ts.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn shifted_schedule_pulls_steps_toward_noise() {
        let s = FlowSolver {
            steps: 4,
            shift: 3.0,
            sigma: 0.0,
            seed: 1,
        };
        let ts = s.timesteps();
        assert_eq!(ts[0], 1.0);
        assert_eq!(ts[4], 0.0);
        // warp(1/steps) = warp(0.25) = 3*.25/(1+2*.25) = 0.5 > 0.25
        assert!((ts[3] - 0.5).abs() < 1e-12);
        assert!(ts[3] > 0.25);
        assert!(ts.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn euler_integrates_constant_field_exactly() {
        let dev = Device::Cpu;
        let s = FlowSolver::distilled(2);
        // x' = v with v = 1 -> x(0) = x(1) + integral = x0 + (0 - 1) * 1
        let x = Tensor::ones((1, 2, 1, 2, 2), DType::F32, &dev).unwrap();
        let v = Tensor::ones((1, 2, 1, 2, 2), DType::F32, &dev).unwrap();
        let ts = s.timesteps();
        let mut cur = x;
        for i in 0..ts.len() - 1 {
            cur = s.step(&cur, &v, ts[i], ts[i + 1], i, &dev).unwrap();
        }
        let vals = cur.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // total displacement = sum(dt) = 0 - 1 = -1 -> 1 + (-1) = 0
        assert!(vals.iter().all(|x| x.abs() < 1e-6), "{vals:?}");
    }

    #[test]
    fn sde_mode_stays_finite() {
        let dev = Device::Cpu;
        let s = FlowSolver {
            steps: 3,
            shift: 1.0,
            sigma: 0.5,
            seed: 9,
        };
        let x = Tensor::zeros((1, 1, 1, 2, 2), DType::F32, &dev).unwrap();
        let v = Tensor::ones((1, 1, 1, 2, 2), DType::F32, &dev).unwrap();
        let ts = s.timesteps();
        let y = s.step(&x, &v, ts[0], ts[1], 0, &dev).unwrap();
        assert!(y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|t| t.is_finite()));
    }
}
