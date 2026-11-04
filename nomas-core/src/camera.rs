//! Camera conditioning: Plücker ray embeddings and per-frame trajectories.
//!
//! Each generated latent frame is paired with a camera (extrinsics + shared
//! intrinsics). Rays through the patch grid are converted to Plücker
//! coordinates `(origin, direction)` and projected through a small MLP into a
//! per-frame camera embedding that modulates the DiT, mirroring the
//! HunyuanVideo/Hy-Nomas camera pathway (`_prepare_camera_keyboard`,
//! `camera_chunk_size: 5`).

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

/// Pinhole intrinsics (normalized focal length in pixels).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Intrinsics {
    /// Focal length along x, in pixels.
    pub fx: f64,
    /// Focal length along y, in pixels.
    pub fy: f64,
    /// Principal point x, in pixels.
    pub cx: f64,
    /// Principal point y, in pixels.
    pub cy: f64,
}

impl Intrinsics {
    /// Square-pixel camera for a `w x h` image.
    pub fn for_resolution(w: usize, h: usize) -> Self {
        let f = 0.5 * w as f64 * 2.0f64.sqrt(); // ~90 deg FOV-ish default
        Self {
            fx: f,
            fy: f,
            cx: w as f64 / 2.0,
            cy: h as f64 / 2.0,
        }
    }
}

/// Camera-to-world rigid pose (column-major 4x4, last row implicit `[0,0,0,1]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraPose {
    /// Column-major 3x4: `[c0, c1, c2, t]` flattened row-major as
    /// `[m00, m10, m20, m01, ...]` — i.e. `m[col*3 + row]`.
    pub cols: [f64; 12],
}

impl CameraPose {
    /// Identity pose (camera at origin looking down -z).
    pub fn identity() -> Self {
        let mut cols = [0.0; 12];
        cols[0] = 1.0; // (0,0)
        cols[4] = 1.0; // (1,1)
        cols[8] = 1.0; // (2,2)
        Self { cols }
    }

    /// Build from rotation columns and translation.
    pub fn from_rt(c0: [f64; 3], c1: [f64; 3], c2: [f64; 3], t: [f64; 3]) -> Self {
        let mut cols = [0.0; 12];
        cols[0..3].copy_from_slice(&c0);
        cols[3..6].copy_from_slice(&c1);
        cols[6..9].copy_from_slice(&c2);
        cols[9..12].copy_from_slice(&t);
        Self { cols }
    }

    /// World-space camera center.
    pub fn center(&self) -> [f64; 3] {
        [self.cols[9], self.cols[10], self.cols[11]]
    }

    /// Rotation column `i` (camera basis vector).
    fn col(&self, i: usize) -> [f64; 3] {
        let o = i * 3;
        [self.cols[o], self.cols[o + 1], self.cols[o + 2]]
    }

    /// Apply the camera-to-world transform to a camera-space direction.
    fn rot(&self, v: [f64; 3]) -> [f64; 3] {
        let c0 = self.col(0);
        let c1 = self.col(1);
        let c2 = self.col(2);
        [
            c0[0] * v[0] + c1[0] * v[1] + c2[0] * v[2],
            c0[1] * v[0] + c1[1] * v[1] + c2[1] * v[2],
            c0[2] * v[0] + c1[2] * v[1] + c2[2] * v[2],
        ]
    }
}

/// A camera trajectory: one pose per latent frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraTrajectory {
    /// Poses, one per latent frame.
    pub poses: Vec<CameraPose>,
    /// Shared intrinsics.
    pub intrinsics: Intrinsics,
}

impl CameraTrajectory {
    /// A static camera with `frames` identity poses.
    pub fn static_camera(frames: usize, intrinsics: Intrinsics) -> Self {
        Self {
            poses: vec![CameraPose::identity(); frames],
            intrinsics,
        }
    }

    /// Dolly forward: camera moves along its viewing axis each frame.
    pub fn dolly_forward(frames: usize, step: f64, intrinsics: Intrinsics) -> Self {
        let mut poses = Vec::with_capacity(frames);
        for f in 0..frames {
            let z = step * f as f64;
            // looking down -z, translating forward means decreasing z.
            poses.push(CameraPose::from_rt(
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -z],
            ));
        }
        Self { poses, intrinsics }
    }

    /// Orbit around the world origin on the xz plane.
    pub fn orbit(frames: usize, radius: f64, total_deg: f64, intrinsics: Intrinsics) -> Self {
        let mut poses = Vec::with_capacity(frames);
        for f in 0..frames {
            let ang = total_deg.to_radians() * f as f64 / frames.max(1) as f64;
            let (s, c) = ang.sin_cos();
            let pos = [radius * s, 0.0, radius * c];
            // Look at the origin: forward = -pos/|pos|; build a stable basis.
            let fwd = [-pos[0], -pos[1], -pos[2]];
            let norm = (fwd[0].powi(2) + fwd[1].powi(2) + fwd[2].powi(2)).sqrt();
            let fwd = [fwd[0] / norm, fwd[1] / norm, fwd[2] / norm];
            let world_up = [0.0, 1.0, 0.0];
            let right = [
                world_up[1] * fwd[2] - world_up[2] * fwd[1],
                world_up[2] * fwd[0] - world_up[0] * fwd[2],
                world_up[0] * fwd[1] - world_up[1] * fwd[0],
            ];
            let rn = (right[0].powi(2) + right[1].powi(2) + right[2].powi(2)).sqrt();
            let right = [right[0] / rn, right[1] / rn, right[2] / rn];
            let up = [
                fwd[1] * right[2] - fwd[2] * right[1],
                fwd[2] * right[0] - fwd[0] * right[2],
                fwd[0] * right[1] - fwd[1] * right[0],
            ];
            // Camera basis columns (camera-to-world): x = right, y = up, z = -fwd.
            poses.push(CameraPose::from_rt(
                right,
                up,
                [-fwd[0], -fwd[1], -fwd[2]],
                pos,
            ));
        }
        Self { poses, intrinsics }
    }
}

/// Plücker rays for one frame: `(grid_h, grid_w, 6)` with `[ox, oy, oz, dx, dy, dz]`,
/// normalized directions, F32.
///
/// Rays are cast through patch-grid cell centers so that the number of camera
/// conditioning points matches the token grid when desired (upstream feeds a
/// per-frame ray map alongside the latent).
pub fn plucker_rays(
    pose: &CameraPose,
    intr: &Intrinsics,
    grid_h: usize,
    grid_w: usize,
    dev: &Device,
) -> Result<Tensor> {
    let c = pose.center();
    let mut data = Vec::with_capacity(grid_h * grid_w * 6);
    for py in 0..grid_h {
        for px in 0..grid_w {
            let u = (px as f64 + 0.5) / grid_w as f64 * 2.0 - 1.0;
            let v = 1.0 - 2.0 * (py as f64 + 0.5) / grid_h as f64;
            // NDC ray in camera space (looking down -z).
            let dir_cam = [u, v, -1.0];
            let world = pose.rot(dir_cam);
            let n = (world[0].powi(2) + world[1].powi(2) + world[2].powi(2)).sqrt();
            let _ = (intr.fx, intr.cx); // focal only sets the FOV of the NDC cone
            data.extend_from_slice(&[c[0] as f32, c[1] as f32, c[2] as f32]);
            data.extend_from_slice(&[
                (world[0] / n) as f32,
                (world[1] / n) as f32,
                (world[2] / n) as f32,
            ]);
        }
    }
    Tensor::from_vec(data, (grid_h, grid_w, 6usize), dev)
}

/// Per-frame pooled Plücker embedding: `(frames, 6)` mean over the grid.
pub fn trajectory_embeddings(
    traj: &CameraTrajectory,
    grid_h: usize,
    grid_w: usize,
    dev: &Device,
) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(traj.poses.len());
    for pose in &traj.poses {
        let rays = plucker_rays(pose, &traj.intrinsics, grid_h, grid_w, dev)?;
        let pooled = rays
            .to_dtype(DType::F32)?
            .mean(1)? // (grid_h, 6)
            .mean(0)?; // (6,)
        rows.push(pooled);
    }
    Tensor::stack(&rows, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_rays_point_down_negative_z() {
        let dev = Device::Cpu;
        let rays = plucker_rays(
            &CameraPose::identity(),
            &Intrinsics::for_resolution(8, 8),
            2,
            2,
            &dev,
        )
        .unwrap();
        let v = rays.to_vec3::<f32>().unwrap();
        // Center of the top-left cell: u=-0.5, v=+0.5 -> dir ∝ (-0.5, 0.5, -1)
        let d: &[f32] = &v[0][0];
        let norm = (d[3] * d[3] + d[4] * d[4] + d[5] * d[5]).sqrt();
        assert!((d[3] / norm - -0.40824).abs() < 1e-4, "dx {}", d[3]);
        assert!((d[5] / norm - -0.81649).abs() < 1e-4, "dz {}", d[5]);
        // Origin at 0
        assert_eq!((d[0], d[1], d[2]), (0.0, 0.0, 0.0));
    }

    #[test]
    fn dolly_changes_origins_only() {
        let dev = Device::Cpu;
        let traj = CameraTrajectory::dolly_forward(3, 0.5, Intrinsics::for_resolution(8, 8));
        let emb = trajectory_embeddings(&traj, 2, 2, &dev).unwrap();
        let v = emb.to_vec2::<f32>().unwrap();
        assert_eq!(v[0][2], 0.0);
        assert!((v[1][2] - -0.5).abs() < 1e-5);
        assert!((v[2][2] - -1.0).abs() < 1e-5);
        // Directions stay identical for a pure translation.
        for d in 3..6 {
            assert!((v[0][d] - v[2][d]).abs() < 1e-6);
        }
    }

    #[test]
    fn orbit_looks_at_origin() {
        let dev = Device::Cpu;
        let traj = CameraTrajectory::orbit(4, 2.0, 90.0, Intrinsics::for_resolution(8, 8));
        let emb = trajectory_embeddings(&traj, 1, 1, &dev).unwrap();
        let v = emb.to_vec2::<f32>().unwrap();
        for f in 0..4 {
            let (ox, oz) = (v[f][0] as f64, v[f][2] as f64);
            // direction should point from camera to origin: -origin/|origin|
            let n = (ox * ox + oz * oz).sqrt();
            assert!((v[f][3] as f64 - (-ox / n)).abs() < 1e-4, "frame {f} dx");
            assert!((v[f][5] as f64 - (-oz / n)).abs() < 1e-4, "frame {f} dz");
        }
    }
}
