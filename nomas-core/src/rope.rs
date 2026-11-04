//! 3D Rotary Position Embeddings — a faithful port of
//! `hymm/models/dit/modules/posemb_layers.py`.
//!
//! # Layout
//!
//! RoPE dimensions are split per axis: `[d_t, d_h, d_w]` with
//! `d_t + d_h + d_w = head_dim`. Frequency pairs are grouped per axis: the
//! first `d_t/2` pairs encode the temporal coordinate, the next `d_h/2` the
//! height and the last `d_w/2` the width — exactly matching the upstream
//! concatenation order in `get_nd_rotary_pos_embed`.
//!
//! # Reference semantics preserved
//!
//! * `get_1d_rotary_pos_embed` — `freqs = theta^(-arange(0, d, 2)/d)`, NTK-style
//!   `theta_rescale_factor` (`theta *= f^(d/(d-2))`) and position
//!   `interpolation_factor`.
//! * `get_rotary_pos_embed_with_offset` — segment `N` starts at temporal
//!   positions `[N * block_size, ...]` instead of `[0, ...]`, which is what
//!   makes the *lifelong* KV cache coherent: history keys were rotated with the
//!   positions they had when committed, and never need re-rotation after
//!   frame-subsampled history rebuilds.
//! * `get_packed_rotary_pos_embed` — per-sample grids of differing `(T, H, W)`
//!   concatenated along the token axis.
//! * `get_teacher_forcing_temporal_positions` — LaCT-style positions where all
//!   noise repetitions *and* the clean chunk of a logical chunk share the same
//!   temporal coordinates.

use candle_core::{DType, Device, Result, Tensor};

use crate::error::Error;
use crate::tensor_utils::{meshgrid_nd, outer, SplitMix64};

/// `(cos, sin)` tables of shape `(S, head_dim / 2)` — one frequency *pair* per
/// column; applying them with [`apply_rotary`] reproduces the upstream
/// `repeat_interleave`-based math bit-for-bit in F32.
#[derive(Debug, Clone)]
pub struct RotaryTable {
    /// Cosine per frequency pair: `(S, D/2)`.
    pub cos: Tensor,
    /// Sine per frequency pair: `(S, D/2)`.
    pub sin: Tensor,
}

impl RotaryTable {
    /// Number of token positions covered.
    pub fn seq_len(&self) -> usize {
        self.cos.dims()[0]
    }

    /// Head dimension covered.
    pub fn head_dim(&self) -> usize {
        self.cos.dims()[1] * 2
    }
}

/// 1-D RoPE frequency table — port of `get_1d_rotary_pos_embed` with
/// `use_real=True` but storing one angle per frequency *pair*.
///
/// * `dim` — per-axis RoPE dimension (even).
/// * `pos` — `(S,)` F32 position coordinates.
/// * `theta_rescale_factor` — NTK rescale applied per axis:
///   `theta *= f^(dim/(dim-2))`.
/// * `interpolation_factor` — linear position interpolation.
#[allow(clippy::too_many_arguments)]
fn rope_1d_pairs(
    dim: usize,
    pos: &Tensor,
    theta: f64,
    theta_rescale_factor: f64,
    interpolation_factor: f64,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut theta = theta;
    if theta_rescale_factor != 1.0 && dim > 2 {
        theta *= theta_rescale_factor.powf(dim as f64 / (dim - 2) as f64);
    }
    // freqs = 1 / theta^(arange(0, dim, 2) / dim), shape (dim/2,)
    let half = dim / 2;
    // freqs = 1 / theta^(2i/d) — computed as exp((2i/d) * ln theta) to keep
    // the exponent on the tensor side.
    let exponents: Vec<f32> = (0..half).map(|i| (2 * i) as f32 / dim as f32).collect();
    let freqs = Tensor::from_vec(exponents, (half,), dev)?
        .affine(theta.ln(), 0.0)?
        .exp()?
        .recip()?;
    let pos = pos.to_dtype(DType::F32)?;
    let scaled = pos.affine(interpolation_factor, 0.0)?;
    let angles = outer(&scaled, &freqs)?; // (S, dim/2)
    let cos = angles.cos()?;
    let sin = angles.sin()?;
    Ok((cos, sin))
}

/// Compute `(cos, sin)` tables of shape `(T*H*W, head_dim/2)` for a 3-D grid.
///
/// Direct port of `get_nd_rotary_pos_embed` with `use_real=True` over the axes
/// `(t, h, w)`; per-axis rescale/interpolation factors are applied uniformly.
#[allow(clippy::too_many_arguments)]
pub fn nd_rotary_table(
    rope_dim_list: &[usize; 3],
    start: (f64, f64, f64),
    stop: (f64, f64, f64),
    num: (usize, usize, usize),
    theta: f64,
    theta_rescale_factor: f64,
    interpolation_factor: f64,
    dev: &Device,
) -> Result<RotaryTable> {
    let grid = meshgrid_nd(start, stop, num, dev)?; // (3, T, H, W)
    let dims = num;
    let total = dims.0 * dims.1 * dims.2;
    let mut cos_parts = Vec::with_capacity(3);
    let mut sin_parts = Vec::with_capacity(3);
    for (axis, &dim) in rope_dim_list.iter().enumerate() {
        let coords = grid.get(axis)?.reshape((total,))?.contiguous()?;
        let (c, s) = rope_1d_pairs(
            dim,
            &coords,
            theta,
            theta_rescale_factor,
            interpolation_factor,
            dev,
        )?;
        cos_parts.push(c);
        sin_parts.push(s);
    }
    Ok(RotaryTable {
        cos: Tensor::cat(&cos_parts, 1)?,
        sin: Tensor::cat(&sin_parts, 1)?,
    })
}

/// Convenience wrapper: grid starting at `(t0, 0, 0)` — port of
/// `get_rotary_pos_embed_with_offset`.
///
/// `rope_sizes` is `(tt, th, tw)` in *token* (patched) units; the temporal
/// axis starts at `time_offset` so segment `N` continues the global timeline.
pub fn rotary_table_with_offset(
    rope_dim_list: &[usize; 3],
    rope_sizes: (usize, usize, usize),
    time_offset: f64,
    theta: f64,
    dev: &Device,
) -> Result<RotaryTable> {
    let (tt, th, tw) = rope_sizes;
    nd_rotary_table(
        rope_dim_list,
        (time_offset, 0.0, 0.0),
        (time_offset + tt as f64, th as f64, tw as f64),
        (tt, th, tw),
        theta,
        1.0,
        1.0,
        dev,
    )
}

/// Packed per-sample RoPE — port of `get_packed_rotary_pos_embed`.
///
/// Every packed sample may have its own `(T, H, W)` *latent* shape; each is
/// encoded with an independent zero-based grid and concatenated along tokens.
pub fn packed_rotary_table(
    rope_dim_list: &[usize; 3],
    sample_shapes: &[(usize, usize, usize)],
    theta: f64,
    dev: &Device,
) -> Result<RotaryTable> {
    let mut cos_parts = Vec::with_capacity(sample_shapes.len());
    let mut sin_parts = Vec::with_capacity(sample_shapes.len());
    for &(t, h, w) in sample_shapes {
        let table = nd_rotary_table(
            rope_dim_list,
            (0.0, 0.0, 0.0),
            (t as f64, h as f64, w as f64),
            (t, h, w),
            theta,
            1.0,
            1.0,
            dev,
        )?;
        cos_parts.push(table.cos);
        sin_parts.push(table.sin);
    }
    Ok(RotaryTable {
        cos: Tensor::cat(&cos_parts, 0)?,
        sin: Tensor::cat(&sin_parts, 0)?,
    })
}

/// LaCT-style teacher-forcing temporal positions — port of
/// `get_teacher_forcing_temporal_positions`.
///
/// For the interleaved layout
/// `[noisy_0_r0, …, noisy_0_r{R-1}, clean_0, noisy_1_r0, …, clean_{N-1}]`
/// every sub-chunk of logical chunk `c` receives positions
/// `[c*chunk_size, …]`, i.e. noise repetitions and the clean target share the
/// same temporal coordinates.
pub fn teacher_forcing_temporal_positions(
    num_ar_chunks: usize,
    chunk_size: usize,
    num_repeat: usize,
) -> Vec<f64> {
    let chunks_per_window = num_repeat + 1;
    let total_frames = num_ar_chunks * chunks_per_window * chunk_size;
    let mut positions = vec![0f64; total_frames];
    for chunk_idx in 0..num_ar_chunks {
        let base_pos = (chunk_idx * chunk_size) as f64;
        let window_start = chunk_idx * chunks_per_window * chunk_size;
        for sub_chunk in 0..chunks_per_window {
            let chunk_start = window_start + sub_chunk * chunk_size;
            for frame in 0..chunk_size {
                positions[chunk_start + frame] = base_pos + frame as f64;
            }
        }
    }
    positions
}

/// Build a [`RotaryTable`] from *explicit* per-token axis coordinates — the
/// generalization used by teacher forcing and by history rebuilds where the
/// temporal axis is non-contiguous.
///
/// * `coords` — three `(S,)` F32 tensors `(t, h, w)`, one entry per token.
pub fn rotary_table_from_coords(
    rope_dim_list: &[usize; 3],
    coords: [&Tensor; 3],
    theta: f64,
    theta_rescale_factor: f64,
    interpolation_factor: f64,
    dev: &Device,
) -> Result<RotaryTable> {
    let mut cos_parts = Vec::with_capacity(3);
    let mut sin_parts = Vec::with_capacity(3);
    for (&dim, coord) in rope_dim_list.iter().zip(coords.iter()) {
        let (c, s) = rope_1d_pairs(
            dim,
            coord,
            theta,
            theta_rescale_factor,
            interpolation_factor,
            dev,
        )?;
        cos_parts.push(c);
        sin_parts.push(s);
    }
    Ok(RotaryTable {
        cos: Tensor::cat(&cos_parts, 1)?,
        sin: Tensor::cat(&sin_parts, 1)?,
    })
}

/// Rotate `x` (any leading dims, last dim = head dim) with the pair table.
///
/// `x` is viewed as interleaved `(real, imag)` pairs exactly like
/// `apply_rotary_emb` upstream: `out = x*cos + rotate_half(x)*sin` computed in
/// F32 and cast back to `x`'s dtype.
pub fn apply_rotary(x: &Tensor, table: &RotaryTable) -> crate::Result<Tensor> {
    let shape = x.dims().to_vec();
    let d = *shape.last().expect("non-empty tensor");
    if d != table.head_dim() {
        return Err(Error::Msg(format!(
            "rope head dim mismatch: {} vs {}",
            d,
            table.head_dim()
        )));
    }
    let half = d / 2;

    let xf = x.to_dtype(DType::F32)?;
    let pair_shape: Vec<usize> = shape[..shape.len() - 1]
        .iter()
        .copied()
        .chain([half, 2])
        .collect();
    let xp = xf.reshape(pair_shape)?;
    // The interleave axis is the last one: split (real, imag).
    let xr = xp
        .narrow(candle_core::D::Minus1, 0, 1)?
        .squeeze(candle_core::D::Minus1)?
        .contiguous()?;
    let xi = xp
        .narrow(candle_core::D::Minus1, 1, 1)?
        .squeeze(candle_core::D::Minus1)?
        .contiguous()?;

    // Broadcast (S, D/2) against x's (..., S, H, D/2): the table needs shape
    // (1, S, 1, ..., 1, D/2) — one leading batch axis, then the token axis,
    // then 1s for the head axes in between.
    let mut cos_b = table.cos.clone();
    let mut sin_b = table.sin.clone();
    if shape.len() >= 3 {
        cos_b = cos_b.unsqueeze(0)?;
        sin_b = sin_b.unsqueeze(0)?;
    }
    for _ in 3..shape.len() {
        cos_b = cos_b.unsqueeze(2)?;
        sin_b = sin_b.unsqueeze(2)?;
    }

    let out_r = (xr.broadcast_mul(&cos_b)? - xi.broadcast_mul(&sin_b)?)?;
    let out_i = (xi.broadcast_mul(&cos_b)? + xr.broadcast_mul(&sin_b)?)?;
    // Re-interleave at the end: (..., D/2, 2) -> (..., D).
    let out = Tensor::stack(&[&out_r, &out_i], shape.len())?;
    let out = out.flatten_from(shape.len() - 1)?;
    Ok(out.to_dtype(x.dtype())?)
}

/// Deterministic pseudo-position stream for synthesized codebooks/tests.
pub fn seeded_stream(seed: u64) -> SplitMix64 {
    SplitMix64::new(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_utils::ToVecExt;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn rope_1d_matches_closed_form() {
        let dev = Device::Cpu;
        let theta = 256.0f64;
        let dim = 8usize;
        let pos = Tensor::from_vec(vec![0f32, 1.0, 2.0, 3.0], (4usize,), &dev).unwrap();
        let (cos, sin) = rope_1d_pairs(dim, &pos, theta, 1.0, 1.0, &dev).unwrap();
        let c = cos.to_vec2::<f32>().unwrap();
        let s = sin.to_vec2::<f32>().unwrap();
        // freqs: theta^(-2i/8) for i in 0..4
        for i in 0..4 {
            let freq = theta.powf(-(2.0 * i as f64) / dim as f64);
            for (row, p) in [0f32, 1.0, 2.0, 3.0].iter().enumerate() {
                let ang = *p as f64 * freq;
                assert!(
                    close(c[row][i], ang.cos() as f32, 1e-5),
                    "cos mismatch at {row},{i}"
                );
                assert!(
                    close(s[row][i], ang.sin() as f32, 1e-5),
                    "sin mismatch at {row},{i}"
                );
            }
        }
    }

    #[test]
    fn theta_rescale_formula() {
        let dev = Device::Cpu;
        let pos = Tensor::from_vec(vec![1f32], (1usize,), &dev).unwrap();
        let dim = 8usize;
        let factor = 2.0f64;
        let (cos, _) = rope_1d_pairs(dim, &pos, 256.0, factor, 1.0, &dev).unwrap();
        let rescaled_theta = 256.0 * factor.powf(dim as f64 / (dim - 2) as f64);
        let expected = rescaled_theta.powf(0.0).cos() as f32; // i=0 -> freq 1
        assert!(close(cos.to_vec2::<f32>().unwrap()[0][0], expected, 1e-5));
    }

    #[test]
    fn apply_rotary_identity_at_zero_position() {
        let dev = Device::Cpu;
        let table = nd_rotary_table(
            &[4, 4, 4],
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 2.0),
            (1, 2, 2),
            256.0,
            1.0,
            1.0,
            &dev,
        )
        .unwrap();
        let x = crate::tensor_utils::deterministic_randn([1, 4, 2, 12], 3, &dev).unwrap();
        let out = apply_rotary(&x, &table).unwrap();
        let xv = x.to_vec4::<f32>().unwrap();
        let ov = out.to_vec4::<f32>().unwrap();
        // Token 0 of the grid is (t,h,w)=(0,0,0): RoPE must be the identity
        // there (all rotation angles are zero).
        for h in 0..2 {
            for d in 0..12 {
                assert!(
                    close(xv[0][0][h][d], ov[0][0][h][d], 1e-4),
                    "position all-zero (t=0,h=0,w=0) must be identity; got drift at d={d}"
                );
            }
        }
        assert!(
            ov[0][1]
                .iter()
                .any(|row| row.iter().any(|v| (v - 0.0).abs() > 0.0)),
            "non-zero positions must actually rotate"
        );
    }

    #[test]
    fn apply_rotary_is_orthogonal_rotation() {
        let dev = Device::Cpu;
        let table = nd_rotary_table(
            &[8, 8, 8],
            (0.0, 0.0, 0.0),
            (3.0, 2.0, 4.0),
            (3, 2, 4),
            256.0,
            1.0,
            1.0,
            &dev,
        )
        .unwrap();
        let x = crate::tensor_utils::deterministic_randn([2, 24, 3, 24], 5, &dev).unwrap();
        let out = apply_rotary(&x, &table).unwrap();
        // Norms must be preserved per (…, head) slice.
        let nx = x
            .to_dtype(DType::F32)
            .unwrap()
            .powf(2.0)
            .unwrap()
            .sum_keepdim(3usize)
            .unwrap()
            .sqrt()
            .unwrap();
        let no = out
            .to_dtype(DType::F32)
            .unwrap()
            .powf(2.0)
            .unwrap()
            .sum_keepdim(3usize)
            .unwrap()
            .sqrt()
            .unwrap();
        let nv = nx.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let ov = no.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in nv.iter().zip(ov.iter()) {
            assert!(close(*a, *b, 1e-3), "norm not preserved: {a} vs {b}");
        }
    }

    #[test]
    fn temporal_offset_slices_the_full_table() {
        let dev = Device::Cpu;
        let full = nd_rotary_table(
            &[8, 8, 8],
            (0.0, 0.0, 0.0),
            (6.0, 2.0, 2.0),
            (6, 2, 2),
            256.0,
            1.0,
            1.0,
            &dev,
        )
        .unwrap();
        let offset = rotary_table_with_offset(&[8, 8, 8], (3, 2, 2), 3.0, 256.0, &dev).unwrap();
        let full_cos = full.cos.to_vec2::<f32>().unwrap();
        let off_cos = offset.cos.to_vec2::<f32>().unwrap();
        let full_sin = full.sin.to_vec2::<f32>().unwrap();
        let off_sin = offset.sin.to_vec2::<f32>().unwrap();
        let tpf = 2 * 2; // tokens per temporal step
        for row in 0..3 * tpf {
            for col in 0..12 {
                assert!(close(full_cos[3 * tpf + row][col], off_cos[row][col], 1e-5));
                assert!(close(full_sin[3 * tpf + row][col], off_sin[row][col], 1e-5));
            }
        }
    }

    #[test]
    fn packed_table_is_concatenation() {
        let dev = Device::Cpu;
        let packed = packed_rotary_table(&[4, 4, 4], &[(2, 2, 2), (1, 3, 2)], 256.0, &dev).unwrap();
        let a = nd_rotary_table(
            &[4, 4, 4],
            (0., 0., 0.),
            (2., 2., 2.),
            (2, 2, 2),
            256.0,
            1.0,
            1.0,
            &dev,
        )
        .unwrap();
        let b = nd_rotary_table(
            &[4, 4, 4],
            (0., 0., 0.),
            (1., 3., 2.),
            (1, 3, 2),
            256.0,
            1.0,
            1.0,
            &dev,
        )
        .unwrap();
        let pv = packed.cos.to_vec2::<f32>().unwrap();
        let av = a.cos.to_vec2::<f32>().unwrap();
        let bv = b.cos.to_vec2::<f32>().unwrap();
        assert_eq!(pv.len(), av.len() + bv.len());
        for i in 0..av.len() {
            assert_eq!(pv[i], av[i]);
        }
        for i in 0..bv.len() {
            assert_eq!(pv[av.len() + i], bv[i]);
        }
    }

    #[test]
    fn lact_positions_share_chunk_coordinates() {
        let pos = teacher_forcing_temporal_positions(3, 2, 2);
        // [n0r0 n0r1 clean0 | n1r0 n1r1 clean1 | ...] each group of 6
        assert_eq!(pos.len(), 3 * 3 * 2);
        assert_eq!(&pos[0..2], &[0.0, 1.0]); // noisy_0_r0
        assert_eq!(&pos[2..4], &[0.0, 1.0]); // noisy_0_r1
        assert_eq!(&pos[4..6], &[0.0, 1.0]); // clean_0
        assert_eq!(&pos[6..8], &[2.0, 3.0]); // noisy_1_r0
        assert_eq!(&pos[10..12], &[2.0, 3.0]); // clean_1
        assert_eq!(&pos[16..18], &[4.0, 5.0]); // clean_2
    }
}
