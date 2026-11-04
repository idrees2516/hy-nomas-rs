//! Tensor helpers shared by the RoPE, cache and model code.
//!
//! These are small, allocation-conscious ports of the NumPy/PyTorch idioms the
//! upstream model uses (`meshgrid`, `outer`, `repeat_interleave`, deterministic
//! Gaussian sampling with a SplitMix64 + Box-Muller stream).

pub use candle_core::D;
use candle_core::{DType, Device, Result, Tensor};

/// `np.linspace(start, stop, num, endpoint=False)` over one axis.
pub fn linspace_exclusive(start: f64, stop: f64, num: usize, dev: &Device) -> Result<Tensor> {
    if num == 0 {
        return Tensor::from_vec(Vec::<f32>::new(), (0usize,), dev);
    }
    let step = (stop - start) / num as f64;
    let idx = Tensor::arange(0f32, num as f32, dev)?;
    let out = idx.affine(step, start)?;
    out.to_dtype(DType::F32)
}

/// Port of `get_meshgrid_nd` over exactly 3 axes.
///
/// Returns a `(3, T, H, W)` F32 tensor where row 0 is the temporal grid and
/// rows 1/2 are the height/width grids (linspace, endpoint excluded).
pub fn meshgrid_nd(
    start: (f64, f64, f64),
    stop: (f64, f64, f64),
    num: (usize, usize, usize),
    dev: &Device,
) -> Result<Tensor> {
    let (t, h, w) = num;
    let axes = [
        linspace_exclusive(start.0, stop.0, t, dev)?,
        linspace_exclusive(start.1, stop.1, h, dev)?,
        linspace_exclusive(start.2, stop.2, w, dev)?,
    ];
    let total = t * h * w;
    let mut data = Vec::with_capacity(3 * total);
    // Row-major tile matching `torch.meshgrid(..., "ij")` with axis order
    // (T, H, W): an axis value repeats across the *later* axes (block) and
    // cycles across the *earlier* ones.
    let blocks = [h * w, w, 1];
    for (axis_idx, axis) in axes.iter().enumerate() {
        let v = axis.to_vec1::<f32>()?;
        let n = v.len();
        let block = blocks[axis_idx];
        for i in 0..total {
            let idx = (i / block) % n;
            data.push(v[idx]);
        }
    }
    Tensor::from_vec(data, (3usize, t, h, w), dev)
}

/// `torch.outer(a, b)` for 1-D F32 tensors.
pub fn outer(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    a.reshape((a.dims()[0], 1))?
        .matmul(&b.reshape((1, b.dims()[0]))?)
}

/// Deterministic SplitMix64 stream.
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    pub state: u64,
}

impl SplitMix64 {
    /// New stream from a seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Standard uniform in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller (one spare value cached).
    #[inline]
    pub fn next_normal(&mut self, spare: &mut Option<f64>) -> f64 {
        if let Some(s) = spare.take() {
            return s;
        }
        let mut u1 = self.next_f64();
        if u1 < f64::MIN_POSITIVE {
            u1 = f64::MIN_POSITIVE;
        }
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        *spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// Deterministic standard-normal F32 tensor of `shape`, sampled from a
/// SplitMix64 + Box-Muller stream seeded by `seed`.
///
/// Reproducibility across runs, platforms and `candle` versions is guaranteed —
/// this is what the tests, examples and CLI rely on.
pub fn deterministic_randn(
    shape: impl Into<Vec<usize>>,
    seed: u64,
    dev: &Device,
) -> Result<Tensor> {
    deterministic_randn_affine(shape, seed, 0.0, 1.0, dev)
}

/// [`deterministic_randn`] with an affine transform (`x * scale + bias`).
pub fn deterministic_randn_affine(
    shape: impl Into<Vec<usize>>,
    seed: u64,
    mean: f64,
    std: f64,
    dev: &Device,
) -> Result<Tensor> {
    let shape: Vec<usize> = shape.into();
    let n: usize = shape.iter().product();
    let mut rng = SplitMix64::new(seed);
    let mut spare: Option<f64> = None;
    let data: Vec<f32> = (0..n)
        .map(|_| (rng.next_normal(&mut spare) * std + mean) as f32)
        .collect();
    Tensor::from_vec(data, shape, dev)
}

/// Uniform `[low, high)` F32 tensor from the same deterministic stream.
pub fn deterministic_uniform(
    shape: impl Into<Vec<usize>>,
    seed: u64,
    low: f64,
    high: f64,
    dev: &Device,
) -> Result<Tensor> {
    let shape: Vec<usize> = shape.into();
    let n: usize = shape.iter().product();
    let mut rng = SplitMix64::new(seed);
    let data: Vec<f32> = (0..n)
        .map(|_| (low + rng.next_f64() * (high - low)) as f32)
        .collect();
    Tensor::from_vec(data, shape, dev)
}

/// Xavier/Glorot-style initialization for a `(out, in)` weight matrix,
/// deterministic in `seed`.
pub fn xavier(out: usize, inp: usize, seed: u64, dtype: DType, dev: &Device) -> Result<Tensor> {
    let std = (2.0 / (out + inp) as f64).sqrt();
    deterministic_randn_affine([out, inp], seed, 0.0, std, dev)?.to_dtype(dtype)
}

/// Per-token-per-head absolute maximum over the last dim: `(B, L, H, D) ->
/// (B, L, H, 1)` in F32.
pub fn absmax_last_dim(t: &Tensor) -> Result<Tensor> {
    let f = t.to_dtype(DType::F32)?;
    f.abs()?.max_keepdim(D::Minus1)
}

/// Build an I64 index tensor from a `Vec<usize>` (for `index_select`/`gather`).
pub fn idx_tensor(indices: &[usize], dev: &Device) -> Result<Tensor> {
    let data: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
    Tensor::from_vec(data, (indices.len(),), dev)
}

/// Read an I64/F32/whatever integer-ish tensor into `Vec<usize>`.
pub fn to_usize_vec(t: &Tensor) -> Result<Vec<usize>> {
    let v = t.to_dtype(DType::I64)?.to_vec1::<i64>()?;
    Ok(v.into_iter().map(|x| x as usize).collect())
}

/// `repeat_interleave(t, repeats, dim=1)` for the (B, S, ...) token layout.
pub fn repeat_interleave_dim1(t: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(t.clone());
    }
    let shape = t.dims();
    let b = shape[0];
    let s = shape[1];
    let rest = shape[2..].to_vec();
    let mut parts = Vec::with_capacity(s);
    for i in 0..s {
        parts.push(t.narrow(1, i, 1)?.contiguous()?);
    }
    // (B, 1, ...) -> (B, repeats, ...) per slice, then concat.
    let mut expanded = Vec::with_capacity(s);
    for part in &parts {
        let mut wide = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            wide.push(part);
        }
        expanded.push(Tensor::cat(&wide, 1)?);
    }
    let _ = (b, rest);
    Tensor::cat(&expanded, 1)
}

/// Rank-5 host view.
pub type Vec5<T> = Vec<Vec<Vec<Vec<Vec<T>>>>>;

/// Extensions for reading 4-D/5-D tensors on the host (candle ships
/// `to_vec1..3` only).
pub trait ToVecExt {
    /// Read a rank-4 tensor into nested `Vec`s.
    fn to_vec4<T: candle_core::WithDType>(&self) -> candle_core::Result<Vec<Vec<Vec<Vec<T>>>>>;

    /// Read a rank-5 tensor into nested `Vec`s.
    fn to_vec5<T: candle_core::WithDType>(&self) -> candle_core::Result<Vec5<T>>;
}

impl ToVecExt for Tensor {
    fn to_vec4<T: candle_core::WithDType>(&self) -> candle_core::Result<Vec<Vec<Vec<Vec<T>>>>> {
        let dims = self.dims();
        if dims.len() != 4 {
            candle_core::bail!("to_vec4 expects a rank-4 tensor, got {:?}", dims);
        }
        let (a, b, c, d) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = self.flatten_all()?.to_vec1::<T>()?;
        let mut out = Vec::with_capacity(a);
        let mut it = flat.into_iter();
        for _ in 0..a {
            let mut l1 = Vec::with_capacity(b);
            for _ in 0..b {
                let mut l2 = Vec::with_capacity(c);
                for _ in 0..c {
                    let mut l3 = Vec::with_capacity(d);
                    for _ in 0..d {
                        l3.push(it.next().expect("length checked"));
                    }
                    l2.push(l3);
                }
                l1.push(l2);
            }
            out.push(l1);
        }
        Ok(out)
    }

    fn to_vec5<T: candle_core::WithDType>(&self) -> candle_core::Result<Vec5<T>> {
        let dims = self.dims();
        if dims.len() != 5 {
            candle_core::bail!("to_vec5 expects a rank-5 tensor, got {:?}", dims);
        }
        let (a, b, c, d, e) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let flat = self.flatten_all()?.to_vec1::<T>()?;
        let mut out = Vec::with_capacity(a);
        let mut it = flat.into_iter();
        for _ in 0..a {
            let mut l1 = Vec::with_capacity(b);
            for _ in 0..b {
                let mut l2 = Vec::with_capacity(c);
                for _ in 0..c {
                    let mut l3 = Vec::with_capacity(d);
                    for _ in 0..d {
                        let mut l4 = Vec::with_capacity(e);
                        for _ in 0..e {
                            l4.push(it.next().expect("length checked"));
                        }
                        l3.push(l4);
                    }
                    l2.push(l3);
                }
                l1.push(l2);
            }
            out.push(l1);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_reference() {
        let dev = Device::Cpu;
        let t = linspace_exclusive(0.0, 3.0, 3, &dev).unwrap();
        let v = t.to_vec1::<f32>().unwrap();
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
        assert!((v[2] - 2.0).abs() < 1e-6);

        let t2 = linspace_exclusive(4.0, 8.0, 2, &dev).unwrap();
        let v2 = t2.to_vec1::<f32>().unwrap();
        assert!((v2[0] - 4.0).abs() < 1e-6 && (v2[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn meshgrid_matches_torch_ij() {
        let dev = Device::Cpu;
        let g = meshgrid_nd((0.0, 0.0, 0.0), (2.0, 2.0, 3.0), (2, 2, 3), &dev).unwrap();
        let v = g.to_dtype(DType::F32).unwrap().to_vec4::<f32>().unwrap();
        // t axis
        assert_eq!(v[0][0][0][0], 0.0);
        assert_eq!(v[0][1][0][0], 1.0);
        // h axis
        assert_eq!(v[1][0][1][0], 1.0);
        assert_eq!(v[1][0][0][0], 0.0);
        // w axis: linspace(0, 3, 3) = [0, 1, 2]
        assert_eq!(v[2][0][0][1], 1.0);
        assert_eq!(v[2][0][0][2], 2.0);
    }

    #[test]
    fn deterministic_randn_is_reproducible() {
        let dev = Device::Cpu;
        let a = deterministic_randn([2, 5], 7, &dev).unwrap();
        let b = deterministic_randn([2, 5], 7, &dev).unwrap();
        let va = a.to_vec2::<f32>().unwrap();
        let vb = b.to_vec2::<f32>().unwrap();
        assert_eq!(va, vb);
        // sanity: not all zeros, finite, reasonable range
        let flat: Vec<f32> = va.into_iter().flatten().collect();
        assert!(flat.iter().all(|x| x.is_finite()));
        assert!(flat.iter().any(|x| x.abs() > 0.01));
    }

    #[test]
    fn outer_works() {
        let dev = Device::Cpu;
        let a = Tensor::from_vec(vec![1f32, 2f32], (2usize,), &dev).unwrap();
        let b = Tensor::from_vec(vec![3f32, 4f32, 5f32], (3usize,), &dev).unwrap();
        let o = outer(&a, &b).unwrap();
        let v = o.to_vec2::<f32>().unwrap();
        assert_eq!(v, vec![vec![3.0, 4.0, 5.0], vec![6.0, 8.0, 10.0]]);
    }

    #[test]
    fn xavier_scale() {
        let dev = Device::Cpu;
        let t = xavier(64, 128, 3, DType::F32, &dev).unwrap();
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let var = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
        let expected = 2.0 / (64.0 + 128.0);
        assert!(
            (var - expected).abs() / expected < 0.25,
            "var {var} vs {expected}"
        );
    }
}
