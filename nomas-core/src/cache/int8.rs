//! INT8 KV quantization — port of `int8_kv_cache.py`.
//!
//! Quantization is **per token per head**: for each `(b, l, h)` row the
//! absolute peak over the head dim defines a symmetric scale
//! `s = peak / 127` (`1.0` when the row is all zeros), values are rounded to
//! integers in `[-127, 127]` and stored with an `f32` scale of shape
//! `(B, L, H, 1)`.
//!
//! > **Container note.** `candle` 0.11 has no signed 8-bit dtype, so the
//!> symmetric int8 payload is stored as `u8` with a zero-point of `128`
//!> (`u8 = q + 128`, i.e. values `1..=255`) — the standard uint8 emulation of
//!> int8 with identical numerics.
//!
//! The archive quantizes each completed segment **once**; later frame
//! selections decode the stored payload without ever requantizing.

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};

/// Zero-point of the u8 container (`q + ZERO_POINT`).
const ZERO_POINT: f32 = 128.0;

/// A quantized KV block: `(B, L, H, D)` u8 payloads + `(B, L, H, 1)` f32
/// scales, for both keys and values.
#[derive(Debug, Clone)]
pub struct Int8KvBlock {
    /// Quantized keys `(B, L, H, D)`, `DType::U8` (int8 + 128).
    pub k: Tensor,
    /// Quantized values `(B, L, H, D)`, `DType::U8`.
    pub v: Tensor,
    /// Key scales `(B, L, H, 1)`, `DType::F32`.
    pub k_scale: Tensor,
    /// Value scales `(B, L, H, 1)`, `DType::F32`.
    pub v_scale: Tensor,
}

impl Int8KvBlock {
    /// Allocate an empty block (upstream `make_int8_kv_block`).
    pub fn alloc(
        batch: usize,
        len: usize,
        heads: usize,
        head_dim: usize,
        dev: &Device,
    ) -> Result<Self> {
        Ok(Self {
            k: Tensor::zeros((batch, len, heads, head_dim), DType::U8, dev)?,
            v: Tensor::zeros((batch, len, heads, head_dim), DType::U8, dev)?,
            k_scale: Tensor::zeros((batch, len, heads, 1), DType::F32, dev)?,
            v_scale: Tensor::zeros((batch, len, heads, 1), DType::F32, dev)?,
        })
    }

    /// Host-side byte footprint (payload + scales).
    pub fn nbytes(&self) -> usize {
        fn sz(t: &Tensor) -> usize {
            t.dims().iter().product::<usize>() * t.dtype().size_in_bytes()
        }
        sz(&self.k) + sz(&self.v) + sz(&self.k_scale) + sz(&self.v_scale)
    }
}

/// Quantize one `(B, L, H, D)` F32/BF16/F16 tensor into payload + scale.
fn quantize_one(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let f = x.to_dtype(DType::F32)?;
    let peak = f.abs()?.max_keepdim(candle_core::D::Minus1)?; // (B, L, H, 1)
                                                              // scale = where(peak > 0, peak / 127, 1.0)
    let zeros = peak.zeros_like()?;
    let ones = peak.ones_like()?;
    let positive = peak.gt(&zeros)?;
    let scale = positive.where_cond(&(peak.affine(1.0 / 127.0, 0.0)?), &ones)?;
    let q = f
        .broadcast_div(&scale)?
        .round()?
        .clamp(-127.0, 127.0)?
        .affine(1.0, ZERO_POINT as f64)?
        .clamp(0.0, 255.0)?
        .to_dtype(DType::U8)?;
    Ok((q, scale))
}

/// Quantize a K/V pair — port of `quantize_kv_pair_into` (finite-check
/// included). Input must be `(B, L, H, D)` in F32/BF16/F16.
pub fn quantize_kv_pair(k: &Tensor, v: &Tensor) -> Result<Int8KvBlock> {
    for (name, t) in [("k", k), ("v", v)] {
        if t.dims().len() != 4 {
            return Err(Error::Geometry(format!(
                "int8 quantize expects (B,L,H,D) {name} tensor, got {:?}",
                t.dims()
            )));
        }
        if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Err(Error::Geometry(format!(
                "int8 quantize expects float {name}, got {:?}",
                t.dtype()
            )));
        }
        let f = t.to_dtype(DType::F32)?;
        // NaN <=> x != x; |x| >= MAX catches +-inf.
        let n_nan = f
            .ne(&f.contiguous()?)?
            .to_dtype(DType::U32)?
            .sum_all()?
            .to_scalar::<u32>()?;
        let big = f.ones_like()?.affine(f32::MAX as f64, 0.0)?;
        let n_inf = f
            .abs()?
            .ge(&big)?
            .to_dtype(DType::U32)?
            .sum_all()?
            .to_scalar::<u32>()?;
        if n_nan + n_inf != 0 {
            return Err(Error::Msg(format!(
                "cannot quantize non-finite {name} values"
            )));
        }
    }
    if k.dims() != v.dims() {
        return Err(Error::Geometry(format!(
            "k/v shape mismatch: {:?} vs {:?}",
            k.dims(),
            v.dims()
        )));
    }
    let (kq, ks) = quantize_one(k)?;
    let (vq, vs) = quantize_one(v)?;
    Ok(Int8KvBlock {
        k: kq,
        v: vq,
        k_scale: ks,
        v_scale: vs,
    })
}

/// Dequantize the valid prefix of an INT8 payload — port of `dequantize_kv`.
///
/// Returns `(B, valid_len, H, D)` in `dtype`.
pub fn dequantize(q: &Tensor, scale: &Tensor, valid_len: usize, dtype: DType) -> Result<Tensor> {
    let l = q.dims()[1];
    if valid_len > l {
        return Err(Error::Geometry(format!(
            "invalid valid_len {valid_len} > {l}"
        )));
    }
    if valid_len == 0 {
        let mut shape = q.dims().to_vec();
        shape[1] = 0;
        return Ok(Tensor::zeros(shape, dtype, q.device())?);
    }
    let qv = q.narrow(1usize, 0, valid_len)?;
    let sv = scale.narrow(1usize, 0, valid_len)?;
    let signed = qv.to_dtype(DType::F32)?.affine(1.0, -(ZERO_POINT as f64))?;
    Ok(signed.broadcast_mul(&sv)?.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_utils::ToVecExt;

    #[test]
    fn roundtrip_error_bound() {
        let dev = Device::Cpu;
        let k = crate::tensor_utils::deterministic_randn([1, 16, 3, 32], 11, &dev).unwrap();
        let block = quantize_kv_pair(&k, &k).unwrap();
        assert_eq!(block.k.dtype(), DType::U8);
        let back = dequantize(&block.k, &block.k_scale, 16, DType::F32).unwrap();
        let orig = k.to_vec4::<f32>().unwrap();
        let rec = back.to_vec4::<f32>().unwrap();
        for l in 0..16 {
            for h in 0..3 {
                let mut peak = 0f32;
                for d in 0..32 {
                    peak = peak.max(orig[0][l][h][d].abs());
                }
                let bound = peak / 127.0 / 2.0 + 1e-6;
                for d in 0..32 {
                    let err = (orig[0][l][h][d] - rec[0][l][h][d]).abs();
                    assert!(err <= bound, "err {err} > {bound} at l={l} h={h} d={d}");
                }
            }
        }
    }

    #[test]
    fn zero_rows_get_unit_scale() {
        let dev = Device::Cpu;
        let k = Tensor::zeros((1, 4, 2, 8), DType::F32, &dev).unwrap();
        let block = quantize_kv_pair(&k, &k).unwrap();
        let scales = block.k_scale.to_vec4::<f32>().unwrap();
        for l in 0..4 {
            for h in 0..2 {
                assert_eq!(scales[0][l][h][0], 1.0);
            }
        }
        // zero rows decode back to exactly zero
        let back = dequantize(&block.k, &block.k_scale, 4, DType::F32).unwrap();
        let v = back.to_vec4::<f32>().unwrap();
        assert!(v[0]
            .iter()
            .flat_map(|l| l.iter())
            .flat_map(|h| h.iter())
            .all(|x| x.abs() < 1e-6));
    }

    #[test]
    fn clamps_to_int8_range() {
        let dev = Device::Cpu;
        let k = Tensor::from_vec(
            vec![-1000f32, 0.0, 1000.0, 0.5, -0.5, 1.0, -1.0, 0.0],
            (1, 1, 1, 8),
            &dev,
        )
        .unwrap();
        let block = quantize_kv_pair(&k, &k).unwrap();
        let qs = block
            .k
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(qs.iter().all(|q| (1.0..=255.0).contains(q)), "{qs:?}");
    }

    #[test]
    fn rejects_nan() {
        let dev = Device::Cpu;
        let mut data = vec![0f32; 8];
        data[3] = f32::NAN;
        let k = Tensor::from_vec(data, (1, 1, 1, 8), &dev).unwrap();
        assert!(quantize_kv_pair(&k, &k).is_err());
    }

    #[test]
    fn rejects_inf() {
        let dev = Device::Cpu;
        let mut data = vec![0f32; 8];
        data[2] = f32::INFINITY;
        let k = Tensor::from_vec(data, (1, 1, 1, 8), &dev).unwrap();
        assert!(quantize_kv_pair(&k, &k).is_err());
    }
}
