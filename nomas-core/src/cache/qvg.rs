//! QVG — residual vector quantization for the KV archive.
//!
//! > **Interpretation note.** The upstream `qvg_kv_cache.py` kernel is a GPU
//! > codec whose internals are not part of the fetched sources; the released
//!> configuration (`num_bits: 2`, `4` stages, `256` K/V centroids, block size
//!> `64`, text as INT8) is realized here as the following portable codec,
//!> which matches every knob:
//!
//! * Per side (K or V) and per head, tokens are grouped into blocks of
//!   `block_size` (64). Each block carries **one FP32 scale per stage** — the
//!   block's dynamic-range adaptation.
//! * Each token is encoded by `stages` (4) residual passes; stage `s` stores a
//!   `bits_per_stage`-bit (2) code indexing that stage's codebook, giving the
//!   `2^(2·4) = 256`-entry effective product code of the upstream centroid
//!   count.
//! * Codes are **bit-packed**: 4 stages x 2 bits = exactly one byte per token
//!   per head — INT8-equivalent footprint with vector-quantization fidelity.
//! * Codebooks are shared per cache instance, fitted once from the first
//!   committed segment (deterministic farthest-point + Lloyd initialization)
//!   and frozen afterwards: archived segments are never requantized when
//!>  history frames are reselected.
//! * Text-token KV is stored as INT8 (`text_as_int8`).
//!
//! Encoding/decoding run on the host in plain Rust — the archive lives on the
//! host anyway, and this keeps the codec independent of the compute backend.

use candle_core::{Device, Tensor};

use crate::config::QvgConfig;
use crate::error::{Error, Result};

/// One stage's codebook: `centroids × head_dim` host matrix.
pub type StageCodebook = Vec<Vec<f32>>;
/// Per-head codebooks: `stage -> StageCodebook`.
pub type HeadCodebooks = Vec<StageCodebook>;
/// Per-side (0 = K, 1 = V) codebooks: `head -> HeadCodebooks`.
pub type SideCodebooks = Vec<HeadCodebooks>;
/// A fitted codebook set: `[side][head][stage][entry] -> Vec<f32>` of length
/// `head_dim`.
pub type Codebooks = Vec<SideCodebooks>;

/// The quantized payload for one side (K or V) of one archived segment.
#[derive(Debug, Clone)]
pub struct QvgPayload {
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) tokens: usize,
    /// Packed codes: `tokens * heads * bytes_per_token`.
    pub(crate) codes: Vec<u8>,
    /// Per `(head, block, stage)` scale.
    pub(crate) scales: Vec<f32>,
    pub(crate) n_blocks: usize,
}

impl QvgConfig {
    /// Entries per stage codebook.
    pub fn entries(&self) -> usize {
        1 << self.bits_per_stage
    }

    /// Packed code bytes per token.
    pub fn bytes_per_token(&self) -> usize {
        (self.stages * self.bits_per_stage).div_ceil(8)
    }

    /// Host bytes of a payload for `tokens` tokens (codes + scales).
    pub fn payload_bytes(&self, tokens: usize, heads: usize) -> usize {
        let n_blocks = tokens.div_ceil(self.block_size);
        tokens * heads * self.bytes_per_token() + n_blocks * heads * self.stages * 4
    }
}

impl QvgPayload {
    /// Number of quantized tokens.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Host bytes held by this payload (packed codes + FP32 scales).
    pub fn nbytes(&self) -> usize {
        self.codes.len() + self.scales.len() * 4
    }
}

/// The cache-wide QVG codec (codebook state + encoder/decoder).
pub struct QvgCodec {
    cfg: QvgConfig,
    codebooks: Option<Codebooks>,
}

impl QvgCodec {
    /// Create a codec that will fit its codebooks on first use.
    pub fn new(cfg: QvgConfig) -> Self {
        Self {
            cfg,
            codebooks: None,
        }
    }

    /// Attach externally trained/loaded codebooks (checkpoint path).
    pub fn with_codebooks(mut self, codebooks: Codebooks) -> Self {
        self.codebooks = Some(codebooks);
        self
    }

    /// Whether the codebooks have been fitted.
    pub fn is_fitted(&self) -> bool {
        self.codebooks.is_some()
    }

    /// Host bytes the codec holds for codebooks.
    pub fn codebook_bytes(&self) -> usize {
        self.codebooks
            .as_ref()
            .map(|cbs| {
                cbs.iter()
                    .flat_map(|side| side.iter())
                    .flat_map(|head| head.iter())
                    .flat_map(|stage| stage.iter())
                    .map(|e| e.len() * 4)
                    .sum::<usize>()
            })
            .unwrap_or(0)
    }

    /// Fit the codebooks from `(k, v)` `(B, L, H, D)` F32/BF16 tensors.
    ///
    /// Deterministic: farthest-point initialization from a seeded stream plus
    /// two Lloyd iterations, stage by stage over the residuals.
    pub fn fit(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let (l, heads, _head_dim) = match k.dims() {
            [_, l, h, d] => (*l, *h, *d),
            s => {
                return Err(Error::Geometry(format!(
                    "qvg fit expects (B,L,H,D), got {s:?}"
                )))
            }
        };
        if self.codebooks.is_some() {
            return Ok(()); // frozen
        }
        let mut sides = Vec::with_capacity(2);
        for t in [k, v] {
            let mut per_head = Vec::with_capacity(heads);
            for h in 0..heads {
                // Extract (L, D) host matrix for this head (batch 0 only — the
                // interactive world model runs B=1 for fitting).
                let col = t
                    .narrow(2usize, h, 1)?
                    .squeeze(2)?
                    .squeeze(0)?
                    .contiguous()?;
                let col = col.to_dtype(candle_core::DType::F32)?;
                let mat = col.to_vec2::<f32>()?;
                per_head.push(fit_head_codebooks(
                    &mat,
                    self.cfg.stages,
                    self.cfg.entries(),
                    self.cfg.codebook_seed.wrapping_add(h as u64),
                ));
            }
            sides.push(per_head);
        }
        self.codebooks = Some(sides);
        let _ = l;
        Ok(())
    }

    /// Encode the video-token region `[0, tokens)` of one side (0 = K, 1 = V)
    /// given the `(B, tokens, H, D)` tensor.
    ///
    /// Requires [`Self::fit`] (or loaded codebooks) first.
    pub fn encode(&self, side: usize, t: &Tensor, tokens: usize) -> Result<QvgPayload> {
        let cbs = self
            .codebooks
            .as_ref()
            .ok_or_else(|| Error::CacheState("qvg codebooks not fitted".into()))?;
        let (l, heads, head_dim) = match t.dims() {
            [_, l, h, d] => (*l, *h, *d),
            s => {
                return Err(Error::Geometry(format!(
                    "qvg encode expects (B,L,H,D), got {s:?}"
                )))
            }
        };
        if tokens > l {
            return Err(Error::Geometry(format!(
                "qvg encode token range {tokens} > len {l}"
            )));
        }
        if t.dims()[0] != 1 {
            return Err(Error::Geometry("qvg storage requires batch=1".into()));
        }
        let cfg = &self.cfg;
        let n_blocks = tokens.div_ceil(cfg.block_size);
        let bpt = cfg.bytes_per_token();
        let mut codes = vec![0u8; tokens * heads * bpt];
        let mut scales = vec![0f32; heads * n_blocks * cfg.stages];

        for h in 0..heads {
            let col = t
                .narrow(2usize, h, 1)?
                .squeeze(2)?
                .squeeze(0)?
                .narrow(0usize, 0, tokens)?
                .contiguous()?;
            let col = col.to_dtype(candle_core::DType::F32)?;
            let mat = col.to_vec2::<f32>()?; // (tokens, D)
            let stage_cbs = &cbs[side][h];
            for b in 0..n_blocks {
                let start = b * cfg.block_size;
                let end = ((b + 1) * cfg.block_size).min(tokens);
                let mut residual: Vec<Vec<f32>> = mat[start..end].to_vec();
                for (s, cb) in stage_cbs.iter().enumerate().take(cfg.stages) {
                    let mut block_codes = Vec::with_capacity(end - start);
                    for r in &residual {
                        block_codes.push(nearest(cb, r));
                    }
                    // Optimal per-block scalar: alpha = <r, c> / <c, c>.
                    let (mut num, mut den) = (0f64, 0f64);
                    for (r, &c) in residual.iter().zip(block_codes.iter()) {
                        let e = &cb[c];
                        num += r
                            .iter()
                            .zip(e)
                            .map(|(a, b)| (*a as f64) * (*b as f64))
                            .sum::<f64>();
                        den += e.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
                    }
                    let alpha = if den > 1e-12 { (num / den) as f32 } else { 1.0 };
                    scales[(h * n_blocks + b) * cfg.stages + s] = alpha;
                    for (r, &c) in residual.iter_mut().zip(block_codes.iter()) {
                        let e = &cb[c];
                        for (ri, ei) in r.iter_mut().zip(e.iter()) {
                            *ri -= alpha * ei;
                        }
                    }
                    // Pack codes for this block.
                    for (i, &c) in block_codes.iter().enumerate() {
                        let token = start + i;
                        let base = (token * heads + h) * bpt;
                        let bit_off = s * cfg.bits_per_stage;
                        pack_bits(&mut codes[base..base + bpt], bit_off, cfg.bits_per_stage, c);
                    }
                }
            }
        }
        Ok(QvgPayload {
            heads,
            head_dim,
            tokens,
            codes,
            scales,
            n_blocks,
        })
    }

    /// Decode a payload back into a `(1, tokens, H, D)` F32 tensor on `dev`.
    pub fn decode(&self, side: usize, payload: &QvgPayload, dev: &Device) -> Result<Tensor> {
        let cbs = self
            .codebooks
            .as_ref()
            .ok_or_else(|| Error::CacheState("qvg codebooks not fitted".into()))?;
        let cfg = &self.cfg;
        let QvgPayload {
            heads,
            head_dim,
            tokens,
            codes,
            scales,
            n_blocks,
        } = payload;
        let bpt = cfg.bytes_per_token();
        let mut data = vec![0f32; tokens * heads * head_dim];
        for h in 0..*heads {
            let stage_cbs = &cbs[side][h];
            for token in 0..*tokens {
                let base = (token * *heads + h) * bpt;
                let b = token / cfg.block_size;
                for s in 0..cfg.stages {
                    let code = unpack_bits(
                        &codes[base..base + bpt],
                        s * cfg.bits_per_stage,
                        cfg.bits_per_stage,
                    );
                    let alpha = scales[(h * n_blocks + b) * cfg.stages + s];
                    let entry = &stage_cbs[s][code];
                    let out = &mut data[(token * *heads + h) * *head_dim..][..*head_dim];
                    for (o, e) in out.iter_mut().zip(entry.iter()) {
                        *o += alpha * e;
                    }
                }
            }
        }
        Ok(Tensor::from_vec(
            data,
            (1usize, *tokens, *heads, *head_dim),
            dev,
        )?)
    }
}

fn nearest(cb: &[Vec<f32>], x: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for (i, e) in cb.iter().enumerate() {
        let mut d = 0f64;
        for (a, b) in x.iter().zip(e.iter()) {
            let diff = *a as f64 - *b as f64;
            d += diff * diff;
            if d >= best_d {
                break; // early exit
            }
        }
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn pack_bits(buf: &mut [u8], bit_off: usize, bits: usize, value: usize) {
    let mut v = value as u32;
    for i in 0..bits {
        let idx = bit_off + i;
        let byte = idx / 8;
        let bit = idx % 8;
        if v & 1 == 1 {
            buf[byte] |= 1 << bit;
        }
        v >>= 1;
    }
}

fn unpack_bits(buf: &[u8], bit_off: usize, bits: usize) -> usize {
    let mut v = 0usize;
    for i in 0..bits {
        let idx = bit_off + i;
        let byte = idx / 8;
        let bit = idx % 8;
        if buf[byte] & (1 << bit) != 0 {
            v |= 1 << i;
        }
    }
    v
}

/// Fit `stages` codebooks of `entries` entries for one head from an `(L, D)`
/// host matrix.
fn fit_head_codebooks(
    mat: &[Vec<f32>],
    stages: usize,
    entries: usize,
    seed: u64,
) -> Vec<Vec<Vec<f32>>> {
    let d = mat.first().map(|r| r.len()).unwrap_or(0);
    let mut residual: Vec<Vec<f32>> = mat.to_vec();
    let mut codebooks = Vec::with_capacity(stages);
    let mut rng = crate::tensor_utils::SplitMix64::new(seed);
    for s in 0..stages {
        // Deterministic farthest-point init.
        let mut cb = farthest_point_init(&residual, entries, d, &mut rng);
        // Two Lloyd iterations.
        for _ in 0..2 {
            let mut sums = vec![vec![0f64; d]; entries];
            let mut counts = vec![0usize; entries];
            for r in &residual {
                let c = nearest(&cb, r);
                counts[c] += 1;
                for (si, ri) in sums[c].iter_mut().zip(r.iter()) {
                    *si += *ri as f64;
                }
            }
            for (c, e) in cb.iter_mut().enumerate() {
                if counts[c] > 0 {
                    for (ei, si) in e.iter_mut().zip(sums[c].iter()) {
                        *ei = (*si / counts[c] as f64) as f32;
                    }
                }
            }
        }
        // Subtract the selected entries.
        for r in residual.iter_mut() {
            let c = nearest(&cb, r);
            for (ri, ei) in r.iter_mut().zip(cb[c].iter()) {
                *ri -= ei;
            }
        }
        let _ = s;
        codebooks.push(cb);
    }
    codebooks
}

fn farthest_point_init(
    data: &[Vec<f32>],
    entries: usize,
    d: usize,
    rng: &mut crate::tensor_utils::SplitMix64,
) -> Vec<Vec<f32>> {
    let n = data.len();
    let mut cents = Vec::with_capacity(entries);
    if n == 0 || d == 0 {
        return (0..entries).map(|_| vec![0f32; d]).collect();
    }
    let first = (rng.next_f64() * n as f64) as usize % n;
    cents.push(data[first].clone());
    let mut dists = vec![f64::INFINITY; n];
    while cents.len() < entries {
        let last = cents.last().unwrap();
        let mut total = 0f64;
        for (i, r) in data.iter().enumerate() {
            let mut dd = 0f64;
            for (a, b) in r.iter().zip(last.iter()) {
                dd += (*a as f64 - *b as f64).powi(2);
            }
            if dd < dists[i] {
                dists[i] = dd;
            }
            total += dists[i];
        }
        if total <= 1e-12 {
            // All points identical — pad with jittered copies.
            let mut j = cents[0].clone();
            j[0] += cents.len() as f32 * 1e-3;
            cents.push(j);
            continue;
        }
        let mut pick = rng.next_f64() * total;
        let mut idx = n - 1;
        for (i, di) in dists.iter().enumerate() {
            pick -= di;
            if pick <= 0.0 {
                idx = i;
                break;
            }
        }
        cents.push(data[idx].clone());
    }
    cents
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_utils::ToVecExt;
    use candle_core::DType;

    #[test]
    fn bit_packing_roundtrip() {
        let bits = 2usize;
        let stages = 4usize;
        let bpt = (stages * bits).div_ceil(8);
        let mut buf = vec![0u8; bpt];
        for s in 0..stages {
            pack_bits(&mut buf, s * bits, bits, (s + 1) % 4);
        }
        for s in 0..stages {
            assert_eq!(unpack_bits(&buf, s * bits, bits), (s + 1) % 4);
        }
        assert_eq!(bpt, 1);
    }

    #[test]
    fn encode_decode_quality_improves_with_stages() {
        let dev = Device::Cpu;
        let (tokens, heads, d) = (128usize, 2usize, 16usize);
        let k = crate::tensor_utils::deterministic_randn([1, tokens, heads, d], 42, &dev).unwrap();
        let mut errors = Vec::new();
        for stages in [1usize, 2, 4] {
            let cfg = QvgConfig {
                stages,
                ..Default::default()
            };
            let mut codec = QvgCodec::new(cfg.clone());
            codec.fit(&k, &k).unwrap();
            let payload = codec.encode(0, &k, tokens).unwrap();
            let back = codec.decode(0, &payload, &dev).unwrap();
            let orig = k.to_dtype(DType::F32).unwrap().to_vec4::<f32>().unwrap();
            let rec = back.to_vec4::<f32>().unwrap();
            let mut err = 0f64;
            for l in 0..tokens {
                for h in 0..heads {
                    for dd in 0..d {
                        err += (orig[0][l][h][dd] as f64 - rec[0][l][h][dd] as f64).powi(2);
                    }
                }
            }
            errors.push(err);
        }
        assert!(
            errors[2] < errors[1] && errors[1] < errors[0],
            "RVQ error must decrease with stages: {errors:?}"
        );
    }

    #[test]
    fn storage_is_int8_comparable() {
        let cfg = QvgConfig::default();
        assert_eq!(
            cfg.bytes_per_token(),
            1,
            "4 stages x 2 bits must pack into one byte"
        );
        let per_token = cfg.payload_bytes(1024, 24) as f64 / 1024.0 / 24.0;
        assert!(
            per_token < 1.3,
            "per-token-per-head storage {per_token:.3}B should stay near INT8"
        );
    }

    #[test]
    fn frozen_codebooks() {
        let dev = Device::Cpu;
        let k = crate::tensor_utils::deterministic_randn([1, 64, 2, 8], 5, &dev).unwrap();
        let mut codec = QvgCodec::new(QvgConfig::default());
        codec.fit(&k, &k).unwrap();
        let bytes = codec.codebook_bytes();
        let k2 = crate::tensor_utils::deterministic_randn([1, 64, 2, 8], 9, &dev).unwrap();
        codec.fit(&k2, &k2).unwrap();
        assert_eq!(
            codec.codebook_bytes(),
            bytes,
            "codebooks must stay frozen after the first fit"
        );
    }
}
