# hy-nomas-rs

**Rust port of the Hy-Nomas interactive video world model core** — segmented autoregressive generation, action-aware causal attention, and a lifelong KV cache, re-engineered on [`candle`](https://github.com/huggingface/candle).

> [Hy-Nomas](https://github.com/Tencent-Hunyuan/Hy-Nomas)（[Hunyuan](https://github.com/Tencent-Hunyuan) 的自由形态交互式视频世界模型，8B DiT）核心的 Rust 移植：分段自回归生成、动作感知因果注意力与终身 KV 缓存。**本文末尾附中文说明。**

---

## What is ported

| Upstream (Python) | This port (Rust) | Notes |
| --- | --- | --- |
| `hymm/models/dit/modules/posemb_layers.py` | `nomas-core/src/rope.rs` | 3D RoPE: per-axis dims `[t, h, w]`, theta **256**, NTK theta rescale, position interpolation, temporal offsets for causal segments, packed multi-sample grids, LaCT teacher-forcing positions |
| `nomas_cpt.py` action masking | `masks.rs` + `actions.rs` | `contained_actions (start_time, end_time, text)` resolved to per-frame token-visibility windows; intra-segment bidirectional / frame-causal / group-causal modes |
| `hymm/models/camera` + `camera_cfg.py` | `camera.rs` + `pipeline/guidance.rs` | Plücker ray conditioning per latent frame, dolly/orbit/static trajectories, two-axis CFG (text + camera) |
| `swa_cache.py` | `cache/swa.rs` | sliding window with **frame-aligned eviction**; a segment's text survives while any of its video frames is resident |
| `uniform_history_cache.py` | `cache/uniform_history.rs` + `cache/history.rs` | CPU archive of every completed segment + working-set rebuild via `uniform` / `recent_mix_uniform` / `temporal_context(gamma)` frame sampling (exact integer formulas), interactive **rollback** without re-encoding |
| `int8_kv_cache.py` | `cache/int8.rs` | per-token-per-head symmetric INT8 with FP32 scales; archived once, never requantized (u8 container + zero-point 128 — see module docs) |
| `qvg_kv_cache.py` | `cache/qvg.rs` | residual vector quantization over 64-token blocks: 4 stages × 2-bit codes (256-entry effective product code), frozen cache-wide codebooks, bit-packed to 1 byte/token/head, text tokens stored as INT8 |
| `PhaseSwappedKVCache` (CPU offload) | `cache/phase_swap.rs` | park/unpark of the working set around decode phases, with d2h/h2d byte accounting |
| `hymm/pipeline/causal_ar.py` | `pipeline/causal_ar.rs` | the `seg_ar` / `chunk_ar` engine: per-turn prompts + actions + camera, KV capture at `cache_at_step`, `cache_noise_sigma` observation noise, segment commit, streaming decode callback, `SessionReport` metrics |
| `hymm/pipeline/solvers` | `pipeline/solver.rs` | flow-matching Euler + SD3-style shifted timesteps + additive-noise (SDE) mode; 4/6/8-step distilled presets |
| DiT blocks (`modules/*`) | `modules/*` | patchify (pure reshape, `(1,2,2)`), QK-RMSNorm, AdaLN-Zero (6-param), SwiGLU/GELU FFN, optional top-k **MoE**, adaLN final head |
| `hymm/models/text_encoder`, Nomas-PE | `modules/text_embed.rs` | `TextEncoder` trait for real LLM encoders + a deterministic `HashTextEncoder` for tests/demos |
| DCAE-V + PixelUMM decoder | `vae.rs` | latent geometry spec (`32ch`, spatial /8, temporal /4 with I-frames) + `VideoCodec` trait; a passthrough preview codec ships for tests |
| checkpoints | `weights.rs` | safetensors loading with a stable key convention and a matched/missing/unexpected `CheckpointReport` |

## Quick start

```bash
cargo test                                   # 97 tests: math parity, cache invariants, e2e sessions
cargo run -p nomas-cli -- info               # print the configuration
cargo run -p nomas-cli -- demo               # scripted 4-turn interactive session (INT8 + uniform history)
cargo run -p nomas-cli -- demo --storage qvg --sampling temporal
cargo run -p nomas-cli -- session            # interactive REPL (text + :orbit/:dolly/:rollback)
cargo run -p nomas-core --example tiny_rollout
```

Example output (`nomas demo --segments 4 --frames 3 --storage int8 --sampling uniform --past-frames 6 --phase-swap`):

```text
segment  0:  3 frames |  4 steps |  192 video + 16 text tokens | cache  208 tokens ( 1 archived) |  162 ms
segment  3:  3 frames |  4 steps |  192 video + 10 text tokens | cache  435 tokens ( 4 archived) |  255 ms
session: 12 frames in 875 ms
cache  : 4 archived segments, 0.26 MB archive, 4 rebuilds, 8 encode calls, 2.04 MB d2h / 3.30 MB h2d
```

The demo runs a deterministically initialized **tiny** DiT (no checkpoint needed). `NomasConfig::nomas_8b()` encodes the 8B preset (dim 3072 × 48 layers × 24 heads, BF16, INT8 uniform-history cache, 6-step solver); the geometry is overridden from real checkpoints at load time.

## Architecture of one interactive turn

```text
            ┌──────────────────────────── per turn (segment) ───────────────────────────┐
 text ──► TextEncoder ──► projector ──► [ video tokens | text tokens (actions inside) ] ─┐
 camera ─► Plücker rays ─► MLP ─────────► per-frame additive conditioning               │
            noise x_T ──► flow-matching Euler loop (distilled 4/6/8 steps) ◄────────────┤
                              │  dual CFG: v_uc + s_text·(v_tnc − v_uc) + s_cam·(v_cond − v_tnc)
                              │  attention: [ lifelong history | current ] with the
                              │             action-window mask + 3D RoPE (theta 256)
                              └─► capture per-layer K/V at cache_at_step
                                          │
                                          ▼
                      HistoryCache.commit ─► archive (F32 / INT8 / QVG) ─► rebuild working set
                      (recent frame-aligned SWA │ uniform / recent-mix / temporal-context)
                                          │
                                          ▼
                      decode via VideoCodec (phase-swap park/unpark) ──► SessionReport
```

## The lifelong KV cache in one page

* **Packing** — every segment contributes `[video tokens (frames × tokens/frame) | text block]`, with `action_start` marking the first action token (upstream `ArchivedSegment` layout). Keys are stored **already rotated** with the RoPE positions of their original frames, so history rebuilds never re-rotate.
* **Recent / SWA** — bounded token capacity, cuts land on frame boundaries; a fully-evicted video takes its text with it only when the budget demands it.
* **Archived sampling** — the full archive stays on the host; the working set is rebuilt from `past_frames` sampled frames (endpoint-inclusive uniform grid, half-recent half-uniform mix, or power-spaced `temporal_context`) plus the text of every represented segment. Future turns can reselect frames that are not currently resident.
* **Compression** — INT8 (per-token-per-head scales) or QVG RVQ (4 × 2-bit stages, block-64 scales, frozen codebooks, 1 packed byte/token/head — INT8-class footprint with vector-quantization fidelity; text stays INT8). In the demo above QVG shrinks the archive **13×** versus INT8 at the same turn count.
* **Rollback** — drop the last committed segment and rebuild; requires an archived (non-recent) sampling, mirroring the upstream `UnsupportedStreamingRollbackError`.
* **Accounting** — `CacheMetrics` reports archive bytes, encode calls, d2h/h2d bytes, rebuilds and evictions (`kv_codec_stats` / `inference_metrics` upstream).

## Design notes & honest interpretations

The upstream GPU kernels (`magi`/`flex`/`triton` attention, the QVG codec internals, PixelUMM) are not part of the published sources this port was written from. Everything algorithmic above is ported exactly; the following are **documented interpretations** rather than line-by-line ports:

* **QVG** is realized as block-scaled residual VQ matching every released knob (`num_bits: 2`, 4 stages, 256 centroids, block 64, text as INT8). See `cache/qvg.rs` for the codec description.
* **Attention** runs as portable SDPA (F32 softmax, additive masks). The mask semantics (inter-segment causal, intra-segment bidirectional, per-action windows, chunk groups) are the ported contract; fused/flash kernels slot behind the same signature on CUDA builds.
* **`cache_noise_sigma`** is applied as observation noise on the KV-capture pass (the model sees `x_t + σ·ε`), preserving its anti-drift intent for the KV-commit path. The latent-conditioning path (`ar_cache_latent_cond_only`) is not exercised by default.
* **8B preset dimensions** follow HunyuanVideo-family conventions and are placeholders until read from a real checkpoint (`weights::load_dit` reports every matched/missing key).
* **candle tensors are immutable**, so the upstream in-place KV arena becomes a *materialized* working set (concatenation of archived slices) with identical token geometry; the phase-swap wrapper still performs real device moves under the `cuda` feature.
* **CPU-first**: the crate runs anywhere candle does; enable the `cuda` feature (CUDA toolkit required) for GPU tensors. BF16 weights are supported; tests default to F32 for portability.

## Repository layout

```text
nomas-core/            the library
  src/rope.rs          3D RoPE (offsets, packing, LaCT, rescale/interpolation)
  src/actions.rs       timed actions -> token visibility windows
  src/camera.rs        Plücker rays, trajectories
  src/masks.rs         action-aware causal mask builder
  src/modules/         patchify, norms, timestep, AdaLN, FFN, MoE, attention, blocks
  src/cache/           history.rs, swa.rs, uniform_history.rs, int8.rs, qvg.rs, phase_swap.rs
  src/model/dit.rs     NomasDiT + HistoryView (pending chunks)
  src/pipeline/        causal_ar.rs (seg/chunk engine), solver.rs, guidance.rs, prompt.rs
  src/vae.rs           DCAE-V spec + VideoCodec
  src/weights.rs       safetensors loading + key diagnostics
  tests/               session.rs (all cache policies, e2e), interactive.rs (rollback, phase swap)
nomas-cli/             the `nomas` binary (info / demo / session)
```

## Testing

97 tests cover: RoPE closed-form parity, offset slicing, packed concatenation, orthogonality; action windows (frame centers, empty windows); mask semantics (history visibility, bidirectional/frame/group causality, action queries); INT8 round-trip error bounds and NaN/Inf rejection; QVG bit-packing, stage-wise error decrease, frozen codebooks, storage footprint; SWA frame-aligned eviction edge cases; all history samplers on exhaustive `(total, past)` budgets (count/uniqueness/chronology); solver schedules and exact Euler integration; dual-CFG algebra; end-to-end sessions across every storage × sampling combination, chunk-vs-segment rollout, determinism, streaming decode, rollback and phase-swap byte accounting.

## Credits & license

* Upstream model and research: **Tencent Hunyuan — [Hy-Nomas](https://github.com/Tencent-Hunyuan/Hy-Nomas)** (*Hy-Nomas: Towards Free-form Interactive World Generation*). Please respect the upstream repository's license and terms for model weights.
* This port: MIT (see `LICENSE`). It is an independent re-implementation for research and engineering purposes and ships **no** model weights.

---

## 中文说明

**hy-nomas-rs** 将 [Tencent-Hunyuan/Hy-Nomas](https://github.com/Tencent-Hunyuan/Hy-Nomas)（自由形态交互式视频世界模型，8B 单流 DiT）的核心移植到 Rust（基于 [candle](https://github.com/huggingface/candle) 张量库）。

**移植的核心能力**

* **分段自回归生成**：`seg_ar`（段级联合去噪，段内双向注意力）与 `chunk_ar`（段内按 k 帧分块、逐块去噪、已去噪块作为干净历史）两种推演模式，对应上游 `inference_chunk_size = 1 / 0 / k`。
* **动作感知因果注意力**：`contained_actions (start_time, end_time, description)` 解析为逐帧 token 可见窗口；段间因果、段内双向/帧级/分组因果；3D RoPE（theta=256，含时间偏移、打包序列、LaCT 教师强制位置、NTK 缩放）。
* **终身 KV 缓存**：帧对齐滑窗（SWA）、CPU 全量存档 + `uniform / recent_mix_uniform / temporal_context(gamma)` 帧采样重建、INT8（逐 token 逐 head 对称量化 + FP32 scale）、QVG 残差矢量量化（4 级 × 2bit，块大小 64，码本冻结，1 字节/token/head）、交互式回滚（rollback）、相位交换 CPU 卸载（park/unpark + d2h/h2d 字节统计）。
* **流水线**：流匹配 Euler / 蒸馏少步（4/6/8 步）/ 移位时间步 / SDE 噪声模式；文本 + 相机双轴 CFG（Plücker 光线嵌入，doll/orbit/静止轨迹）；`cache_at_step` KV 捕获与 `cache_noise_sigma` 观测噪声。
* **接口**：DCAE-V 隐空间规格（32 通道、空间 1/8、时间 1/4）与 `VideoCodec` 解码器接口（内置演示用直通编解码器）、safetensors 权重加载与键位诊断报告。

**快速开始**

```bash
cargo test                                # 97 项测试：数学一致性、缓存不变量、端到端会话
cargo run -p nomas-cli -- demo            # 脚本化多轮交互演示（INT8 + uniform 历史采样）
cargo run -p nomas-cli -- demo --storage qvg --sampling temporal
cargo run -p nomas-cli -- session         # 交互式 REPL（文本 + :orbit/:dolly/:rollback）
```

演示默认使用确定性随机初始化的 tiny 模型，无需下载权重；`NomasConfig::nomas_8b()` 提供 8B 预设（加载真实 checkpoint 时自动被其几何覆盖）。

**诚实的说明**：上游未公开 GPU kernel 源码（magi/flex 注意力、QVG 编解码内部、PixelUMM 解码器）。本移植对所有公开的算法语义逐条对齐，对未公开部分（QVG 的具体编码格式、`cache_noise_sigma` 的作用路径、8B 具体维度）给出**文档化的解释性实现**，并在源码注释与本 README 中逐一标注。本仓库不含任何模型权重，请遵循上游仓库的许可条款。

许可证：MIT（本移植）；模型与研究工作版权归腾讯混元所有。
