//! End-to-end session tests across every cache policy.

use candle_core::{DType, Device};

use nomas_core::actions::ActionSpec;
use nomas_core::camera::{CameraTrajectory, Intrinsics};
use nomas_core::config::{CacheConfig, CacheStorage, HistorySampling, NomasConfig, SolverConfig};
use nomas_core::modules::text_embed::HashTextEncoder;
use nomas_core::pipeline::{CausalArPipeline, FlowSolver, Guidance, SegmentPlan, SessionPlan};
use nomas_core::vae::PassthroughCodec;
use nomas_core::weights::random_dit;

fn config(storage: CacheStorage, sampling: HistorySampling, past_frames: usize) -> NomasConfig {
    let mut cfg = NomasConfig::tiny();
    cfg.cache = CacheConfig {
        storage,
        sampling,
        capacity: None,
        past_frames,
        cache_at_step: -1,
        cache_noise_sigma: 0.0,
        phase_swap: false,
    };
    cfg.solver = SolverConfig {
        steps: 3,
        shift: 1.0,
        sigma: 0.0,
        seed: 1234,
    };
    cfg
}

fn pipeline(cfg: &NomasConfig, chunk_mode: usize) -> nomas_core::Result<CausalArPipeline> {
    let dev = Device::Cpu;
    let dit = random_dit(cfg, 0x5EED, DType::F32, &dev)?;
    let encoder = HashTextEncoder::new(cfg.text_embed_dim, 256, 7, &dev)?;
    CausalArPipeline::new(
        dit,
        Box::new(encoder),
        Box::new(PassthroughCodec),
        FlowSolver::from(&cfg.solver),
        Guidance {
            text: 2.0,
            camera: 1.5,
        },
        chunk_mode,
    )
}

fn plan(segments: usize, frames: usize) -> SessionPlan {
    let intr = Intrinsics::for_resolution(32, 32);
    SessionPlan {
        segments: (0..segments)
            .map(|i| SegmentPlan {
                prompt: format!("interactive turn {i} of the world session"),
                actions: vec![ActionSpec {
                    start_time: 0.0,
                    end_time: 2.0,
                    description: format!("event {i} unfolds"),
                }],
                camera: Some(match i % 2 {
                    0 => CameraTrajectory::dolly_forward(frames, 0.2, intr),
                    _ => CameraTrajectory::orbit(frames, 2.0, 20.0, intr),
                }),
                latent_frames: frames,
                max_text_tokens: 32,
            })
            .collect(),
        latent_hw: (16, 16),
        latent_fps: 4.0,
        seed: None,
    }
}

fn assert_finite(t: &candle_core::Tensor, what: &str) {
    let v = t
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{what} produced non-finite values"
    );
}

#[test]
fn session_f32_lifelong_cache_grows_unbounded() {
    let cfg = config(CacheStorage::Floating, HistorySampling::Recent, 64);
    let mut p = pipeline(&cfg, 1).unwrap();
    let report = p.generate_session(&plan(4, 3), None).unwrap();
    assert_eq!(report.total_frames, 12);
    assert_eq!(report.segments.len(), 4);
    // Unbounded recent keeps everything: every segment archived.
    assert_eq!(report.cache.archived_segments, 4);
    // Cache tokens grow monotonically across segments.
    let toks: Vec<usize> = report.segments.iter().map(|s| s.cache_tokens).collect();
    assert!(toks.windows(2).all(|w| w[0] < w[1]), "{toks:?}");
    assert_eq!(report.cache.rebuilds, 4);
}

#[test]
fn session_int8_uniform_history_respects_frame_budget() {
    let cfg = config(CacheStorage::Int8, HistorySampling::Uniform, 5);
    let mut p = pipeline(&cfg, 1).unwrap();
    let report = p.generate_session(&plan(4, 3), None).unwrap();
    // 4 segments x 3 frames = 12 frames; uniform over 5 frames.
    // Working set: 5 frames x tokens_per_frame + text blocks of represented
    // segments (frame 0 -> seg 0, frames 3/6/9 -> later segments).
    let tpf = 8 * 8; // 16x16 latent, (1,2,2) patch -> 8x8 tokens
    let expected_frames = 5;
    // Segments whose frames were selected keep their text: uniform over 12
    // frames with budget 5 -> indices 0, 2 (or 3), 5/6, 8/9, 11.
    let expected_text = report
        .segments
        .last()
        .map(|s| s.text_tokens)
        .expect("last segment always represented (frame 11)") as usize;
    let last = report.segments.last().unwrap();
    assert!(
        (last.cache_tokens - expected_frames * tpf).is_multiple_of(expected_text.max(1)),
        "cache tokens {} must be frames*{} + N*text({})",
        last.cache_tokens,
        tpf,
        expected_text
    );
    assert!(
        last.cache_tokens < 12 * tpf + 4 * expected_text,
        "history was subsampled"
    );
    assert!(report.cache.encode_calls >= 4, "int8 encode per segment");
}

#[test]
fn session_qvg_storage() {
    let cfg = config(
        CacheStorage::Qvg(Default::default()),
        HistorySampling::Uniform,
        6,
    );
    let mut p = pipeline(&cfg, 1).unwrap();
    let report = p.generate_session(&plan(3, 3), None).unwrap();
    assert_eq!(report.total_frames, 9);
    assert!(report.cache.encode_calls >= 3);
    // QVG archive should be far smaller than float storage.
    let float_bytes = 3 * 9 * 8 * 8 * (4/*video*/) * 64 * 8 * 2; // tokens * heads * dim * 2(k,v) * f32
    assert!(
        (report.cache.archived_bytes as f64) < float_bytes as f64,
        "qvg archive {}B should beat float {}B",
        report.cache.archived_bytes,
        float_bytes
    );
}

#[test]
fn session_temporal_context_and_recent_mix() {
    for sampling in [
        HistorySampling::TemporalContext { gamma: 2.0 },
        HistorySampling::RecentMixUniform,
    ] {
        let cfg = config(CacheStorage::Floating, sampling, 4);
        let mut p = pipeline(&cfg, 1).unwrap();
        let report = p.generate_session(&plan(4, 2), None).unwrap();
        assert_eq!(report.total_frames, 8);
        let last = report.segments.last().unwrap();
        assert!(last.cache_tokens > 0);
    }
}

#[test]
fn session_swa_capacity_bounds_working_set() {
    let mut cfg = config(CacheStorage::Floating, HistorySampling::Recent, 64);
    let tpf = 8 * 8;
    cfg.cache.capacity = Some(2 * tpf + 40); // ~2 frames + text worth of tokens
    let mut p = pipeline(&cfg, 1).unwrap();
    let report = p.generate_session(&plan(4, 3), None).unwrap();
    for seg in &report.segments {
        assert!(
            seg.cache_tokens <= 2 * tpf + 40 + 2 * tpf, // partial trim rounds up to frames
            "working set {} exceeded the SWA budget",
            seg.cache_tokens
        );
    }
    assert!(report.cache.evicted_tokens > 0, "SWA must evict eventually");
}

#[test]
fn chunk_ar_matches_seg_ar_shapes() {
    let cfg = config(CacheStorage::Floating, HistorySampling::Recent, 64);
    let plan = plan(2, 4);
    let mut seg = pipeline(&cfg, 1).unwrap();
    let r1 = seg.generate_session(&plan, None).unwrap();
    let mut chunk = pipeline(&cfg, 2).unwrap(); // chunks of 2 frames
    let r2 = chunk.generate_session(&plan, None).unwrap();
    assert_eq!(r1.total_frames, r2.total_frames);
    assert_eq!(r1.segments[1].video_tokens, r2.segments[1].video_tokens);
    // chunk_ar runs more solver steps (per chunk) and lands on the same cache
    // geometry because the segment commit packs the same tokens.
    assert!(
        r2.segments[0].steps > r1.segments[0].steps,
        "chunk mode denoises each chunk separately"
    );
}

#[test]
fn determinism_with_fixed_seed() {
    let cfg = config(CacheStorage::Floating, HistorySampling::Recent, 64);
    let plan = plan(2, 2);
    let a = pipeline(&cfg, 1)
        .unwrap()
        .generate_session(&plan, None)
        .unwrap();
    let b = pipeline(&cfg, 1)
        .unwrap()
        .generate_session(&plan, None)
        .unwrap();
    assert_eq!(a.segments.len(), b.segments.len());
    for (x, y) in a.segments.iter().zip(b.segments.iter()) {
        // every field except wall-clock duration must match exactly
        assert_eq!(
            (
                x.index,
                x.latent_frames,
                x.steps,
                x.video_tokens,
                x.text_tokens,
                x.cache_tokens,
                x.archived_segments,
                x.final_velocity_mean.to_bits()
            ),
            (
                y.index,
                y.latent_frames,
                y.steps,
                y.video_tokens,
                y.text_tokens,
                y.cache_tokens,
                y.archived_segments,
                y.final_velocity_mean.to_bits()
            )
        );
    }
    assert_eq!(a.total_frames, b.total_frames);
}

#[test]
fn streaming_callback_sees_decoded_segments() {
    let cfg = config(CacheStorage::Floating, HistorySampling::Recent, 64);
    let mut p = pipeline(&cfg, 1).unwrap();
    let mut seen = 0usize;
    let mut cb = |_: &nomas_core::pipeline::SegmentReport, frames: &candle_core::Tensor| {
        assert_eq!(frames.dims()[2], 2, "decoded latent frames");
        assert_finite(frames, "decoded segment");
        seen += 1;
    };
    let report = p.generate_session(&plan(3, 2), Some(&mut cb)).unwrap();
    assert_eq!(seen, 3);
    assert_eq!(report.total_frames, 6);
}

#[test]
fn latents_are_finite_for_all_storages() {
    for storage in [
        CacheStorage::Floating,
        CacheStorage::Int8,
        CacheStorage::Qvg(Default::default()),
    ] {
        let cfg = config(storage, HistorySampling::Uniform, 4);
        let mut p = pipeline(&cfg, 1).unwrap();
        let plan = plan(3, 2);
        let mut last: Option<candle_core::Tensor> = None;
        let mut cb = |_: &nomas_core::pipeline::SegmentReport, frames: &candle_core::Tensor| {
            last = Some(frames.clone());
        };
        p.generate_session(&plan, Some(&mut cb)).unwrap();
        let l = last.expect("decoded");
        assert_finite(&l, "passthrough decode");
    }
}
