//! End-to-end tiny rollout — two interactive segments with timed actions,
//! camera control, the lifelong KV cache and phase-swapped offload, using a
//! deterministically initialized tiny DiT (no checkpoint needed).
//!
//! Run with: `cargo run -p nomas-core --example tiny_rollout`

use candle_core::{DType, Device};

use nomas_core::actions::ActionSpec;
use nomas_core::camera::{CameraTrajectory, Intrinsics};
use nomas_core::config::{CacheConfig, CacheStorage, HistorySampling, NomasConfig, SolverConfig};
use nomas_core::modules::text_embed::HashTextEncoder;
use nomas_core::pipeline::{CausalArPipeline, FlowSolver, Guidance};
use nomas_core::vae::PassthroughCodec;
use nomas_core::weights::random_dit;

fn main() {
    if let Err(e) = run() {
        eprintln!("tiny_rollout failed: {e}");
        std::process::exit(1);
    }
}

#[allow(dead_code)]
fn run() -> nomas_core::Result<()> {
    let dev = Device::Cpu;
    let mut cfg = NomasConfig::tiny();
    cfg.cache = CacheConfig {
        storage: CacheStorage::Int8,
        sampling: HistorySampling::Uniform,
        capacity: None,
        past_frames: 6,
        cache_at_step: -1,
        cache_noise_sigma: 0.0,
        phase_swap: true,
    };
    cfg.solver = SolverConfig {
        steps: 4,
        shift: 1.0,
        sigma: 0.0,
        seed: 42,
    };

    let dit = random_dit(&cfg, 0x5EED, DType::F32, &dev)?;
    println!(
        "tiny DiT: {} params over {} blocks",
        dit.n_params(),
        cfg.n_layers
    );

    let encoder = HashTextEncoder::new(cfg.text_embed_dim, 256, 7, &dev)?;
    let mut pipeline = CausalArPipeline::new(
        dit,
        Box::new(encoder),
        Box::new(PassthroughCodec),
        FlowSolver::from(&cfg.solver),
        Guidance {
            text: 3.0,
            camera: 2.0,
        },
        1, // seg_ar, full bidirectional
    )?;

    let intr = Intrinsics::for_resolution(64, 64);
    let plan = [
        nomas_core::pipeline::SegmentPlan {
            prompt: "a quiet island at dawn, waves rolling in".into(),
            actions: vec![ActionSpec {
                start_time: 0.0,
                end_time: 1.5,
                description: "waves crash".into(),
            }],
            camera: Some(CameraTrajectory::dolly_forward(3, 0.2, intr)),
            latent_frames: 3,
            max_text_tokens: 32,
        },
        nomas_core::pipeline::SegmentPlan {
            prompt: "the camera keeps pushing toward the shore".into(),
            actions: vec![],
            camera: Some(CameraTrajectory::orbit(3, 2.0, 30.0, intr)),
            latent_frames: 3,
            max_text_tokens: 32,
        },
    ];

    pipeline.begin((16, 16))?;
    for (i, seg) in plan.iter().enumerate() {
        let (report, latent) = pipeline.turn(seg, 4.0, (16, 16))?;
        let frames = pipeline.decode(&latent)?;
        println!(
            "segment {i}: {} frames, {} steps, {} video tokens, cache {} tokens ({} archived) in {:.0} ms; decoded {:?}",
            report.latent_frames,
            report.steps,
            report.video_tokens,
            report.cache_tokens,
            report.archived_segments,
            report.duration_ms,
            frames.dims()
        );
    }
    let session = pipeline.finish()?;
    println!(
        "session: {} frames total in {:.0} ms; cache: {} archived segments, {:.1} MB archive, {} rebuilds",
        session.total_frames,
        session.total_ms,
        session.cache.archived_segments,
        session.cache.archived_bytes as f64 / 1e6,
        session.cache.rebuilds,
    );
    Ok(())
}
