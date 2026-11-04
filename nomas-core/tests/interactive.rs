//! Interactive rollback + phase-swap offload integration tests.

use candle_core::{DType, Device};

use nomas_core::actions::ActionSpec;
use nomas_core::camera::{CameraTrajectory, Intrinsics};
use nomas_core::config::{CacheConfig, CacheStorage, HistorySampling, NomasConfig, SolverConfig};
use nomas_core::modules::text_embed::HashTextEncoder;
use nomas_core::pipeline::{CausalArPipeline, FlowSolver, Guidance, SegmentPlan};
use nomas_core::vae::PassthroughCodec;
use nomas_core::weights::random_dit;

fn config(sampling: HistorySampling, phase_swap: bool) -> NomasConfig {
    let mut cfg = NomasConfig::tiny();
    cfg.cache = CacheConfig {
        storage: CacheStorage::Floating,
        sampling,
        capacity: None,
        past_frames: 32,
        cache_at_step: -1,
        cache_noise_sigma: 0.0,
        phase_swap,
    };
    cfg.solver = SolverConfig {
        steps: 2,
        shift: 1.0,
        sigma: 0.0,
        seed: 7,
    };
    cfg
}

fn pipeline(cfg: &NomasConfig) -> nomas_core::Result<CausalArPipeline> {
    let dev = Device::Cpu;
    let dit = random_dit(cfg, 0x5EED, DType::F32, &dev)?;
    let encoder = HashTextEncoder::new(cfg.text_embed_dim, 256, 7, &dev)?;
    CausalArPipeline::new(
        dit,
        Box::new(encoder),
        Box::new(PassthroughCodec),
        FlowSolver::from(&cfg.solver),
        Guidance::none(),
        1,
    )
}

fn turn_plan(i: usize) -> SegmentPlan {
    SegmentPlan {
        prompt: format!("turn {i}"),
        actions: vec![ActionSpec::whole_segment(format!("action {i}"))],
        camera: Some(CameraTrajectory::static_camera(
            2,
            Intrinsics::for_resolution(32, 32),
        )),
        latent_frames: 2,
        max_text_tokens: 32,
    }
}

#[test]
fn rollback_restores_cache_state() {
    let cfg = config(HistorySampling::Uniform, false);
    let mut p = pipeline(&cfg).unwrap();
    p.begin((16, 16)).unwrap();
    for i in 0..3 {
        let (report, _) = p.turn(&turn_plan(i), 4.0, (16, 16)).unwrap();
        assert_eq!(report.archived_segments, i + 1);
    }
    // Snapshot the cache state after 3 turns.
    let before = { p.debug_cache_snapshot() };
    // Roll back turn 2.
    p.rollback().unwrap();
    let after = p.debug_cache_snapshot();
    assert_eq!(after.0, 2, "one segment rolled back");
    assert_eq!(after.1, 4, "frame timeline rewound");
    assert!(after.2 <= before.2, "working set must not grow on rollback");
    // Re-roll with a different prompt: the cache accepts the new write.
    let (report, _) = p.turn(&turn_plan(99), 4.0, (16, 16)).unwrap();
    assert_eq!(report.archived_segments, 3);
    let final_ = p.debug_cache_snapshot();
    assert_eq!(final_.0, 3);
    assert_eq!(final_.1, 6);
}

#[test]
fn recent_mode_rejects_rollback() {
    let cfg = config(HistorySampling::Recent, false);
    let mut p = pipeline(&cfg).unwrap();
    p.begin((16, 16)).unwrap();
    let _ = p.turn(&turn_plan(0), 4.0, (16, 16)).unwrap();
    assert!(
        p.rollback().is_err(),
        "recent-mode SWA has no archive to roll back to"
    );
}

#[test]
fn rollback_empty_session_fails() {
    let cfg = config(HistorySampling::Uniform, false);
    let mut p = pipeline(&cfg).unwrap();
    p.begin((16, 16)).unwrap();
    assert!(p.rollback().is_err());
}

#[test]
fn phase_swap_moves_working_set() {
    let cfg = config(HistorySampling::Uniform, true);
    let mut p = pipeline(&cfg).unwrap();
    p.begin((16, 16)).unwrap();
    let (report, latent) = p.turn(&turn_plan(0), 4.0, (16, 16)).unwrap();
    assert!(report.cache_tokens > 0);
    // decode parks + unparks the cache (phase swap).
    let frames = p.decode(&latent).unwrap();
    assert_eq!(frames.dims()[2], 2);
    let session = p.finish().unwrap();
    // d2h/h2d counters observed the park/unpark transfers.
    assert!(
        session.cache.d2h_bytes > 0,
        "park must account device->host bytes"
    );
    assert!(
        session.cache.h2d_bytes > 0,
        "unpark must account host->device bytes"
    );
}

#[test]
fn multi_turn_session_with_rollback_flow() {
    // Full interactive loop: generate, decode, roll back, regenerate.
    let cfg = config(HistorySampling::Uniform, true);
    let mut p = pipeline(&cfg).unwrap();
    p.begin((16, 16)).unwrap();
    for i in 0..2 {
        let (_, latent) = p.turn(&turn_plan(i), 4.0, (16, 16)).unwrap();
        let _ = p.decode(&latent).unwrap();
    }
    p.rollback().unwrap();
    let (_, latent) = p.turn(&turn_plan(42), 4.0, (16, 16)).unwrap();
    let frames = p.decode(&latent).unwrap();
    assert!(frames
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    let session = p.finish().unwrap();
    assert_eq!(session.total_frames, 4);
    assert_eq!(session.cache.rollbacks, 1);
}
