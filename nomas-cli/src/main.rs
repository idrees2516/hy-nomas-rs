//! `nomas` — interactive CLI for the Hy-Nomas Rust port.
//!
//! ```text
//! nomas info                        # print the model/cache/solver configuration
//! nomas demo                        # run a scripted multi-segment world session
//! nomas session                     # interactive multi-turn REPL (text + camera)
//! ```

use std::io::{self, BufRead, Write};

use candle_core::{DType, Device};
use clap::{Parser, Subcommand, ValueEnum};

use nomas_core::actions::ActionSpec;
use nomas_core::camera::{CameraTrajectory, Intrinsics};
use nomas_core::config::{CacheConfig, CacheStorage, HistorySampling, NomasConfig, SolverConfig};
use nomas_core::modules::text_embed::HashTextEncoder;
use nomas_core::pipeline::{CausalArPipeline, FlowSolver, Guidance, SegmentPlan};
use nomas_core::vae::PassthroughCodec;
use nomas_core::weights::random_dit;

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum StorageArg {
    F32,
    Int8,
    Qvg,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum SamplingArg {
    Recent,
    Uniform,
    RecentMix,
    Temporal,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum RolloutArg {
    /// Whole-segment joint denoising, bidirectional intra-segment attention
    /// (upstream `inference_chunk_size = 1`).
    SegAr,
    /// Chunk-level AR inside each segment (upstream `inference_chunk_size = k > 1`).
    ChunkAr,
}

#[derive(Parser)]
#[command(
    name = "nomas",
    version,
    about = "Hy-Nomas interactive video world model — Rust port",
    long_about = "Segmented autoregressive world generation with action-aware causal attention, \
                  a lifelong KV cache (SWA / INT8 / uniform-history / QVG), 3D RoPE, flow-matching \
                  solvers and camera control. Runs a tiny deterministic model out of the box; \
                  load real checkpoints via --checkpoint."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print the active configuration.
    Info,
    /// Run a scripted multi-segment interactive session and print reports.
    Demo {
        /// Number of interactive segments/turns.
        #[arg(long, default_value_t = 4)]
        segments: usize,
        /// Latent frames per segment.
        #[arg(long, default_value_t = 3)]
        frames: usize,
        /// Latent resolution (height = width).
        #[arg(long, default_value_t = 16)]
        res: usize,
        /// Denoising steps (distilled presets: 4 / 6 / 8).
        #[arg(long, default_value_t = 4)]
        steps: usize,
        /// KV cache storage backend.
        #[arg(long, value_enum, default_value = "int8")]
        storage: StorageArg,
        /// History sampling policy.
        #[arg(long, value_enum, default_value = "uniform")]
        sampling: SamplingArg,
        /// Frame budget for archived-history sampling.
        #[arg(long, default_value_t = 6)]
        past_frames: usize,
        /// Token capacity of the working set (0 = unbounded/lifelong).
        #[arg(long, default_value_t = 0)]
        capacity: usize,
        /// Rollout granularity.
        #[arg(long, value_enum, default_value = "seg-ar")]
        rollout: RolloutArg,
        /// Park the KV working set on the host between segments.
        #[arg(long, default_value_t = false)]
        phase_swap: bool,
        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Interactive multi-turn REPL: type a prompt per segment, optional camera moves.
    Session {
        /// Latent frames per turn.
        #[arg(long, default_value_t = 3)]
        frames: usize,
        /// Latent resolution (height = width).
        #[arg(long, default_value_t = 16)]
        res: usize,
        /// Denoising steps.
        #[arg(long, default_value_t = 4)]
        steps: usize,
        /// KV cache storage backend.
        #[arg(long, value_enum, default_value = "int8")]
        storage: StorageArg,
        /// History sampling policy.
        #[arg(long, value_enum, default_value = "uniform")]
        sampling: SamplingArg,
        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

fn storage_from(arg: StorageArg) -> CacheStorage {
    match arg {
        StorageArg::F32 => CacheStorage::Floating,
        StorageArg::Int8 => CacheStorage::Int8,
        StorageArg::Qvg => CacheStorage::Qvg(Default::default()),
    }
}

fn sampling_from(arg: SamplingArg) -> HistorySampling {
    match arg {
        SamplingArg::Recent => HistorySampling::Recent,
        SamplingArg::Uniform => HistorySampling::Uniform,
        SamplingArg::RecentMix => HistorySampling::RecentMixUniform,
        SamplingArg::Temporal => HistorySampling::TemporalContext { gamma: 2.0 },
    }
}

fn build_config(
    storage: StorageArg,
    sampling: SamplingArg,
    past_frames: usize,
    capacity: usize,
    steps: usize,
    phase_swap: bool,
    seed: u64,
) -> NomasConfig {
    let mut cfg = NomasConfig::tiny();
    cfg.cache = CacheConfig {
        storage: storage_from(storage),
        sampling: sampling_from(sampling),
        capacity: if capacity == 0 { None } else { Some(capacity) },
        past_frames: past_frames.max(1),
        cache_at_step: -1,
        cache_noise_sigma: 0.0,
        phase_swap,
    };
    cfg.solver = SolverConfig {
        steps,
        shift: 1.0,
        sigma: 0.0,
        seed,
    };
    cfg
}

fn build_pipeline(cfg: &NomasConfig, chunk_mode: usize) -> anyhow::Result<CausalArPipeline> {
    let dev = Device::Cpu;
    let dit = random_dit(cfg, 0x5EED, DType::F32, &dev)?;
    let encoder = HashTextEncoder::new(cfg.text_embed_dim, 256, 7, &dev)?;
    Ok(CausalArPipeline::new(
        dit,
        Box::new(encoder),
        Box::new(PassthroughCodec),
        FlowSolver::from(&cfg.solver),
        Guidance {
            text: 3.0,
            camera: 2.0,
        },
        chunk_mode,
    )?)
}

fn print_config(cfg: &NomasConfig) {
    println!("nomas-core configuration");
    println!(
        "  transformer : dim={} layers={} heads={} (head_dim {})",
        cfg.dim,
        cfg.n_layers,
        cfg.n_heads,
        cfg.head_dim()
    );
    println!(
        "  latent      : {}ch, patch {:?}",
        cfg.in_channels, cfg.patch_size
    );
    println!(
        "  text        : dim={} separator={}",
        cfg.text_embed_dim, cfg.text_separator
    );
    println!(
        "  rope        : theta={} dims={:?}",
        cfg.rope_theta,
        cfg.rope_dims()
    );
    println!(
        "  ffn         : {:?} (ratio {})",
        cfg.mlp_activation, cfg.mlp_ratio
    );
    println!(
        "  cache       : storage={} sampling={} capacity={:?} past_frames={}",
        cfg.cache.storage, cfg.cache.sampling, cfg.cache.capacity, cfg.cache.past_frames
    );
    println!(
        "  solver      : steps={} shift={} sigma={} seed={:#x}",
        cfg.solver.steps, cfg.solver.shift, cfg.solver.sigma, cfg.solver.seed
    );
}

fn print_segment(i: usize, report: &nomas_core::pipeline::SegmentReport) {
    println!(
        "  segment {i:>2}: {:>2} frames | {:>2} steps | {:>4} video + {:>2} text tokens | cache {:>4} tokens ({:>2} archived) | {:>6.1} ms | |v| = {:.4}",
        report.latent_frames,
        report.steps,
        report.video_tokens,
        report.text_tokens,
        report.cache_tokens,
        report.archived_segments,
        report.duration_ms,
        report.final_velocity_mean,
    );
}

fn print_session(report: &nomas_core::pipeline::SessionReport) {
    println!(
        "  session: {} frames in {:.1} ms",
        report.total_frames, report.total_ms
    );
    println!(
        "  cache  : {} archived segments, {:.2} MB archive, {} rebuilds, {} encode calls, {:.2} MB d2h / {:.2} MB h2d",
        report.cache.archived_segments,
        report.cache.archived_bytes as f64 / 1e6,
        report.cache.rebuilds,
        report.cache.encode_calls,
        report.cache.d2h_bytes as f64 / 1e6,
        report.cache.h2d_bytes as f64 / 1e6,
    );
}

#[allow(clippy::too_many_arguments)]
fn demo(
    segments: usize,
    frames: usize,
    res: usize,
    steps: usize,
    storage: StorageArg,
    sampling: SamplingArg,
    past_frames: usize,
    capacity: usize,
    rollout: RolloutArg,
    phase_swap: bool,
    seed: u64,
) -> anyhow::Result<()> {
    let cfg = build_config(
        storage,
        sampling,
        past_frames,
        capacity,
        steps,
        phase_swap,
        seed,
    );
    print_config(&cfg);
    let chunk_mode = match rollout {
        RolloutArg::SegAr => 1,
        RolloutArg::ChunkAr => 2,
    };
    let mut pipeline = build_pipeline(&cfg, chunk_mode)?;
    let intr = Intrinsics::for_resolution(64, 64);

    let prompts = [
        "a quiet island at dawn, waves rolling toward the shore",
        "the camera pushes forward over the water",
        "gulls sweep past while the light turns golden",
        "storm clouds gather on the horizon",
        "rain begins to fall across the waves",
        "the storm passes and the sun breaks through",
    ];
    let mut latents = Vec::new();
    pipeline.begin((res, res))?;
    for i in 0..segments {
        let plan = SegmentPlan {
            prompt: prompts[i % prompts.len()].to_string(),
            actions: vec![ActionSpec {
                start_time: 0.0,
                end_time: 1.5,
                description: if i % 2 == 0 {
                    "waves crash on the rocks"
                } else {
                    "wind picks up"
                }
                .to_string(),
            }],
            camera: Some(if i % 3 == 0 {
                CameraTrajectory::dolly_forward(frames, 0.2, intr)
            } else if i % 3 == 1 {
                CameraTrajectory::orbit(frames, 2.0, 30.0, intr)
            } else {
                CameraTrajectory::static_camera(frames, intr)
            }),
            latent_frames: frames,
            max_text_tokens: 32,
        };
        let (report, latent) = pipeline.turn(&plan, 4.0, (res, res))?;
        print_segment(i, &report);
        latents.push(latent);
    }
    // decode + phase-swap check
    for (i, latent) in latents.iter().enumerate() {
        let frames_t = pipeline.decode(latent)?;
        if i == 0 {
            println!("  decoded segment 0 to {:?}", frames_t.dims());
        }
    }
    let session = pipeline.finish()?;
    print_session(&session);
    Ok(())
}

fn session_repl(
    frames: usize,
    res: usize,
    steps: usize,
    storage: StorageArg,
    sampling: SamplingArg,
    seed: u64,
) -> anyhow::Result<()> {
    let cfg = build_config(storage, sampling, 8, 0, steps, false, seed);
    println!(
        "nomas interactive session — type a prompt per segment (empty line or Ctrl-D to quit)"
    );
    println!("  commands: :orbit [deg] | :dolly [step] | :static | :rollback | :quit");
    let mut pipeline = build_pipeline(&cfg, 1)?;
    let intr = Intrinsics::for_resolution(64, 64);
    pipeline.begin((res, res))?;
    let stdin = io::stdin();
    let mut turn = 0usize;
    let mut camera_next: Option<CameraTrajectory> = None;
    loop {
        print!("turn {turn} prompt> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() || line == ":quit" {
            break;
        }
        if let Some(rest) = line.strip_prefix(":orbit") {
            let deg: f64 = rest.trim().parse().unwrap_or(30.0);
            camera_next = Some(CameraTrajectory::orbit(frames, 2.0, deg, intr));
            println!("  camera: orbit {deg} deg over the turn");
            continue;
        }
        if let Some(rest) = line.strip_prefix(":dolly") {
            let step: f64 = rest.trim().parse().unwrap_or(0.2);
            camera_next = Some(CameraTrajectory::dolly_forward(frames, step, intr));
            println!("  camera: dolly forward by {step}");
            continue;
        }
        if line == ":static" {
            camera_next = Some(CameraTrajectory::static_camera(frames, intr));
            println!("  camera: static");
            continue;
        }
        if line == ":rollback" {
            match pipeline.rollback() {
                Ok(()) => {
                    turn = turn.saturating_sub(1);
                    println!("  rolled back the last turn");
                }
                Err(e) => println!("  rollback failed: {e}"),
            }
            continue;
        }
        let plan = SegmentPlan {
            prompt: line.to_string(),
            actions: vec![],
            camera: camera_next.take(),
            latent_frames: frames,
            max_text_tokens: 32,
        };
        match pipeline.turn(&plan, 4.0, (res, res)) {
            Ok((report, _latent)) => {
                print_segment(turn, &report);
                turn += 1;
            }
            Err(e) => println!("  generation failed: {e}"),
        }
    }
    let session = pipeline.finish()?;
    print_session(&session);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Info => {
            let cfg = NomasConfig::tiny();
            print_config(&cfg);
            println!("  (presets: NomasConfig::tiny() / ::nomas_8b(); see nomas-core docs)");
            Ok(())
        }
        Commands::Demo {
            segments,
            frames,
            res,
            steps,
            storage,
            sampling,
            past_frames,
            capacity,
            rollout,
            phase_swap,
            seed,
        } => demo(
            segments,
            frames,
            res,
            steps,
            storage,
            sampling,
            past_frames,
            capacity,
            rollout,
            phase_swap,
            seed,
        ),
        Commands::Session {
            frames,
            res,
            steps,
            storage,
            sampling,
            seed,
        } => session_repl(frames, res, steps, storage, sampling, seed),
    }
}
