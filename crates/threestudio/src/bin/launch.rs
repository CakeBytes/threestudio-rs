//! `launch` — the Rust counterpart of threestudio's `launch.py`.
//!
//! ```text
//! launch --config configs/dreamfusion-toy.yaml --train [--gpu] [key=value ...]
//! ```

use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use clap::Parser;
use std::path::PathBuf;
use threestudio::config::ExperimentConfig;
use threestudio::systems::dreamfusion::{write_ppm, DreamFusion};

#[derive(Parser, Debug)]
#[command(version, about = "threestudio-rs launcher")]
struct Args {
    /// Path to a YAML experiment config.
    #[arg(long)]
    config: PathBuf,
    /// Run training.
    #[arg(long)]
    train: bool,
    /// Render the eval orbit after (or without) training.
    #[arg(long)]
    test: bool,
    /// Use the Vulkan (wgpu) backend. Requires the `vulkan` feature.
    #[arg(long)]
    gpu: bool,
    /// Config overrides, e.g. `trainer.max_steps=10`.
    extras: Vec<String>,
}

fn run<B: AutodiffBackend>(cfg: ExperimentConfig, args: &Args, device: B::Device) -> Result<()> {
    let out_dir = PathBuf::from(&cfg.exp_root_dir).join(&cfg.name);
    let mut sys = DreamFusion::<B>::new(cfg.clone(), device)?;
    let every = cfg.trainer.log_every_n_steps.max(1);
    if args.train {
        sys.fit(cfg.trainer.max_steps, |l| {
            if l.step % every == 0 || l.step == 1 {
                println!(
                    "step {:>5} | loss {:.5} | sds {:.5} | grad_norm {:.4} | t {}",
                    l.step, l.loss, l.loss_sds, l.grad_norm, l.timestep
                );
            }
        });
        println!(
            "eval mse to target colour: {:.5}",
            sys.eval_mse_to_color(cfg.system.guidance.mock_target_color, 4)
        );
    }
    if args.test {
        std::fs::create_dir_all(&out_dir)?;
        for (i, view) in sys.data.eval_orbit(8, 15.0).iter().enumerate() {
            let path = out_dir.join(format!("view_{i:03}.ppm"));
            write_ppm(&path, view.width, view.height, &sys.render_rgb8(view))?;
        }
        println!("wrote eval renders to {}", out_dir.display());
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut cfg = ExperimentConfig::from_file(&args.config)?;
    cfg.apply_overrides(&args.extras)?;
    println!(
        "experiment: {} (system: {}, guidance: {})",
        cfg.name, cfg.system_type, cfg.system.guidance_type
    );

    if args.gpu {
        #[cfg(feature = "vulkan")]
        {
            anyhow::ensure!(
                threestudio::backend::gpu_available(),
                "no Vulkan adapter found"
            );
            return run::<threestudio::backend::GpuAutodiff>(cfg, &args, Default::default());
        }
        #[cfg(not(feature = "vulkan"))]
        anyhow::bail!("built without the `vulkan` feature; rebuild with --features vulkan");
    }
    run::<threestudio::backend::CpuAutodiff>(cfg, &args, Default::default())
}
