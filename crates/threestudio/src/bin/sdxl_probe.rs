//! `sdxl-probe`: one real SDXL (e.g. Pony Diffusion V6 XL) denoising step on CPU,
//! streaming UNet blocks from the checkpoint, with sanity metrics.
//!
//! ```text
//! sdxl-probe --checkpoint ~/.cache/threestudio-rs/pony/ponyDiffusionV6XL_v6.safetensors --resolution 256
//! ```

use anyhow::Result;
use burn::prelude::*;
use burn::tensor::{Distribution, ElementConversion};
use clap::Parser;
use std::time::Instant;
use threestudio::backend::CpuAutodiff;
use threestudio::config::GuidanceConfig;
use threestudio::models::guidance::{sd_alphas_cumprod, NoisePredictor};
use threestudio::models::prompt::PromptEmbedding;
use threestudio::models::sdxl::{expand_home, SdxlNoisePredictor};

#[derive(Parser, Debug)]
struct Args {
    #[arg(
        long,
        default_value = "~/.cache/threestudio-rs/pony/ponyDiffusionV6XL_v6.safetensors"
    )]
    checkpoint: String,
    /// Pixel resolution (latent = resolution / 8).
    #[arg(long, default_value_t = 256)]
    resolution: usize,
    #[arg(long, default_value_t = 500)]
    timestep: usize,
    #[arg(
        long,
        default_value = "score_9, score_8_up, score_7_up, a red ball on a white background"
    )]
    prompt: String,
    #[arg(long, default_value = "score_4, score_5, score_6")]
    negative: String,
    /// Keep the whole UNet resident (needs ~11 GB free RAM) instead of streaming.
    #[arg(long)]
    resident: bool,
}

fn peak_rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmHWM")).and_then(|l| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<f64>().ok())
            })
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

type IB = threestudio::backend::Cpu;

fn cosine(a: Tensor<IB, 4>, b: Tensor<IB, 4>) -> f32 {
    let dot: f32 = (a.clone() * b.clone()).sum().into_scalar().elem();
    let na: f32 = a.powf_scalar(2.0).sum().sqrt().into_scalar().elem();
    let nb: f32 = b.powf_scalar(2.0).sum().sqrt().into_scalar().elem();
    dot / (na * nb)
}

fn mse(a: Tensor<IB, 4>, b: Tensor<IB, 4>) -> f32 {
    (a - b).powf_scalar(2.0).mean().into_scalar().elem()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Default::default();
    let gcfg = GuidanceConfig {
        resolution: args.resolution,
        guidance_scale: 1.0,
        low_memory: !args.resident,
        ..Default::default()
    };
    let path = expand_home(&args.checkpoint);
    let t0 = Instant::now();
    let pred = SdxlNoisePredictor::<CpuAutodiff>::from_checkpoint(&path, &gcfg, &device)?;
    println!(
        "loaded VAE encoder + CLIP-L + OpenCLIP stems in {:.1}s (peak RSS {:.0} MB)",
        t0.elapsed().as_secs_f64(),
        peak_rss_mb()
    );

    // synthetic render: red disk on white
    let r = args.resolution;
    let mut px = Vec::with_capacity(r * r * 3);
    for y in 0..r {
        for x in 0..r {
            let (dx, dy) = (x as f32 / r as f32 - 0.5, y as f32 / r as f32 - 0.5);
            let inside = dx * dx + dy * dy < 0.09;
            px.extend_from_slice(if inside {
                &[0.9, 0.1, 0.1]
            } else {
                &[1.0, 1.0, 1.0]
            });
        }
    }
    let rgb = Tensor::<CpuAutodiff, 4>::from_data(TensorData::new(px, [1, r, r, 3]), &device);
    let t1 = Instant::now();
    let z = NoisePredictor::<CpuAutodiff>::encode(&pred, rgb).inner();
    let zmean: f32 = z.clone().mean().into_scalar().elem();
    let zstd: f32 = (z.clone() - zmean)
        .powf_scalar(2.0)
        .mean()
        .sqrt()
        .into_scalar()
        .elem();
    println!(
        "VAE encode {r}x{r} -> {:?} in {:.1}s (latent mean {zmean:.3}, std {zstd:.3})",
        z.dims(),
        t1.elapsed().as_secs_f64()
    );

    let ab = sd_alphas_cumprod(1000)[args.timestep];
    IB::seed(1234);
    let noise = Tensor::<IB, 4>::random(z.dims(), Distribution::Normal(0.0, 1.0), &device);
    let xt = z.clone().mul_scalar(ab.sqrt()) + noise.clone().mul_scalar((1.0 - ab).sqrt());
    let prompt = PromptEmbedding {
        prompt: args.prompt.clone(),
        negative_prompt: args.negative.clone(),
        text_embeddings: None,
    };
    let t2 = Instant::now();
    let (eu, ec) = pred.eps_pair(xt.clone(), args.timestep, &prompt)?;
    let dt = t2.elapsed().as_secs_f64();
    println!(
        "UNet (CFG batch of 2, t={}, {}) incl. text encoding: {:.1}s, peak RSS {:.0} MB",
        args.timestep,
        if args.resident {
            "resident"
        } else {
            "streamed from disk"
        },
        dt,
        peak_rss_mb()
    );
    let x0 = (xt.clone() - ec.clone().mul_scalar((1.0 - ab).sqrt())).div_scalar(ab.sqrt());
    let naive = xt.div_scalar(ab.sqrt());
    println!(
        "cos(eps_cond, true noise)   = {:.4}",
        cosine(ec.clone(), noise.clone())
    );
    println!(
        "cos(eps_uncond, true noise) = {:.4}",
        cosine(eu.clone(), noise)
    );
    println!(
        "|eps_cond - eps_uncond| rms = {:.4}",
        mse(ec.clone(), eu).sqrt()
    );
    println!(
        "x0 MSE: predicted {:.4} vs naive (eps=0) {:.4}",
        mse(x0, z.clone()),
        mse(naive, z)
    );
    let finite = ec
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite());
    println!("all outputs finite: {finite}");
    Ok(())
}
