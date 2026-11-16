//! Guidance (`threestudio/models/guidance`).
//!
//! The score-distillation (SDS) update from `StableDiffusionGuidance` is ported
//! faithfully, but the diffusion UNet is abstracted behind [`NoisePredictor`].
//! No Stable Diffusion weights run in Rust yet, so the shipped predictor is
//! [`MockDenoiser`]: the exact denoiser for a data distribution concentrated on
//! a single colour, which makes SDS pull renders toward that colour.

use super::prompt::PromptEmbedding;
use crate::config::GuidanceConfig;
use burn::prelude::*;
use burn::tensor::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Result of one guidance evaluation.
pub struct GuidanceOutput<B: Backend> {
    /// Scalar loss whose gradient w.r.t. the render equals the SDS gradient.
    pub loss_sds: Tensor<B, 1>,
    pub grad_norm: f32,
    pub timestep: usize,
}

/// Anything that turns a rendered image batch into a differentiable loss.
pub trait Guidance<B: Backend> {
    /// `rgb`: `[batch, height, width, 3]` in `[0, 1]`.
    fn step(&mut self, rgb: Tensor<B, 4>, prompt: &PromptEmbedding) -> GuidanceOutput<B>;
}

/// epsilon-prediction network, i.e. the UNet in `StableDiffusionGuidance`.
pub trait NoisePredictor<B: Backend> {
    fn predict_noise(
        &self,
        x_noisy: Tensor<B, 4>,
        timestep: usize,
        alpha_bar: f32,
        prompt: &PromptEmbedding,
    ) -> Tensor<B, 4>;
}

/// Ideal denoiser for a point-mass data distribution at `target` colour.
#[derive(Debug, Clone)]
pub struct MockDenoiser {
    pub target: [f32; 3],
}

impl<B: Backend> NoisePredictor<B> for MockDenoiser {
    fn predict_noise(
        &self,
        x_noisy: Tensor<B, 4>,
        _timestep: usize,
        alpha_bar: f32,
        _prompt: &PromptEmbedding,
    ) -> Tensor<B, 4> {
        let target =
            Tensor::<B, 1>::from_floats(self.target, &x_noisy.device()).reshape([1, 1, 1, 3]);
        (x_noisy - target.mul_scalar(alpha_bar.sqrt())).div_scalar((1.0 - alpha_bar).sqrt())
    }
}

/// Stable Diffusion's `scaled_linear` DDPM schedule (beta 0.00085 -> 0.012).
pub fn sd_alphas_cumprod(num_train_timesteps: usize) -> Vec<f32> {
    let (b0, b1) = (0.00085f64.sqrt(), 0.012f64.sqrt());
    let n = num_train_timesteps.max(2);
    let mut acc = 1.0f64;
    (0..n)
        .map(|i| {
            let b = b0 + (b1 - b0) * i as f64 / (n - 1) as f64;
            acc *= 1.0 - b * b;
            acc as f32
        })
        .collect()
}

/// Score distillation sampling, generic over the noise predictor.
pub struct ScoreDistillationGuidance<P> {
    pub predictor: P,
    pub cfg: GuidanceConfig,
    alphas_cumprod: Vec<f32>,
    rng: StdRng,
}

impl<P> ScoreDistillationGuidance<P> {
    pub fn new(predictor: P, cfg: GuidanceConfig, seed: u64) -> Self {
        let alphas_cumprod = sd_alphas_cumprod(cfg.num_train_timesteps);
        Self {
            predictor,
            cfg,
            alphas_cumprod,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn sample_timestep(&mut self) -> usize {
        let n = self.alphas_cumprod.len() as f32;
        let lo = (n * self.cfg.min_step_percent) as usize;
        let hi = ((n * self.cfg.max_step_percent) as usize).max(lo + 1);
        self.rng
            .random_range(lo..hi)
            .min(self.alphas_cumprod.len() - 1)
    }
}

impl<B: Backend, P: NoisePredictor<B>> Guidance<B> for ScoreDistillationGuidance<P> {
    fn step(&mut self, rgb: Tensor<B, 4>, prompt: &PromptEmbedding) -> GuidanceOutput<B> {
        let device = rgb.device();
        let batch = rgb.dims()[0];
        let t = self.sample_timestep();
        let ab = self.alphas_cumprod[t];

        // forward diffusion + noise prediction happen without gradients
        let latents = rgb.clone();
        let noise = Tensor::<B, 4>::random(latents.dims(), Distribution::Normal(0.0, 1.0), &device);
        let x_noisy = latents.clone().detach().mul_scalar(ab.sqrt())
            + noise.clone().mul_scalar((1.0 - ab).sqrt());
        let eps = self
            .predictor
            .predict_noise(x_noisy, t, ab, prompt)
            .detach();

        // w(t) = 1 - alpha_bar, as in threestudio's `weighting_strategy: sds`
        let mut grad = (eps - noise).mul_scalar(1.0 - ab);
        if let Some(clip) = self.cfg.grad_clip {
            grad = grad.clamp(-clip, clip);
        }
        let grad_norm: f32 = grad
            .clone()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .into_scalar()
            .elem();

        // reparameterisation trick: d(loss)/d(latents) = grad
        let target = (latents.clone() - grad).detach();
        let loss_sds = (latents - target)
            .powf_scalar(2.0)
            .sum()
            .mul_scalar(0.5 / batch as f32);

        GuidanceOutput {
            loss_sds,
            grad_norm,
            timestep: t,
        }
    }
}

/// Placeholder for `stable-diffusion-guidance`. Running SD in Burn needs a UNet,
/// VAE and CLIP text encoder port plus weight loading; not done yet.
pub struct StableDiffusionGuidance;

impl StableDiffusionGuidance {
    pub fn new() -> anyhow::Result<Self> {
        anyhow::bail!(
            "stable-diffusion-guidance is not ported yet; use guidance_type: mock-sds-guidance"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Cpu, CpuAutodiff};

    fn prompt() -> PromptEmbedding {
        PromptEmbedding {
            prompt: "x".into(),
            negative_prompt: String::new(),
            text_embeddings: None,
        }
    }

    #[test]
    fn schedule_is_monotone() {
        let ac = sd_alphas_cumprod(1000);
        assert_eq!(ac.len(), 1000);
        assert!(ac.windows(2).all(|w| w[1] < w[0]));
        assert!(ac[0] < 1.0 && ac[999] > 0.0);
    }

    #[test]
    fn sds_is_zero_at_target() {
        let target = [0.2, 0.4, 0.6];
        let mut g =
            ScoreDistillationGuidance::new(MockDenoiser { target }, GuidanceConfig::default(), 0);
        let img = Tensor::<Cpu, 1>::from_floats(target, &Default::default())
            .reshape([1, 1, 1, 3])
            .repeat_dim(1, 4)
            .repeat_dim(2, 4);
        let out = Guidance::<Cpu>::step(&mut g, img, &prompt());
        assert!(out.grad_norm < 1e-3, "grad norm {}", out.grad_norm);
    }

    #[test]
    fn sds_gradient_points_toward_target() {
        let target = [1.0, 0.0, 0.0];
        let mut g =
            ScoreDistillationGuidance::new(MockDenoiser { target }, GuidanceConfig::default(), 1);
        let device = Default::default();
        let img = Tensor::<CpuAutodiff, 4>::full([1, 2, 2, 3], 0.5, &device).require_grad();
        let out = Guidance::<CpuAutodiff>::step(&mut g, img.clone(), &prompt());
        let grads = out.loss_sds.backward();
        let gi = img
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // red channel below target -> negative gradient; others above -> positive
        for px in gi.chunks(3) {
            assert!(px[0] < 0.0 && px[1] > 0.0 && px[2] > 0.0, "{px:?}");
        }
    }

    #[test]
    fn stable_diffusion_is_stubbed() {
        assert!(StableDiffusionGuidance::new().is_err());
    }
}
