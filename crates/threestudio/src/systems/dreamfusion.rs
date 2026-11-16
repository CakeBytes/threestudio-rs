//! DreamFusion system (`threestudio/systems/dreamfusion.py`): render a random
//! view, score it with a guidance model, back-propagate into the geometry.

use crate::config::ExperimentConfig;
use crate::data::{RandomCameraDataset, RayBatch};
use crate::models::geometry::ImplicitVolume;
use crate::models::guidance::{
    Guidance, MockDenoiser, ScoreDistillationGuidance, StableDiffusionGuidance,
};
use crate::models::prompt::PromptProcessor;
use crate::models::renderer::NeRFVolumeRenderer;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

/// Per-step training log.
#[derive(Debug, Clone, Copy)]
pub struct StepLog {
    pub step: usize,
    pub loss: f32,
    pub loss_sds: f32,
    pub grad_norm: f32,
    pub timestep: usize,
}

pub struct DreamFusion<B: AutodiffBackend> {
    pub cfg: ExperimentConfig,
    pub geometry: ImplicitVolume<B>,
    pub renderer: NeRFVolumeRenderer,
    pub prompt_processor: PromptProcessor,
    pub guidance: Box<dyn Guidance<B>>,
    pub data: RandomCameraDataset,
    pub device: B::Device,
    pub global_step: usize,
}

/// Build the guidance named by `system.guidance_type`.
pub fn build_guidance<B: Backend>(cfg: &ExperimentConfig) -> anyhow::Result<Box<dyn Guidance<B>>> {
    match cfg.system.guidance_type.as_str() {
        "mock-sds-guidance" => Ok(Box::new(ScoreDistillationGuidance::new(
            MockDenoiser {
                target: cfg.system.guidance.mock_target_color,
            },
            cfg.system.guidance.clone(),
            cfg.seed,
        ))),
        "stable-diffusion-guidance" => {
            StableDiffusionGuidance::new()?;
            unreachable!()
        }
        other => anyhow::bail!("unknown guidance_type: {other}"),
    }
}

impl<B: AutodiffBackend> DreamFusion<B> {
    pub fn new(cfg: ExperimentConfig, device: B::Device) -> anyhow::Result<Self> {
        anyhow::ensure!(
            cfg.system.geometry_type == "implicit-volume",
            "unsupported geometry_type: {}",
            cfg.system.geometry_type
        );
        anyhow::ensure!(
            cfg.system.renderer_type == "nerf-volume-renderer",
            "unsupported renderer_type: {}",
            cfg.system.renderer_type
        );
        B::seed(cfg.seed);
        let guidance = build_guidance::<B>(&cfg)?;
        Ok(Self {
            geometry: ImplicitVolume::new(&cfg.system.geometry, &device),
            renderer: NeRFVolumeRenderer::new(cfg.system.renderer.clone()),
            prompt_processor: PromptProcessor::new(cfg.system.prompt_processor.clone()),
            data: RandomCameraDataset::new(cfg.data.clone(), cfg.seed),
            guidance,
            device,
            cfg,
            global_step: 0,
        })
    }

    /// Render a batch to an `[1, H, W, 3]` image tensor (with gradients).
    pub fn render_image(&self, batch: &RayBatch, randomized: bool) -> Tensor<B, 4> {
        let (o, d) = batch.to_tensors::<B>(&self.device);
        let out = self.renderer.render(
            &self.geometry,
            o,
            d,
            self.cfg.system.background.color,
            randomized,
        );
        out.comp_rgb
            .clamp(0.0, 1.0)
            .reshape([1, batch.height, batch.width, 3])
    }

    /// One optimisation step, like `DreamFusion.training_step` + Lightning's optimizer step.
    pub fn training_step<O: Optimizer<ImplicitVolume<B>, B>>(&mut self, optim: &mut O) -> StepLog {
        let batch = self.data.sample();
        let prompt = self
            .prompt_processor
            .embed(batch.camera.elevation_deg, batch.camera.azimuth_deg);
        let (o, d) = batch.to_tensors::<B>(&self.device);
        let out =
            self.renderer
                .render(&self.geometry, o, d, self.cfg.system.background.color, true);
        let img = out.comp_rgb.reshape([1, batch.height, batch.width, 3]);
        let g = self.guidance.step(img, &prompt);

        let lcfg = &self.cfg.system.loss;
        let mut loss = g.loss_sds.clone().mul_scalar(lcfg.lambda_sds);
        if lcfg.lambda_sparsity > 0.0 {
            loss = loss + out.opacity.mean().mul_scalar(lcfg.lambda_sparsity);
        }
        let loss_val: f32 = loss.clone().into_scalar().elem();
        let loss_sds: f32 = g.loss_sds.into_scalar().elem();

        let grads = GradientsParams::from_grads(loss.backward(), &self.geometry);
        self.geometry = optim.step(self.cfg.system.optimizer.lr, self.geometry.clone(), grads);
        self.global_step += 1;
        StepLog {
            step: self.global_step,
            loss: loss_val,
            loss_sds,
            grad_norm: g.grad_norm,
            timestep: g.timestep,
        }
    }

    /// Default Adam optimizer from `system.optimizer`.
    pub fn default_optimizer(&self) -> impl Optimizer<ImplicitVolume<B>, B> {
        let o = &self.cfg.system.optimizer;
        AdamConfig::new()
            .with_beta_1(o.betas[0])
            .with_beta_2(o.betas[1])
            .with_epsilon(o.eps)
            .init()
    }

    /// Train for `steps` steps, calling `on_step` after each.
    pub fn fit(&mut self, steps: usize, mut on_step: impl FnMut(&StepLog)) -> Vec<StepLog> {
        let mut optim = self.default_optimizer();
        (0..steps)
            .map(|_| {
                let log = self.training_step(&mut optim);
                on_step(&log);
                log
            })
            .collect()
    }

    /// Mean squared error between eval-orbit renders and the mock target colour.
    pub fn eval_mse_to_color(&self, color: [f32; 3], n_views: usize) -> f32 {
        let views = self.data.eval_orbit(n_views, 15.0);
        let mut total = 0.0;
        for v in &views {
            let img = self.render_image(v, false).inner();
            let target = Tensor::<B::InnerBackend, 1>::from_floats(color, &img.device())
                .reshape([1, 1, 1, 3]);
            let mse: f32 = (img - target).powf_scalar(2.0).mean().into_scalar().elem();
            total += mse;
        }
        total / views.len() as f32
    }

    /// Render one view to 8-bit RGB pixels (for export).
    pub fn render_rgb8(&self, batch: &RayBatch) -> Vec<u8> {
        let img = self.render_image(batch, false).inner();
        img.into_data()
            .to_vec::<f32>()
            .expect("f32 image")
            .into_iter()
            .map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }
}

/// Write a binary PPM image.
pub fn write_ppm(
    path: &std::path::Path,
    width: usize,
    height: usize,
    rgb: &[u8],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    write!(f, "P6\n{width} {height}\n255\n")?;
    f.write_all(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuAutodiff;

    fn tiny_cfg() -> ExperimentConfig {
        let mut cfg = ExperimentConfig::default();
        cfg.data.width = 8;
        cfg.data.height = 8;
        cfg.system.renderer.num_samples_per_ray = 16;
        cfg.system.geometry.n_neurons = 32;
        cfg.system.geometry.n_frequencies = 4;
        cfg.system.optimizer.lr = 1e-2;
        cfg.system.guidance.mock_target_color = [1.0, 0.1, 0.1];
        cfg
    }

    #[test]
    fn training_reduces_eval_error() {
        let cfg = tiny_cfg();
        let color = cfg.system.guidance.mock_target_color;
        let mut sys = DreamFusion::<CpuAutodiff>::new(cfg, Default::default()).unwrap();
        let before = sys.eval_mse_to_color(color, 4);
        let logs = sys.fit(40, |_| {});
        let after = sys.eval_mse_to_color(color, 4);
        assert_eq!(logs.len(), 40);
        assert!(logs.iter().all(|l| l.loss.is_finite()));
        assert!(
            after < before * 0.8,
            "eval mse before {before}, after {after}"
        );
    }

    #[test]
    fn sds_loss_trends_down() {
        let mut sys = DreamFusion::<CpuAutodiff>::new(tiny_cfg(), Default::default()).unwrap();
        let logs = sys.fit(40, |_| {});
        let mean = |s: &[StepLog]| s.iter().map(|l| l.grad_norm).sum::<f32>() / s.len() as f32;
        let (first, last) = (mean(&logs[..10]), mean(&logs[30..]));
        assert!(last < first, "grad norm first {first}, last {last}");
    }

    #[test]
    fn unknown_guidance_is_rejected() {
        let mut cfg = tiny_cfg();
        cfg.system.guidance_type = "stable-diffusion-guidance".into();
        assert!(DreamFusion::<CpuAutodiff>::new(cfg, Default::default()).is_err());
    }
}
