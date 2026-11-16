//! YAML experiment configuration, modelled on threestudio's OmegaConf configs
//! (e.g. `configs/dreamfusion-sd.yaml`).

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ExperimentConfig {
    pub name: String,
    pub tag: String,
    pub seed: u64,
    pub exp_root_dir: String,
    pub data_type: String,
    pub data: RandomCameraConfig,
    pub system_type: String,
    pub system: SystemConfig,
    pub trainer: TrainerConfig,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            name: "dreamfusion-toy".into(),
            tag: String::new(),
            seed: 0,
            exp_root_dir: "outputs".into(),
            data_type: "random-camera-datamodule".into(),
            data: RandomCameraConfig::default(),
            system_type: "dreamfusion-system".into(),
            system: SystemConfig::default(),
            trainer: TrainerConfig::default(),
        }
    }
}

/// Mirrors `RandomCameraDataModuleConfig` in `threestudio/data/uncond.py`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RandomCameraConfig {
    pub width: usize,
    pub height: usize,
    pub batch_size: usize,
    pub elevation_range: [f32; 2],
    pub azimuth_range: [f32; 2],
    pub camera_distance_range: [f32; 2],
    pub fovy_range: [f32; 2],
    pub eval_camera_distance: f32,
    pub eval_fovy_deg: f32,
}

impl Default for RandomCameraConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            batch_size: 1,
            elevation_range: [-10.0, 90.0],
            azimuth_range: [-180.0, 180.0],
            camera_distance_range: [1.0, 1.5],
            fovy_range: [40.0, 70.0],
            eval_camera_distance: 1.5,
            eval_fovy_deg: 70.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SystemConfig {
    pub geometry_type: String,
    pub geometry: GeometryConfig,
    pub renderer_type: String,
    pub renderer: RendererConfig,
    pub background: BackgroundConfig,
    pub prompt_processor: PromptProcessorConfig,
    pub guidance_type: String,
    pub guidance: GuidanceConfig,
    pub loss: LossConfig,
    pub optimizer: OptimizerConfig,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            geometry_type: "implicit-volume".into(),
            geometry: GeometryConfig::default(),
            renderer_type: "nerf-volume-renderer".into(),
            renderer: RendererConfig::default(),
            background: BackgroundConfig::default(),
            prompt_processor: PromptProcessorConfig::default(),
            guidance_type: "mock-sds-guidance".into(),
            guidance: GuidanceConfig::default(),
            loss: LossConfig::default(),
            optimizer: OptimizerConfig::default(),
        }
    }
}

/// Mirrors `ImplicitVolume.Config` (with a frequency encoding instead of a hash grid).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GeometryConfig {
    pub radius: f32,
    pub n_frequencies: usize,
    pub n_neurons: usize,
    pub n_hidden_layers: usize,
    pub density_bias: String,
    pub density_blob_scale: f32,
    pub density_blob_std: f32,
}

impl Default for GeometryConfig {
    fn default() -> Self {
        Self {
            radius: 1.0,
            n_frequencies: 6,
            n_neurons: 64,
            n_hidden_layers: 2,
            density_bias: "blob_magic3d".into(),
            density_blob_scale: 10.0,
            density_blob_std: 0.5,
        }
    }
}

/// Mirrors `NeRFVolumeRenderer.Config`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RendererConfig {
    pub num_samples_per_ray: usize,
    pub randomized: bool,
    pub near: f32,
    pub far: f32,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            num_samples_per_ray: 64,
            randomized: true,
            near: 0.1,
            far: 3.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BackgroundConfig {
    pub color: [f32; 3],
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PromptProcessorConfig {
    pub prompt: String,
    pub negative_prompt: String,
}

impl Default for PromptProcessorConfig {
    fn default() -> Self {
        Self {
            prompt: "a red ball".into(),
            negative_prompt: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GuidanceConfig {
    pub min_step_percent: f32,
    pub max_step_percent: f32,
    pub num_train_timesteps: usize,
    pub grad_clip: Option<f32>,
    /// Target colour used by the mock guidance (stands in for a diffusion prior).
    pub mock_target_color: [f32; 3],
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            min_step_percent: 0.02,
            max_step_percent: 0.98,
            num_train_timesteps: 1000,
            grad_clip: None,
            mock_target_color: [1.0, 0.1, 0.1],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LossConfig {
    pub lambda_sds: f32,
    pub lambda_sparsity: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            lambda_sds: 1.0,
            lambda_sparsity: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct OptimizerConfig {
    pub name: String,
    pub lr: f64,
    pub betas: [f32; 2],
    pub eps: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            name: "Adam".into(),
            lr: 1e-2,
            betas: [0.9, 0.99],
            eps: 1e-15,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TrainerConfig {
    pub max_steps: usize,
    pub log_every_n_steps: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            max_steps: 200,
            log_every_n_steps: 10,
        }
    }
}

impl ExperimentConfig {
    pub fn from_yaml_str(s: &str) -> anyhow::Result<Self> {
        Ok(serde_yaml::from_str(s)?)
    }

    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        Self::from_yaml_str(&text)
    }

    /// Apply `a.b.c=value` overrides, like threestudio's CLI `extras`.
    pub fn apply_overrides(&mut self, overrides: &[String]) -> anyhow::Result<()> {
        let mut value = serde_yaml::to_value(&*self)?;
        for ov in overrides {
            let (key, raw) = ov
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("override must be key=value: {ov}"))?;
            let new: serde_yaml::Value = serde_yaml::from_str(raw)?;
            let mut cur = &mut value;
            for part in key.split('.') {
                cur = cur
                    .get_mut(part)
                    .ok_or_else(|| anyhow::anyhow!("unknown config key: {key}"))?;
            }
            *cur = new;
        }
        *self = serde_yaml::from_value(value)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_partial_yaml_with_defaults() {
        let cfg = ExperimentConfig::from_yaml_str(
            "name: test\nsystem:\n  renderer:\n    num_samples_per_ray: 16\n",
        )
        .unwrap();
        assert_eq!(cfg.name, "test");
        assert_eq!(cfg.system.renderer.num_samples_per_ray, 16);
        assert_eq!(cfg.data.width, 64);
    }

    #[test]
    fn overrides_work() {
        let mut cfg = ExperimentConfig::default();
        cfg.apply_overrides(&["trainer.max_steps=7".into(), "data.width=8".into()])
            .unwrap();
        assert_eq!(cfg.trainer.max_steps, 7);
        assert_eq!(cfg.data.width, 8);
        assert!(cfg.apply_overrides(&["nope.x=1".into()]).is_err());
    }

    #[test]
    fn shipped_configs_parse() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs");
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "yaml") {
                ExperimentConfig::from_file(&path)
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            }
        }
    }
}
