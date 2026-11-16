//! Stable Diffusion XL in Burn — enough of it for score-distillation guidance:
//! UNet, both CLIP text encoders, BPE tokenizer and the VAE encoder, loaded
//! straight from a single-file SGM/LDM checkpoint such as Pony Diffusion V6 XL.

pub mod clip;
pub mod nn;
pub mod tokenizer;
pub mod unet;
pub mod vae;
pub mod weights;

use crate::config::GuidanceConfig;
use crate::models::guidance::NoisePredictor;
use crate::models::prompt::PromptEmbedding;
use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use clip::{ClipConfig, ClipLayer, ClipStem};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizer::ClipTokenizer;
use unet::{SdxlUnet, UnetConfig};
use vae::{VaeConfig, VaeEncoder};
use weights::{SafeTensorsFile, VarBuilder};

/// Checkpoint prefixes of the SGM single-file format.
pub const UNET_PREFIX: &str = "model.diffusion_model";
pub const CLIP_L_PREFIX: &str = "conditioner.embedders.0.transformer";
pub const CLIP_G_PREFIX: &str = "conditioner.embedders.1.model";
pub const VAE_PREFIX: &str = "first_stage_model";

/// Keys present in SDXL checkpoints that SDS guidance intentionally does not use.
pub fn is_intentionally_unused(key: &str) -> bool {
    key.starts_with("first_stage_model.decoder.")
        || key.starts_with("first_stage_model.post_quant_conv.")
        || key == "conditioner.embedders.0.transformer.text_model.embeddings.position_ids"
        || key == "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight"
        || key == "conditioner.embedders.0.transformer.text_model.final_layer_norm.bias"
        || key == "conditioner.embedders.1.model.logit_scale"
        || key.starts_with("conditioner.embedders.0.transformer.text_model.encoder.layers.11.")
}

/// Architecture of every SDXL component.
#[derive(Debug, Clone, PartialEq)]
pub struct SdxlConfig {
    pub unet: UnetConfig,
    pub clip_l: ClipConfig,
    pub clip_g: ClipConfig,
    pub vae: VaeConfig,
}

impl SdxlConfig {
    pub fn sdxl() -> Self {
        Self {
            unet: UnetConfig::sdxl(),
            clip_l: ClipConfig::vit_l(),
            clip_g: ClipConfig::vit_bigg(),
            vae: VaeConfig::sdxl(),
        }
    }

    /// Tiny, randomly initialisable variant with the same topology (for tests).
    pub fn tiny() -> Self {
        let unet = UnetConfig::tiny();
        let (wl, wg) = (16, unet.context_dim - 16);
        let mut clip_g = ClipConfig::tiny(clip::ClipLayout::OpenClip, wg);
        clip_g.projection_dim = Some(unet.pooled_dim);
        Self {
            clip_l: ClipConfig::tiny(clip::ClipLayout::HuggingFace, wl),
            clip_g,
            vae: VaeConfig::tiny(),
            unet,
        }
    }
}

/// Expand a leading `~/` in a path.
pub fn expand_home(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => Path::new(&home).join(rest),
        _ => PathBuf::from(p),
    }
}

/// How rendered RGB is mapped to the 4-channel latent space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentEncoder {
    Vae,
    Linear,
}

type Inner<B> = <B as AutodiffBackend>::InnerBackend;
/// `(eps_uncond, eps_cond)`.
pub type EpsPair<B> = (Tensor<Inner<B>, 4>, Tensor<Inner<B>, 4>);

/// Text-conditioning tensors for one prompt.
#[derive(Clone)]
struct TextCond<IB: Backend> {
    context: Tensor<IB, 3>,
    pooled: Tensor<IB, 2>,
}

/// SDXL ε-predictor with classifier-free guidance, usable as a
/// [`NoisePredictor`] inside [`crate::models::guidance::ScoreDistillationGuidance`].
pub struct SdxlNoisePredictor<B: AutodiffBackend> {
    pub cfg: SdxlConfig,
    pub guidance_scale: f32,
    pub resolution: usize,
    pub latent_encoder: LatentEncoder,
    vb: VarBuilder<Inner<B>>,
    unet: Option<SdxlUnet<Inner<B>>>,
    stem_l: ClipStem<Inner<B>>,
    stem_g: ClipStem<Inner<B>>,
    layers_l: Option<Vec<ClipLayer<Inner<B>>>>,
    layers_g: Option<Vec<ClipLayer<Inner<B>>>>,
    vae: Option<VaeEncoder<B>>,
    tokenizer: ClipTokenizer,
    cache: RefCell<HashMap<String, TextCond<Inner<B>>>>,
}

impl<B: AutodiffBackend> SdxlNoisePredictor<B> {
    /// Build from a root `VarBuilder` pair (inner backend for UNet/CLIP, autodiff
    /// backend for the VAE encoder, which needs gradients w.r.t. its input).
    /// With `low_memory`, UNet blocks and OpenCLIP layers are streamed from the
    /// checkpoint on every call instead of being kept resident.
    pub fn from_varbuilders(
        vb: VarBuilder<Inner<B>>,
        vb_ad: VarBuilder<B>,
        cfg: SdxlConfig,
        gcfg: &GuidanceConfig,
        low_memory: bool,
    ) -> Result<Self> {
        let latent_encoder = match gcfg.latent_encoder.as_str() {
            "vae" => LatentEncoder::Vae,
            "linear" => LatentEncoder::Linear,
            other => anyhow::bail!("unknown latent_encoder {other} (expected vae | linear)"),
        };
        let vl = vb.pp(CLIP_L_PREFIX);
        let vg = vb.pp(CLIP_G_PREFIX);
        Ok(Self {
            unet: if low_memory {
                None
            } else {
                Some(SdxlUnet::load(&vb.pp(UNET_PREFIX), cfg.unet.clone())?)
            },
            stem_l: ClipStem::load(&vl, &cfg.clip_l)?,
            stem_g: ClipStem::load(&vg, &cfg.clip_g)?,
            layers_l: Some(clip::load_layers(&vl, &cfg.clip_l)?),
            layers_g: if low_memory {
                None
            } else {
                Some(clip::load_layers(&vg, &cfg.clip_g)?)
            },
            vae: match latent_encoder {
                LatentEncoder::Vae => {
                    Some(VaeEncoder::load(&vb_ad.pp(VAE_PREFIX), cfg.vae.clone())?)
                }
                LatentEncoder::Linear => None,
            },
            guidance_scale: gcfg.guidance_scale,
            resolution: gcfg.resolution,
            latent_encoder,
            tokenizer: ClipTokenizer::new(),
            cache: RefCell::new(HashMap::new()),
            vb,
            cfg,
        })
    }

    /// Load from a single-file SDXL checkpoint (e.g. Pony Diffusion V6 XL).
    pub fn from_checkpoint(path: &Path, gcfg: &GuidanceConfig, device: &B::Device) -> Result<Self> {
        let file = Arc::new(
            SafeTensorsFile::open(path)
                .with_context(|| format!("loading SDXL checkpoint {}", path.display()))?,
        );
        let inner_device: <Inner<B> as Backend>::Device = device.clone();
        Self::from_varbuilders(
            VarBuilder::from_file(file.clone(), &inner_device),
            VarBuilder::from_file(file, device),
            SdxlConfig::sdxl(),
            gcfg,
            gcfg.low_memory,
        )
    }

    fn text_cond(&self, text: &str, zero_if_empty: bool) -> Result<TextCond<Inner<B>>> {
        if let Some(c) = self.cache.borrow().get(text) {
            return Ok(c.clone());
        }
        let device = self.vb.device.clone();
        let cond = if zero_if_empty && text.trim().is_empty() {
            // SDXL's `force_zeros_for_empty_prompt`
            TextCond {
                context: Tensor::zeros(
                    [1, tokenizer::CONTEXT_LENGTH, self.cfg.unet.context_dim],
                    &device,
                ),
                pooled: Tensor::zeros([1, self.cfg.unet.pooled_dim], &device),
            }
        } else {
            let ids_l = vec![self.tokenizer.encode(text, self.cfg.clip_l.pad_id)];
            let ids_g = vec![self.tokenizer.encode(text, self.cfg.clip_g.pad_id)];
            let l = clip::encode(
                &self.vb.pp(CLIP_L_PREFIX),
                &self.cfg.clip_l,
                &self.stem_l,
                self.layers_l.as_deref(),
                &ids_l,
            )?;
            let g = clip::encode(
                &self.vb.pp(CLIP_G_PREFIX),
                &self.cfg.clip_g,
                &self.stem_g,
                self.layers_g.as_deref(),
                &ids_g,
            )?;
            TextCond {
                context: Tensor::cat(vec![l.penultimate, g.penultimate], 2),
                pooled: g.pooled.context("OpenCLIP pooled output")?,
            }
        };
        self.cache
            .borrow_mut()
            .insert(text.to_string(), cond.clone());
        Ok(cond)
    }

    /// Classifier-free-guided ε on the inner (non-autodiff) backend.
    pub fn guided_eps(
        &self,
        x: Tensor<Inner<B>, 4>,
        timestep: usize,
        prompt: &PromptEmbedding,
    ) -> Result<Tensor<Inner<B>, 4>> {
        let (eu, ec) = self.eps_pair(x, timestep, prompt)?;
        Ok(eu.clone() + (ec - eu).mul_scalar(self.guidance_scale))
    }

    /// Unconditional and conditional ε, computed in one batched UNet pass.
    pub fn eps_pair(
        &self,
        x: Tensor<Inner<B>, 4>,
        timestep: usize,
        prompt: &PromptEmbedding,
    ) -> Result<EpsPair<B>> {
        let b = x.dims()[0];
        let c = self.text_cond(&prompt.prompt, false)?;
        let u = self.text_cond(&prompt.negative_prompt, true)?;
        let rep3 = |t: Tensor<Inner<B>, 3>| t.repeat_dim(0, b);
        let rep2 = |t: Tensor<Inner<B>, 2>| t.repeat_dim(0, b);
        let context = Tensor::cat(vec![rep3(u.context), rep3(c.context)], 0);
        let pooled = Tensor::cat(vec![rep2(u.pooled), rep2(c.pooled)], 0);
        let r = self.resolution as f32;
        let y = unet::sdxl_vector_cond(&self.cfg.unet, pooled, [r, r, 0.0, 0.0, r, r]);
        let x2 = Tensor::cat(vec![x.clone(), x], 0);
        let ts = vec![timestep as f32; 2 * b];
        let eps = match &self.unet {
            Some(u) => u.forward(x2, &ts, context, y),
            None => unet::forward_streaming(
                &self.vb.pp(UNET_PREFIX),
                &self.cfg.unet,
                x2,
                &ts,
                context,
                y,
            )?,
        };
        let parts = eps.chunk(2, 0);
        Ok((parts[0].clone(), parts[1].clone()))
    }
}

impl<B: AutodiffBackend> NoisePredictor<B> for SdxlNoisePredictor<B> {
    fn encode(&self, rgb: Tensor<B, 4>) -> Tensor<B, 4> {
        // [b, h, w, 3] in [0, 1] -> [b, 3, res, res] in [-1, 1]
        let x = rgb.permute([0, 3, 1, 2]);
        let r = self.resolution;
        let x = nn::resize_bilinear(x, r, r);
        let x = x.mul_scalar(2.0).sub_scalar(1.0);
        match (&self.vae, self.latent_encoder) {
            (Some(vae), LatentEncoder::Vae) => vae.encode(x),
            _ => vae::rgb_to_latent_linear(x, self.cfg.vae.factor()),
        }
    }

    fn predict_noise(
        &self,
        x_noisy: Tensor<B, 4>,
        timestep: usize,
        _alpha_bar: f32,
        prompt: &PromptEmbedding,
    ) -> Tensor<B, 4> {
        let eps = self
            .guided_eps(x_noisy.inner(), timestep, prompt)
            .expect("SDXL noise prediction failed");
        Tensor::from_inner(eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Cpu, CpuAutodiff};
    use crate::models::guidance::{Guidance, ScoreDistillationGuidance};
    use std::collections::BTreeSet;

    fn default_checkpoint() -> PathBuf {
        std::env::var("THREESTUDIO_SDXL_CKPT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                expand_home("~/.cache/threestudio-rs/pony/ponyDiffusionV6XL_v6.safetensors")
            })
    }

    #[test]
    fn tiny_sdxl_sds_step_runs_and_backprops() {
        let device = Default::default();
        let gcfg = GuidanceConfig {
            resolution: 32,
            guidance_scale: 7.5,
            ..Default::default()
        };
        let pred = SdxlNoisePredictor::<CpuAutodiff>::from_varbuilders(
            VarBuilder::<Cpu>::random(&device),
            VarBuilder::<CpuAutodiff>::random(&device),
            SdxlConfig::tiny(),
            &gcfg,
            false,
        )
        .unwrap();
        let mut g = ScoreDistillationGuidance::new(pred, gcfg, 0);
        let rgb = Tensor::<CpuAutodiff, 4>::full([1, 16, 16, 3], 0.5, &device).require_grad();
        let prompt = PromptEmbedding {
            prompt: "score_9, a red ball, front view".into(),
            negative_prompt: String::new(),
            text_embeddings: None,
        };
        let out = Guidance::<CpuAutodiff>::step(&mut g, rgb.clone(), &prompt);
        assert!(out.grad_norm.is_finite());
        let grads = out.loss_sds.backward();
        let gr = rgb.grad(&grads).expect("gradient reaches the render");
        assert_eq!(gr.dims(), [1, 16, 16, 3]);
    }

    /// Every UNet / text-encoder / VAE-encoder tensor in the real checkpoint is
    /// consumed by our key mapping with the right shape. Skips without the file.
    #[test]
    fn key_mapping_covers_real_checkpoint() {
        let path = default_checkpoint();
        if !path.exists() {
            eprintln!(
                "{} not found; skipping checkpoint key-mapping test",
                path.display()
            );
            return;
        }
        let file = Arc::new(SafeTensorsFile::open(&path).unwrap());
        let device = Default::default();
        let vb = VarBuilder::<Cpu>::meta(file.clone(), &device);
        let cfg = SdxlConfig::sdxl();
        SdxlUnet::load(&vb.pp(UNET_PREFIX), cfg.unet.clone()).unwrap();
        for (prefix, c) in [(CLIP_L_PREFIX, &cfg.clip_l), (CLIP_G_PREFIX, &cfg.clip_g)] {
            ClipStem::load(&vb.pp(prefix), c).unwrap();
            clip::load_layers(&vb.pp(prefix), c).unwrap();
        }
        VaeEncoder::load(&vb.pp(VAE_PREFIX), cfg.vae.clone()).unwrap();
        let used = vb.used_keys();
        let all: BTreeSet<String> = file.tensors.keys().cloned().collect();
        let missing: Vec<_> = used.difference(&all).collect();
        assert!(missing.is_empty(), "mapped keys not in file: {missing:?}");
        let unmapped: Vec<_> = all
            .difference(&used)
            .filter(|k| !is_intentionally_unused(k))
            .collect();
        assert!(
            unmapped.is_empty(),
            "checkpoint keys not mapped: {unmapped:?}"
        );
        let unet_keys = all.iter().filter(|k| k.starts_with(UNET_PREFIX)).count();
        assert_eq!(
            unet_keys,
            used.iter().filter(|k| k.starts_with(UNET_PREFIX)).count()
        );
        eprintln!(
            "mapped {} of {} checkpoint tensors ({} UNet)",
            used.len(),
            all.len(),
            unet_keys
        );
    }
}
