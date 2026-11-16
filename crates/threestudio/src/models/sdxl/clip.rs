//! SDXL text encoders.
//!
//! * `conditioner.embedders.0.transformer.text_model.*` — CLIP ViT-L/14 (HF layout),
//!   penultimate hidden state (768-d), quick-GELU.
//! * `conditioner.embedders.1.model.*` — OpenCLIP ViT-bigG/14 (open_clip layout),
//!   penultimate hidden state (1280-d) plus the pooled, projected EOS embedding.

use super::nn::{attention, causal_mask, LayerNorm, Linear};
use super::tokenizer::{eos_position, CONTEXT_LENGTH};
use super::weights::{Init, VarBuilder};
use anyhow::Result;
use burn::prelude::*;
use burn::tensor::activation::{gelu, sigmoid};

/// Which checkpoint layout the encoder uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipLayout {
    /// HF `CLIPTextModel` names (CLIP ViT-L in SDXL).
    HuggingFace,
    /// `open_clip` names (ViT-bigG in SDXL).
    OpenClip,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClipConfig {
    pub layout: ClipLayout,
    pub vocab_size: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    /// Output projection size (OpenCLIP only).
    pub projection_dim: Option<usize>,
    /// Padding token id used by the matching tokenizer.
    pub pad_id: u32,
}

impl ClipConfig {
    pub fn vit_l() -> Self {
        Self {
            layout: ClipLayout::HuggingFace,
            vocab_size: 49408,
            width: 768,
            layers: 12,
            heads: 12,
            projection_dim: None,
            pad_id: super::tokenizer::EOS,
        }
    }

    pub fn vit_bigg() -> Self {
        Self {
            layout: ClipLayout::OpenClip,
            vocab_size: 49408,
            width: 1280,
            layers: 32,
            heads: 20,
            projection_dim: Some(1280),
            pad_id: 0,
        }
    }

    pub fn tiny(layout: ClipLayout, width: usize) -> Self {
        Self {
            layout,
            vocab_size: 49408,
            width,
            layers: 2,
            heads: 2,
            projection_dim: (layout == ClipLayout::OpenClip).then_some(width),
            pad_id: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClipLayer<B: Backend> {
    ln1: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    ln2: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    quick_gelu: bool,
    heads: usize,
}

impl<B: Backend> ClipLayer<B> {
    pub fn load(vb: &VarBuilder<B>, cfg: &ClipConfig) -> Result<Self> {
        let w = cfg.width;
        Ok(match cfg.layout {
            ClipLayout::HuggingFace => Self {
                ln1: LayerNorm::load(&vb.pp("layer_norm1"), w, 1e-5)?,
                q: Linear::load(&vb.pp("self_attn.q_proj"), w, w, true)?,
                k: Linear::load(&vb.pp("self_attn.k_proj"), w, w, true)?,
                v: Linear::load(&vb.pp("self_attn.v_proj"), w, w, true)?,
                out: Linear::load(&vb.pp("self_attn.out_proj"), w, w, true)?,
                ln2: LayerNorm::load(&vb.pp("layer_norm2"), w, 1e-5)?,
                fc1: Linear::load(&vb.pp("mlp.fc1"), w, 4 * w, true)?,
                fc2: Linear::load(&vb.pp("mlp.fc2"), 4 * w, w, true)?,
                quick_gelu: true,
                heads: cfg.heads,
            },
            ClipLayout::OpenClip => {
                // fused in_proj: [3w, w] -> split into q, k, v
                let wqkv = vb
                    .pp("attn")
                    .get_linear_weight("in_proj_weight", 3 * w, w)?; // [w, 3w]
                let bqkv = vb
                    .pp("attn")
                    .get::<1>("in_proj_bias", [3 * w], Init::Zeros)?;
                let meta = vb.is_meta();
                let part = |i: usize| -> Linear<B> {
                    if meta {
                        return Linear {
                            weight: wqkv.clone(),
                            bias: Some(bqkv.clone()),
                        };
                    }
                    Linear {
                        weight: wqkv.clone().slice([0..w, i * w..(i + 1) * w]),
                        bias: Some(bqkv.clone().narrow(0, i * w, w)),
                    }
                };
                Self {
                    ln1: LayerNorm::load(&vb.pp("ln_1"), w, 1e-5)?,
                    q: part(0),
                    k: part(1),
                    v: part(2),
                    out: Linear::load(&vb.pp("attn.out_proj"), w, w, true)?,
                    ln2: LayerNorm::load(&vb.pp("ln_2"), w, 1e-5)?,
                    fc1: Linear::load(&vb.pp("mlp.c_fc"), w, 4 * w, true)?,
                    fc2: Linear::load(&vb.pp("mlp.c_proj"), 4 * w, w, true)?,
                    quick_gelu: false,
                    heads: cfg.heads,
                }
            }
        })
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: &Tensor<B, 2>) -> Tensor<B, 3> {
        let h = self.ln1.forward(x.clone());
        let a = attention(
            self.q.forward3(h.clone()),
            self.k.forward3(h.clone()),
            self.v.forward3(h),
            self.heads,
            Some(mask.clone()),
        );
        let x = x + self.out.forward3(a);
        let h = self.fc1.forward3(self.ln2.forward(x.clone()));
        let h = if self.quick_gelu {
            h.clone() * sigmoid(h.mul_scalar(1.702))
        } else {
            gelu(h)
        };
        x + self.fc2.forward3(h)
    }
}

/// Embeddings + final norm/projection (the small, always-resident parts).
#[derive(Debug, Clone)]
pub struct ClipStem<B: Backend> {
    token_embedding: Tensor<B, 2>,
    position_embedding: Tensor<B, 2>,
    final_ln: LayerNorm<B>,
    text_projection: Option<Tensor<B, 2>>,
}

fn layer_vb<B: Backend>(vb: &VarBuilder<B>, cfg: &ClipConfig, i: usize) -> VarBuilder<B> {
    match cfg.layout {
        ClipLayout::HuggingFace => vb.pp("text_model.encoder.layers").pp(i),
        ClipLayout::OpenClip => vb.pp("transformer.resblocks").pp(i),
    }
}

impl<B: Backend> ClipStem<B> {
    pub fn load(vb: &VarBuilder<B>, cfg: &ClipConfig) -> Result<Self> {
        let (v, w) = (cfg.vocab_size, cfg.width);
        Ok(match cfg.layout {
            ClipLayout::HuggingFace => Self {
                token_embedding: vb.get(
                    "text_model.embeddings.token_embedding.weight",
                    [v, w],
                    Init::Normal(0.02),
                )?,
                position_embedding: vb.get(
                    "text_model.embeddings.position_embedding.weight",
                    [CONTEXT_LENGTH, w],
                    Init::Normal(0.01),
                )?,
                final_ln: LayerNorm::load(&vb.pp("text_model.final_layer_norm"), w, 1e-5)?,
                text_projection: None,
            },
            ClipLayout::OpenClip => Self {
                token_embedding: vb.get("token_embedding.weight", [v, w], Init::Normal(0.02))?,
                position_embedding: vb.get(
                    "positional_embedding",
                    [CONTEXT_LENGTH, w],
                    Init::Normal(0.01),
                )?,
                final_ln: LayerNorm::load(&vb.pp("ln_final"), w, 1e-5)?,
                text_projection: match cfg.projection_dim {
                    Some(p) => Some(vb.get("text_projection", [w, p], Init::Normal(0.02))?),
                    None => None,
                },
            },
        })
    }

    fn embed(&self, ids: &[Vec<u32>], device: &B::Device) -> Tensor<B, 3> {
        let n = ids.len();
        let flat: Vec<i64> = ids.iter().flatten().map(|&t| t as i64).collect();
        let idx =
            Tensor::<B, 1, Int>::from_data(TensorData::new(flat, [n * CONTEXT_LENGTH]), device);
        let w = self.token_embedding.dims()[1];
        let tok = self
            .token_embedding
            .clone()
            .select(0, idx)
            .reshape([n, CONTEXT_LENGTH, w]);
        tok + self.position_embedding.clone().unsqueeze_dim(0)
    }
}

/// Encoder output: penultimate hidden states and (OpenCLIP) the pooled embedding.
pub struct ClipOutput<B: Backend> {
    /// `[n, 77, width]`
    pub penultimate: Tensor<B, 3>,
    /// `[n, projection_dim]`
    pub pooled: Option<Tensor<B, 2>>,
}

/// Run a CLIP text encoder, loading each transformer layer from `vb` on demand.
/// With `resident = Some(layers)` the given pre-loaded layers are used instead.
pub fn encode<B: Backend>(
    vb: &VarBuilder<B>,
    cfg: &ClipConfig,
    stem: &ClipStem<B>,
    resident: Option<&[ClipLayer<B>]>,
    ids: &[Vec<u32>],
) -> Result<ClipOutput<B>> {
    let device = vb.device.clone();
    let mask = causal_mask::<B>(CONTEXT_LENGTH, &device);
    let mut x = stem.embed(ids, &device);
    let mut penultimate = None;
    for i in 0..cfg.layers {
        if i + 1 == cfg.layers {
            penultimate = Some(x.clone());
            if stem.text_projection.is_none() {
                break; // CLIP-L: only the penultimate state is used
            }
        }
        x = match resident {
            Some(layers) => layers[i].forward(x, &mask),
            None => ClipLayer::load(&layer_vb(vb, cfg, i), cfg)?.forward(x, &mask),
        };
    }
    let pooled = match &stem.text_projection {
        Some(proj) => {
            let last = stem.final_ln.forward(x);
            let w = cfg.width;
            let rows: Vec<Tensor<B, 2>> = ids
                .iter()
                .enumerate()
                .map(|(b, row)| {
                    let p = eos_position(row);
                    last.clone()
                        .slice([b..b + 1, p..p + 1, 0..w])
                        .reshape([1, w])
                })
                .collect();
            Some(Tensor::cat(rows, 0).matmul(proj.clone()))
        }
        None => None,
    };
    Ok(ClipOutput {
        penultimate: penultimate.expect("at least one layer"),
        pooled,
    })
}

/// Load every layer (fast path when memory allows).
pub fn load_layers<B: Backend>(vb: &VarBuilder<B>, cfg: &ClipConfig) -> Result<Vec<ClipLayer<B>>> {
    (0..cfg.layers)
        .map(|i| ClipLayer::load(&layer_vb(vb, cfg, i), cfg))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use crate::models::sdxl::tokenizer::ClipTokenizer;

    #[test]
    fn tiny_encoders_produce_expected_shapes() {
        let device = Default::default();
        let tok = ClipTokenizer::new();
        let ids = vec![tok.encode("score_9, a red ball", 0), tok.encode("", 0)];
        for layout in [ClipLayout::HuggingFace, ClipLayout::OpenClip] {
            let cfg = ClipConfig::tiny(layout, 32);
            let vb = VarBuilder::<Cpu>::random(&device);
            let stem = ClipStem::load(&vb, &cfg).unwrap();
            let out = encode(&vb, &cfg, &stem, None, &ids).unwrap();
            assert_eq!(out.penultimate.dims(), [2, 77, 32]);
            assert_eq!(
                out.pooled.map(|p| p.dims()),
                (layout == ClipLayout::OpenClip).then_some([2, 32])
            );
        }
    }
}
