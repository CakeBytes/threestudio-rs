//! SDXL UNet (`sgm.modules.diffusionmodules.openaimodel.UNetModel`) with the
//! original LDM/SGM parameter names, so weights load straight from a
//! single-file checkpoint under `model.diffusion_model.*`.
//!
//! Key mapping (LDM name → layer):
//!
//! | checkpoint key                                   | layer                                   |
//! |--------------------------------------------------|-----------------------------------------|
//! | `time_embed.{0,2}`                               | timestep MLP (320 → 1280 → 1280)        |
//! | `label_emb.0.{0,2}`                              | "text_time" added-cond MLP (2816 → 1280) |
//! | `input_blocks.0.0`                               | `conv_in`                               |
//! | `input_blocks.N.0.{in_layers.0,in_layers.2,emb_layers.1,out_layers.0,out_layers.3,skip_connection}` | ResBlock (GN, conv, emb proj, GN, conv, 1x1 skip) |
//! | `input_blocks.N.1.{norm,proj_in,transformer_blocks.K.*,proj_out}` | SpatialTransformer (linear proj) |
//! | `transformer_blocks.K.{norm1,attn1,norm2,attn2,norm3,ff.net.0.proj,ff.net.2}` | self-attn, cross-attn, GEGLU FF |
//! | `input_blocks.{3,6}.0.op`                        | stride-2 downsample conv                |
//! | `middle_block.{0,1,2}`                           | ResBlock, SpatialTransformer, ResBlock  |
//! | `output_blocks.N.{0,1}` / `output_blocks.{2,5}.{2,1}.conv` | ResBlock(+skip concat), transformer, upsample |
//! | `out.{0,2}`                                      | final GroupNorm + conv                  |

use super::nn::{
    attention, timestep_embedding, upsample_nearest2x, Conv2d, GroupNorm, LayerNorm, Linear,
};
use super::weights::VarBuilder;
use anyhow::Result;
use burn::prelude::*;
use burn::tensor::activation::{gelu, silu};

/// UNet hyper-parameters (`network_config.params` in SGM yaml).
#[derive(Debug, Clone, PartialEq)]
pub struct UnetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub model_channels: usize,
    pub channel_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// Transformer depth per resolution level (0 = no attention).
    pub transformer_depth: Vec<usize>,
    pub transformer_depth_middle: usize,
    pub num_head_channels: usize,
    pub context_dim: usize,
    /// Pooled text embedding size (OpenCLIP bigG projection = 1280).
    pub pooled_dim: usize,
    /// Per-value sinusoidal size for the 6 `time_ids` (256 in SDXL).
    pub addition_time_embed_dim: usize,
    pub norm_groups: usize,
}

impl UnetConfig {
    /// Stable Diffusion XL base 1.0 (also Pony Diffusion V6 XL).
    pub fn sdxl() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 320,
            channel_mult: vec![1, 2, 4],
            num_res_blocks: 2,
            transformer_depth: vec![0, 2, 10],
            transformer_depth_middle: 10,
            num_head_channels: 64,
            context_dim: 2048,
            pooled_dim: 1280,
            addition_time_embed_dim: 256,
            norm_groups: 32,
        }
    }

    /// Same topology as SDXL, tiny widths (used by CI tests).
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            model_channels: 32,
            channel_mult: vec![1, 2, 4],
            num_res_blocks: 2,
            transformer_depth: vec![0, 1, 2],
            transformer_depth_middle: 1,
            num_head_channels: 16,
            context_dim: 48,
            pooled_dim: 16,
            addition_time_embed_dim: 8,
            norm_groups: 8,
        }
    }

    pub fn time_embed_dim(&self) -> usize {
        self.model_channels * 4
    }

    pub fn adm_in_channels(&self) -> usize {
        self.pooled_dim + 6 * self.addition_time_embed_dim
    }
}

/// Static description of one layer inside a block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerSpec {
    ConvIn { in_ch: usize, out_ch: usize },
    Res { in_ch: usize, out_ch: usize },
    Attn { ch: usize, depth: usize },
    Down { ch: usize },
    Up { ch: usize },
}

/// The block plan of the UNet: input blocks, middle block, output blocks.
#[derive(Debug, Clone)]
pub struct UnetPlan {
    pub input: Vec<Vec<LayerSpec>>,
    pub middle: Vec<LayerSpec>,
    pub output: Vec<Vec<LayerSpec>>,
}

impl UnetPlan {
    pub fn new(cfg: &UnetConfig) -> Self {
        let mc = cfg.model_channels;
        let mut input = vec![vec![LayerSpec::ConvIn {
            in_ch: cfg.in_channels,
            out_ch: mc,
        }]];
        let mut chans = vec![mc];
        let mut ch = mc;
        let levels = cfg.channel_mult.len();
        for (level, &mult) in cfg.channel_mult.iter().enumerate() {
            for _ in 0..cfg.num_res_blocks {
                let out = mult * mc;
                let mut blk = vec![LayerSpec::Res {
                    in_ch: ch,
                    out_ch: out,
                }];
                if cfg.transformer_depth[level] > 0 {
                    blk.push(LayerSpec::Attn {
                        ch: out,
                        depth: cfg.transformer_depth[level],
                    });
                }
                ch = out;
                input.push(blk);
                chans.push(ch);
            }
            if level + 1 != levels {
                input.push(vec![LayerSpec::Down { ch }]);
                chans.push(ch);
            }
        }
        let middle = vec![
            LayerSpec::Res {
                in_ch: ch,
                out_ch: ch,
            },
            LayerSpec::Attn {
                ch,
                depth: cfg.transformer_depth_middle,
            },
            LayerSpec::Res {
                in_ch: ch,
                out_ch: ch,
            },
        ];
        let mut output = Vec::new();
        for (level, &mult) in cfg.channel_mult.iter().enumerate().rev() {
            for i in 0..=cfg.num_res_blocks {
                let skip = chans.pop().expect("skip channel");
                let out = mult * mc;
                let mut blk = vec![LayerSpec::Res {
                    in_ch: ch + skip,
                    out_ch: out,
                }];
                if cfg.transformer_depth[level] > 0 {
                    blk.push(LayerSpec::Attn {
                        ch: out,
                        depth: cfg.transformer_depth[level],
                    });
                }
                ch = out;
                if level > 0 && i == cfg.num_res_blocks {
                    blk.push(LayerSpec::Up { ch });
                }
                output.push(blk);
            }
        }
        Self {
            input,
            middle,
            output,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResBlock<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    emb_proj: Linear<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    skip: Option<Conv2d<B>>,
}

impl<B: Backend> ResBlock<B> {
    pub fn load(vb: &VarBuilder<B>, cfg: &UnetConfig, in_ch: usize, out_ch: usize) -> Result<Self> {
        let g = cfg.norm_groups;
        Ok(Self {
            norm1: GroupNorm::load(&vb.pp("in_layers.0"), g, in_ch, 1e-5)?,
            conv1: Conv2d::load(&vb.pp("in_layers.2"), in_ch, out_ch, 3, 1, 1)?,
            emb_proj: Linear::load(&vb.pp("emb_layers.1"), cfg.time_embed_dim(), out_ch, true)?,
            norm2: GroupNorm::load(&vb.pp("out_layers.0"), g, out_ch, 1e-5)?,
            conv2: Conv2d::load(&vb.pp("out_layers.3"), out_ch, out_ch, 3, 1, 1)?,
            skip: if in_ch != out_ch {
                Some(Conv2d::load(
                    &vb.pp("skip_connection"),
                    in_ch,
                    out_ch,
                    1,
                    1,
                    0,
                )?)
            } else {
                None
            },
        })
    }

    /// `emb_silu` is `silu(emb)`, shared by all res blocks.
    pub fn forward(&self, x: Tensor<B, 4>, emb_silu: Tensor<B, 2>) -> Tensor<B, 4> {
        let h = self.conv1.forward(silu(self.norm1.forward(x.clone())));
        let [b, c] = [h.dims()[0], h.dims()[1]];
        let h = h + self.emb_proj.forward2(emb_silu).reshape([b, c, 1, 1]);
        let h = self.conv2.forward(silu(self.norm2.forward(h)));
        match &self.skip {
            Some(s) => s.forward(x) + h,
            None => x + h,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CrossAttention<B: Backend> {
    to_q: Linear<B>,
    to_k: Linear<B>,
    to_v: Linear<B>,
    to_out: Linear<B>,
    heads: usize,
}

impl<B: Backend> CrossAttention<B> {
    pub fn load(vb: &VarBuilder<B>, dim: usize, ctx_dim: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            to_q: Linear::load(&vb.pp("to_q"), dim, dim, false)?,
            to_k: Linear::load(&vb.pp("to_k"), ctx_dim, dim, false)?,
            to_v: Linear::load(&vb.pp("to_v"), ctx_dim, dim, false)?,
            to_out: Linear::load(&vb.pp("to_out.0"), dim, dim, true)?,
            heads,
        })
    }

    pub fn forward(&self, x: Tensor<B, 3>, ctx: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let ctx = ctx.unwrap_or_else(|| x.clone());
        let q = self.to_q.forward3(x);
        let k = self.to_k.forward3(ctx.clone());
        let v = self.to_v.forward3(ctx);
        self.to_out.forward3(attention(q, k, v, self.heads, None))
    }
}

#[derive(Debug, Clone)]
pub struct BasicTransformerBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn1: CrossAttention<B>,
    norm2: LayerNorm<B>,
    attn2: CrossAttention<B>,
    norm3: LayerNorm<B>,
    ff_proj: Linear<B>,
    ff_out: Linear<B>,
}

impl<B: Backend> BasicTransformerBlock<B> {
    pub fn load(vb: &VarBuilder<B>, dim: usize, ctx_dim: usize, heads: usize) -> Result<Self> {
        let inner = dim * 4;
        Ok(Self {
            norm1: LayerNorm::load(&vb.pp("norm1"), dim, 1e-5)?,
            attn1: CrossAttention::load(&vb.pp("attn1"), dim, dim, heads)?,
            norm2: LayerNorm::load(&vb.pp("norm2"), dim, 1e-5)?,
            attn2: CrossAttention::load(&vb.pp("attn2"), dim, ctx_dim, heads)?,
            norm3: LayerNorm::load(&vb.pp("norm3"), dim, 1e-5)?,
            ff_proj: Linear::load(&vb.pp("ff.net.0.proj"), dim, inner * 2, true)?,
            ff_out: Linear::load(&vb.pp("ff.net.2"), inner, dim, true)?,
        })
    }

    pub fn forward(&self, x: Tensor<B, 3>, ctx: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.attn1.forward(self.norm1.forward(x.clone()), None) + x;
        let x = self.attn2.forward(self.norm2.forward(x.clone()), Some(ctx)) + x;
        // GEGLU
        let h = self.ff_proj.forward3(self.norm3.forward(x.clone()));
        let [b, n, c2] = h.dims();
        let parts = h.chunk(2, 2);
        let (a, gate) = (parts[0].clone(), parts[1].clone());
        debug_assert_eq!(a.dims(), [b, n, c2 / 2]);
        self.ff_out.forward3(a * gelu(gate)) + x
    }
}

#[derive(Debug, Clone)]
pub struct SpatialTransformer<B: Backend> {
    norm: GroupNorm<B>,
    proj_in: Linear<B>,
    blocks: Vec<BasicTransformerBlock<B>>,
    proj_out: Linear<B>,
}

impl<B: Backend> SpatialTransformer<B> {
    pub fn load(vb: &VarBuilder<B>, cfg: &UnetConfig, ch: usize, depth: usize) -> Result<Self> {
        let heads = ch / cfg.num_head_channels;
        let blocks = (0..depth)
            .map(|k| {
                BasicTransformerBlock::load(
                    &vb.pp("transformer_blocks").pp(k),
                    ch,
                    cfg.context_dim,
                    heads,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            norm: GroupNorm::load(&vb.pp("norm"), cfg.norm_groups, ch, 1e-6)?,
            proj_in: Linear::load(&vb.pp("proj_in"), ch, ch, true)?,
            blocks,
            proj_out: Linear::load(&vb.pp("proj_out"), ch, ch, true)?,
        })
    }

    pub fn forward(&self, x: Tensor<B, 4>, ctx: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let hs = self
            .norm
            .forward(x.clone())
            .reshape([b, c, h * w])
            .swap_dims(1, 2);
        let mut hs = self.proj_in.forward3(hs);
        for blk in &self.blocks {
            hs = blk.forward(hs, ctx.clone());
        }
        let hs = self
            .proj_out
            .forward3(hs)
            .swap_dims(1, 2)
            .reshape([b, c, h, w]);
        hs + x
    }
}

#[derive(Debug, Clone)]
pub enum UnetLayer<B: Backend> {
    Conv(Conv2d<B>),
    Res(ResBlock<B>),
    Attn(SpatialTransformer<B>),
    Down(Conv2d<B>),
    Up(Conv2d<B>),
}

impl<B: Backend> UnetLayer<B> {
    pub fn load(vb: &VarBuilder<B>, cfg: &UnetConfig, spec: LayerSpec, idx: usize) -> Result<Self> {
        let vb = vb.pp(idx);
        Ok(match spec {
            LayerSpec::ConvIn { in_ch, out_ch } => {
                Self::Conv(Conv2d::load(&vb, in_ch, out_ch, 3, 1, 1)?)
            }
            LayerSpec::Res { in_ch, out_ch } => Self::Res(ResBlock::load(&vb, cfg, in_ch, out_ch)?),
            LayerSpec::Attn { ch, depth } => {
                Self::Attn(SpatialTransformer::load(&vb, cfg, ch, depth)?)
            }
            LayerSpec::Down { ch } => Self::Down(Conv2d::load(&vb.pp("op"), ch, ch, 3, 2, 1)?),
            LayerSpec::Up { ch } => Self::Up(Conv2d::load(&vb.pp("conv"), ch, ch, 3, 1, 1)?),
        })
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        emb_silu: &Tensor<B, 2>,
        ctx: &Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        match self {
            Self::Conv(c) | Self::Down(c) => c.forward(x),
            Self::Res(r) => r.forward(x, emb_silu.clone()),
            Self::Attn(a) => a.forward(x, ctx.clone()),
            Self::Up(c) => c.forward(upsample_nearest2x(x)),
        }
    }
}

/// One block = a short sequence of layers.
pub type UnetBlock<B> = Vec<UnetLayer<B>>;

fn load_block<B: Backend>(
    vb: &VarBuilder<B>,
    cfg: &UnetConfig,
    specs: &[LayerSpec],
) -> Result<UnetBlock<B>> {
    specs
        .iter()
        .enumerate()
        .map(|(i, s)| UnetLayer::load(vb, cfg, *s, i))
        .collect()
}

fn run_block<B: Backend>(
    blk: &UnetBlock<B>,
    mut x: Tensor<B, 4>,
    emb: &Tensor<B, 2>,
    ctx: &Tensor<B, 3>,
) -> Tensor<B, 4> {
    for l in blk {
        x = l.forward(x, emb, ctx);
    }
    x
}

/// Time + added-condition embedding MLPs.
#[derive(Debug, Clone)]
pub struct UnetEmbeddings<B: Backend> {
    time1: Linear<B>,
    time2: Linear<B>,
    label1: Linear<B>,
    label2: Linear<B>,
}

impl<B: Backend> UnetEmbeddings<B> {
    pub fn load(vb: &VarBuilder<B>, cfg: &UnetConfig) -> Result<Self> {
        let (mc, ted) = (cfg.model_channels, cfg.time_embed_dim());
        Ok(Self {
            time1: Linear::load(&vb.pp("time_embed.0"), mc, ted, true)?,
            time2: Linear::load(&vb.pp("time_embed.2"), ted, ted, true)?,
            label1: Linear::load(&vb.pp("label_emb.0.0"), cfg.adm_in_channels(), ted, true)?,
            label2: Linear::load(&vb.pp("label_emb.0.2"), ted, ted, true)?,
        })
    }

    /// Returns `silu(emb)`, i.e. what every ResBlock's `emb_layers` consumes.
    pub fn forward(&self, cfg: &UnetConfig, timesteps: &[f32], y: Tensor<B, 2>) -> Tensor<B, 2> {
        let t = timestep_embedding::<B>(timesteps, cfg.model_channels, &y.device());
        let temb = self.time2.forward2(silu(self.time1.forward2(t)));
        let aemb = self.label2.forward2(silu(self.label1.forward2(y)));
        silu(temb + aemb)
    }
}

#[derive(Debug, Clone)]
pub struct UnetHead<B: Backend> {
    norm: GroupNorm<B>,
    conv: Conv2d<B>,
}

impl<B: Backend> UnetHead<B> {
    pub fn load(vb: &VarBuilder<B>, cfg: &UnetConfig) -> Result<Self> {
        Ok(Self {
            norm: GroupNorm::load(&vb.pp("out.0"), cfg.norm_groups, cfg.model_channels, 1e-5)?,
            conv: Conv2d::load(
                &vb.pp("out.2"),
                cfg.model_channels,
                cfg.out_channels,
                3,
                1,
                1,
            )?,
        })
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(silu(self.norm.forward(x)))
    }
}

/// Fully materialised SDXL UNet.
#[derive(Debug, Clone)]
pub struct SdxlUnet<B: Backend> {
    pub cfg: UnetConfig,
    emb: UnetEmbeddings<B>,
    input: Vec<UnetBlock<B>>,
    middle: UnetBlock<B>,
    output: Vec<UnetBlock<B>>,
    head: UnetHead<B>,
}

impl<B: Backend> SdxlUnet<B> {
    /// `vb` must point at the UNet root (`model.diffusion_model` in a checkpoint).
    pub fn load(vb: &VarBuilder<B>, cfg: UnetConfig) -> Result<Self> {
        let plan = UnetPlan::new(&cfg);
        let input = plan
            .input
            .iter()
            .enumerate()
            .map(|(i, s)| load_block(&vb.pp("input_blocks").pp(i), &cfg, s))
            .collect::<Result<_>>()?;
        let middle = load_block(&vb.pp("middle_block"), &cfg, &plan.middle)?;
        let output = plan
            .output
            .iter()
            .enumerate()
            .map(|(i, s)| load_block(&vb.pp("output_blocks").pp(i), &cfg, s))
            .collect::<Result<_>>()?;
        Ok(Self {
            emb: UnetEmbeddings::load(vb, &cfg)?,
            head: UnetHead::load(vb, &cfg)?,
            input,
            middle,
            output,
            cfg,
        })
    }

    /// ε-prediction. `x`: `[b, 4, h, w]` (h, w divisible by 4), `timesteps`: one per
    /// batch item, `context`: `[b, 77, context_dim]`, `y`: `[b, adm_in_channels]`.
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        timesteps: &[f32],
        context: Tensor<B, 3>,
        y: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let emb = self.emb.forward(&self.cfg, timesteps, y);
        let mut skips = Vec::new();
        let mut h = x;
        for blk in &self.input {
            h = run_block(blk, h, &emb, &context);
            skips.push(h.clone());
        }
        h = run_block(&self.middle, h, &emb, &context);
        for blk in &self.output {
            let s = skips.pop().expect("skip");
            h = run_block(blk, Tensor::cat(vec![h, s], 1), &emb, &context);
        }
        self.head.forward(h)
    }
}

/// Same computation as [`SdxlUnet::forward`], but loads each block from `vb`
/// right before running it and frees it afterwards. Peak memory is one block
/// (~1.7 GB f32 for SDXL's middle block) instead of the whole 10 GB UNet.
pub fn forward_streaming<B: Backend>(
    vb: &VarBuilder<B>,
    cfg: &UnetConfig,
    x: Tensor<B, 4>,
    timesteps: &[f32],
    context: Tensor<B, 3>,
    y: Tensor<B, 2>,
) -> Result<Tensor<B, 4>> {
    let plan = UnetPlan::new(cfg);
    let emb = UnetEmbeddings::load(vb, cfg)?.forward(cfg, timesteps, y);
    let mut skips = Vec::new();
    let mut h = x;
    for (i, specs) in plan.input.iter().enumerate() {
        let blk = load_block(&vb.pp("input_blocks").pp(i), cfg, specs)?;
        h = run_block(&blk, h, &emb, &context);
        skips.push(h.clone());
    }
    let blk = load_block(&vb.pp("middle_block"), cfg, &plan.middle)?;
    h = run_block(&blk, h, &emb, &context);
    drop(blk);
    for (i, specs) in plan.output.iter().enumerate() {
        let blk = load_block(&vb.pp("output_blocks").pp(i), cfg, specs)?;
        let s = skips.pop().expect("skip");
        h = run_block(&blk, Tensor::cat(vec![h, s], 1), &emb, &context);
    }
    Ok(UnetHead::load(vb, cfg)?.forward(h))
}

/// SDXL's vector conditioning `y = [pooled_text, emb(orig_size), emb(crop), emb(target_size)]`.
pub fn sdxl_vector_cond<B: Backend>(
    cfg: &UnetConfig,
    pooled: Tensor<B, 2>,
    time_ids: [f32; 6],
) -> Tensor<B, 2> {
    let b = pooled.dims()[0];
    let d = cfg.addition_time_embed_dim;
    let t = timestep_embedding::<B>(&time_ids, d, &pooled.device())
        .reshape([1, 6 * d])
        .repeat_dim(0, b);
    Tensor::cat(vec![pooled, t], 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use burn::tensor::Distribution;

    #[test]
    fn sdxl_plan_matches_ldm_layout() {
        let plan = UnetPlan::new(&UnetConfig::sdxl());
        assert_eq!(plan.input.len(), 9);
        assert_eq!(plan.output.len(), 9);
        assert_eq!(plan.input[3], vec![LayerSpec::Down { ch: 320 }]);
        assert_eq!(plan.input[4][1], LayerSpec::Attn { ch: 640, depth: 2 });
        assert_eq!(
            plan.output[0][0],
            LayerSpec::Res {
                in_ch: 2560,
                out_ch: 1280
            }
        );
        assert_eq!(plan.output[2][2], LayerSpec::Up { ch: 1280 });
        assert_eq!(plan.output[5][2], LayerSpec::Up { ch: 640 });
        assert_eq!(
            plan.output[8],
            vec![LayerSpec::Res {
                in_ch: 640,
                out_ch: 320
            }]
        );
        assert_eq!(UnetConfig::sdxl().adm_in_channels(), 2816);
    }

    #[test]
    fn tiny_unet_forward_shape() {
        let device = Default::default();
        let cfg = UnetConfig::tiny();
        let vb = VarBuilder::<Cpu>::random(&device);
        let unet = SdxlUnet::load(&vb, cfg.clone()).unwrap();
        let x = Tensor::<Cpu, 4>::random([2, 4, 16, 16], Distribution::Normal(0.0, 1.0), &device);
        let ctx = Tensor::<Cpu, 3>::random(
            [2, 77, cfg.context_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let pooled =
            Tensor::<Cpu, 2>::random([2, cfg.pooled_dim], Distribution::Normal(0.0, 1.0), &device);
        let y = sdxl_vector_cond(&cfg, pooled, [1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]);
        assert_eq!(y.dims(), [2, cfg.adm_in_channels()]);
        let out = unet.forward(x, &[500.0, 10.0], ctx, y);
        assert_eq!(out.dims(), [2, 4, 16, 16]);
        assert!(out
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
}
