//! SDXL VAE encoder (`first_stage_model.encoder.*` + `quant_conv`), LDM key names.
//! Gradients flow through it (weights are frozen), as in threestudio's
//! `encode_images`.

use super::nn::{attention, Conv2d, GroupNorm};
use super::weights::VarBuilder;
use anyhow::Result;
use burn::prelude::*;
use burn::tensor::activation::silu;

#[derive(Debug, Clone, PartialEq)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub z_channels: usize,
    pub groups: usize,
    pub scaling_factor: f32,
}

impl VaeConfig {
    pub fn sdxl() -> Self {
        Self {
            in_channels: 3,
            ch: 128,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            z_channels: 4,
            groups: 32,
            scaling_factor: 0.13025,
        }
    }

    pub fn tiny() -> Self {
        Self {
            ch: 16,
            ch_mult: vec![1, 2, 2, 2],
            groups: 8,
            ..Self::sdxl()
        }
    }

    /// Spatial down-sampling factor (8 for SDXL).
    pub fn factor(&self) -> usize {
        1 << (self.ch_mult.len() - 1)
    }
}

#[derive(Debug, Clone)]
struct VaeResBlock<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    shortcut: Option<Conv2d<B>>,
}

impl<B: Backend> VaeResBlock<B> {
    fn load(vb: &VarBuilder<B>, g: usize, cin: usize, cout: usize) -> Result<Self> {
        Ok(Self {
            norm1: GroupNorm::load(&vb.pp("norm1"), g, cin, 1e-6)?,
            conv1: Conv2d::load(&vb.pp("conv1"), cin, cout, 3, 1, 1)?,
            norm2: GroupNorm::load(&vb.pp("norm2"), g, cout, 1e-6)?,
            conv2: Conv2d::load(&vb.pp("conv2"), cout, cout, 3, 1, 1)?,
            shortcut: if cin != cout {
                Some(Conv2d::load(&vb.pp("nin_shortcut"), cin, cout, 1, 1, 0)?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.conv1.forward(silu(self.norm1.forward(x.clone())));
        let h = self.conv2.forward(silu(self.norm2.forward(h)));
        match &self.shortcut {
            Some(s) => s.forward(x) + h,
            None => x + h,
        }
    }
}

#[derive(Debug, Clone)]
struct VaeAttn<B: Backend> {
    norm: GroupNorm<B>,
    q: Conv2d<B>,
    k: Conv2d<B>,
    v: Conv2d<B>,
    proj: Conv2d<B>,
}

impl<B: Backend> VaeAttn<B> {
    fn load(vb: &VarBuilder<B>, g: usize, c: usize) -> Result<Self> {
        Ok(Self {
            norm: GroupNorm::load(&vb.pp("norm"), g, c, 1e-6)?,
            q: Conv2d::load(&vb.pp("q"), c, c, 1, 1, 0)?,
            k: Conv2d::load(&vb.pp("k"), c, c, 1, 1, 0)?,
            v: Conv2d::load(&vb.pp("v"), c, c, 1, 1, 0)?,
            proj: Conv2d::load(&vb.pp("proj_out"), c, c, 1, 1, 0)?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let hn = self.norm.forward(x.clone());
        let flat = |t: Tensor<B, 4>| t.reshape([b, c, h * w]).swap_dims(1, 2);
        let a = attention(
            flat(self.q.forward(hn.clone())),
            flat(self.k.forward(hn.clone())),
            flat(self.v.forward(hn)),
            1,
            None,
        );
        x + self.proj.forward(a.swap_dims(1, 2).reshape([b, c, h, w]))
    }
}

/// Res blocks of one resolution level plus its optional downsample conv.
type VaeLevel<B> = (Vec<VaeResBlock<B>>, Option<Conv2d<B>>);

#[derive(Debug, Clone)]
pub struct VaeEncoder<B: Backend> {
    pub cfg: VaeConfig,
    conv_in: Conv2d<B>,
    levels: Vec<VaeLevel<B>>,
    mid1: VaeResBlock<B>,
    mid_attn: VaeAttn<B>,
    mid2: VaeResBlock<B>,
    norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
    quant_conv: Conv2d<B>,
}

impl<B: Backend> VaeEncoder<B> {
    /// `vb` points at `first_stage_model`.
    pub fn load(vb: &VarBuilder<B>, cfg: VaeConfig) -> Result<Self> {
        let e = vb.pp("encoder");
        let g = cfg.groups;
        let mut levels = Vec::new();
        let mut cin = cfg.ch;
        for (i, &m) in cfg.ch_mult.iter().enumerate() {
            let cout = cfg.ch * m;
            let mut blocks = Vec::new();
            for j in 0..cfg.num_res_blocks {
                blocks.push(VaeResBlock::load(
                    &e.pp("down").pp(i).pp("block").pp(j),
                    g,
                    cin,
                    cout,
                )?);
                cin = cout;
            }
            let down = if i + 1 != cfg.ch_mult.len() {
                Some(Conv2d::load(
                    &e.pp("down").pp(i).pp("downsample.conv"),
                    cout,
                    cout,
                    3,
                    2,
                    0,
                )?)
            } else {
                None
            };
            levels.push((blocks, down));
        }
        let z2 = 2 * cfg.z_channels;
        Ok(Self {
            conv_in: Conv2d::load(&e.pp("conv_in"), cfg.in_channels, cfg.ch, 3, 1, 1)?,
            mid1: VaeResBlock::load(&e.pp("mid.block_1"), g, cin, cin)?,
            mid_attn: VaeAttn::load(&e.pp("mid.attn_1"), g, cin)?,
            mid2: VaeResBlock::load(&e.pp("mid.block_2"), g, cin, cin)?,
            norm_out: GroupNorm::load(&e.pp("norm_out"), g, cin, 1e-6)?,
            conv_out: Conv2d::load(&e.pp("conv_out"), cin, z2, 3, 1, 1)?,
            quant_conv: Conv2d::load(&vb.pp("quant_conv"), z2, z2, 1, 1, 0)?,
            levels,
            cfg,
        })
    }

    /// `x`: `[b, 3, H, W]` in `[-1, 1]` → scaled latent mean `[b, 4, H/8, W/8]`.
    pub fn encode(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut h = self.conv_in.forward(x);
        for (blocks, down) in &self.levels {
            for blk in blocks {
                h = blk.forward(h);
            }
            if let Some(d) = down {
                // LDM's asymmetric (0, 1, 0, 1) padding before the stride-2 conv
                h = d.forward(h.pad((0, 1, 0, 1), 0.0));
            }
        }
        let h = self
            .mid2
            .forward(self.mid_attn.forward(self.mid1.forward(h)));
        let moments = self
            .quant_conv
            .forward(self.conv_out.forward(silu(self.norm_out.forward(h))));
        let [b, _, hh, ww] = moments.dims();
        let z = self.cfg.z_channels;
        moments
            .slice([0..b, 0..z, 0..hh, 0..ww])
            .mul_scalar(self.cfg.scaling_factor)
    }
}

/// Documented fallback: a linear RGB→latent map (pseudo-inverse of the widely
/// used SDXL latent→RGB preview factors) applied after 8×8 average pooling.
/// Much cheaper than the VAE but only approximately on the latent manifold.
pub fn rgb_to_latent_linear<B: Backend>(x: Tensor<B, 4>, factor: usize) -> Tensor<B, 4> {
    // latent -> rgb factors (rows: latent channels), as used for SDXL previews
    const F: [[f32; 3]; 4] = [
        [0.3651, 0.4232, 0.4341],
        [-0.2533, -0.0042, 0.1068],
        [0.1076, 0.1111, -0.0362],
        [-0.3165, -0.2492, -0.2188],
    ];
    const BIAS: [f32; 3] = [0.1084, -0.0175, -0.0011];
    // P = (FᵀF)⁻¹ Fᵀ  (3x4), so that latent = (rgb - bias) · P
    let mut ftf = [[0.0f32; 3]; 3];
    for (i, row) in ftf.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..4).map(|k| F[k][i] * F[k][j]).sum();
        }
    }
    let inv = invert3(ftf);
    let mut p = vec![0.0f32; 12];
    for i in 0..3 {
        for k in 0..4 {
            p[i * 4 + k] = (0..3).map(|j| inv[i][j] * F[k][j]).sum();
        }
    }
    let device = x.device();
    let [b, _, h, w] = x.dims();
    let pooled =
        burn::tensor::module::avg_pool2d(x, [factor, factor], [factor, factor], [0, 0], false);
    let (lh, lw) = (h / factor, w / factor);
    let rgb = (pooled.permute([0, 2, 3, 1]).reshape([b * lh * lw, 3]) + 1.0) / 2.0;
    let bias = Tensor::<B, 1>::from_floats(BIAS, &device).reshape([1, 3]);
    let pm = Tensor::<B, 2>::from_data(TensorData::new(p, [3, 4]), &device);
    (rgb - bias)
        .matmul(pm)
        .reshape([b, lh, lw, 4])
        .permute([0, 3, 1, 2])
}

fn invert3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let mut r = [[0.0f32; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            let (a, b) = ((j + 1) % 3, (j + 2) % 3);
            let (c, d) = ((i + 1) % 3, (i + 2) % 3);
            *v = (m[a][c] * m[b][d] - m[a][d] * m[b][c]) / det;
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use burn::tensor::Distribution;

    #[test]
    fn tiny_vae_encoder_shape() {
        let device = Default::default();
        let vb = VarBuilder::<Cpu>::random(&device);
        let enc = VaeEncoder::load(&vb, VaeConfig::tiny()).unwrap();
        let x = Tensor::<Cpu, 4>::random([1, 3, 32, 32], Distribution::Uniform(-1.0, 1.0), &device);
        assert_eq!(enc.encode(x).dims(), [1, 4, 4, 4]);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn inverse_3x3() {
        let m = [[2.0, 0.0, 1.0], [1.0, 3.0, 0.0], [0.0, 1.0, 4.0]];
        let r = invert3(m);
        for i in 0..3 {
            for j in 0..3 {
                let v: f32 = (0..3).map(|k| m[i][k] * r[k][j]).sum();
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn linear_fallback_roundtrips_preview_factors() {
        let device = Default::default();
        let x = Tensor::<Cpu, 4>::full([1, 3, 16, 16], 0.2, &device);
        let z = rgb_to_latent_linear(x, 8);
        assert_eq!(z.dims(), [1, 4, 2, 2]);
    }
}
