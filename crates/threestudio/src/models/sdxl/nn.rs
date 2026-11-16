//! Frozen inference layers loaded through [`VarBuilder`]. Parameters are plain
//! tensors (not `Param`s), so autodiff never tracks them: gradients only flow
//! through the activations, which is what score distillation needs.

use super::weights::{Init, VarBuilder};
use anyhow::Result;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;

#[derive(Debug, Clone)]
pub struct Linear<B: Backend> {
    /// `[in, out]`
    pub weight: Tensor<B, 2>,
    pub bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> Linear<B> {
    pub fn load(vb: &VarBuilder<B>, in_dim: usize, out_dim: usize, bias: bool) -> Result<Self> {
        Ok(Self {
            weight: vb.get_linear_weight("weight", out_dim, in_dim)?,
            bias: if bias {
                Some(vb.get("bias", [out_dim], Init::Zeros)?)
            } else {
                None
            },
        })
    }

    pub fn forward2(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let y = x.matmul(self.weight.clone());
        match &self.bias {
            Some(b) => y + b.clone().unsqueeze_dim(0),
            None => y,
        }
    }

    /// `[b, n, in] -> [b, n, out]`
    pub fn forward3(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, c] = x.dims();
        let y = self.forward2(x.reshape([b * n, c]));
        let out = y.dims()[1];
        y.reshape([b, n, out])
    }
}

#[derive(Debug, Clone)]
pub struct Conv2d<B: Backend> {
    pub weight: Tensor<B, 4>,
    pub bias: Option<Tensor<B, 1>>,
    pub stride: usize,
    pub padding: usize,
}

impl<B: Backend> Conv2d<B> {
    pub fn load(
        vb: &VarBuilder<B>,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let fan_in = (in_ch * kernel * kernel) as f64;
        Ok(Self {
            weight: vb.get(
                "weight",
                [out_ch, in_ch, kernel, kernel],
                Init::Normal(1.0 / fan_in.sqrt()),
            )?,
            bias: Some(vb.get("bias", [out_ch], Init::Zeros)?),
            stride,
            padding,
        })
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        conv2d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            ConvOptions::new([self.stride; 2], [self.padding; 2], [1, 1], 1),
        )
    }
}

#[derive(Debug, Clone)]
pub struct GroupNorm<B: Backend> {
    pub gamma: Tensor<B, 1>,
    pub beta: Tensor<B, 1>,
    pub groups: usize,
    pub eps: f64,
}

impl<B: Backend> GroupNorm<B> {
    pub fn load(vb: &VarBuilder<B>, groups: usize, channels: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            gamma: vb.get("weight", [channels], Init::Ones)?,
            beta: vb.get("bias", [channels], Init::Zeros)?,
            groups,
            eps,
        })
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let g = self.groups;
        let xg = x.reshape([b, g, (c / g) * h * w]);
        let mean = xg.clone().mean_dim(2);
        let xc = xg - mean;
        let var = xc.clone().powf_scalar(2.0).mean_dim(2);
        let xn = (xc / var.add_scalar(self.eps).sqrt()).reshape([b, c, h, w]);
        xn * self.gamma.clone().reshape([1, c, 1, 1]) + self.beta.clone().reshape([1, c, 1, 1])
    }
}

#[derive(Debug, Clone)]
pub struct LayerNorm<B: Backend> {
    pub gamma: Tensor<B, 1>,
    pub beta: Tensor<B, 1>,
    pub eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    pub fn load(vb: &VarBuilder<B>, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            gamma: vb.get("weight", [dim], Init::Ones)?,
            beta: vb.get("bias", [dim], Init::Zeros)?,
            eps,
        })
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean = x.clone().mean_dim(2);
        let xc = x - mean;
        let var = xc.clone().powf_scalar(2.0).mean_dim(2);
        let xn = xc / var.add_scalar(self.eps).sqrt();
        xn * self.gamma.clone().reshape([1, 1, -1]) + self.beta.clone().reshape([1, 1, -1])
    }
}

/// Multi-head scaled dot-product attention on `[b, n, c]` tensors.
/// `mask` (if any) is additive, shaped `[n_q, n_k]`.
pub fn attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    heads: usize,
    mask: Option<Tensor<B, 2>>,
) -> Tensor<B, 3> {
    let [b, nq, c] = q.dims();
    let nk = k.dims()[1];
    let d = c / heads;
    let split = |t: Tensor<B, 3>, n: usize| t.reshape([b, n, heads, d]).swap_dims(1, 2);
    let (q, k, v) = (split(q, nq), split(k, nk), split(v, nk));
    let mut scores = q
        .matmul(k.swap_dims(2, 3))
        .mul_scalar(1.0 / (d as f64).sqrt());
    if let Some(m) = mask {
        scores = scores + m.reshape([1, 1, nq, nk]);
    }
    softmax(scores, 3)
        .matmul(v)
        .swap_dims(1, 2)
        .reshape([b, nq, c])
}

/// Additive causal mask `[n, n]` (0 on/below the diagonal, -1e9 above).
pub fn causal_mask<B: Backend>(n: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            m[i * n + j] = -1e9;
        }
    }
    Tensor::from_data(TensorData::new(m, [n, n]), device)
}

/// Sinusoidal embedding used by SGM/LDM (`timestep_embedding`): `[cos, sin]`.
pub fn timestep_embedding<B: Backend>(
    values: &[f32],
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let half = dim / 2;
    let mut out = Vec::with_capacity(values.len() * dim);
    for &t in values {
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10000f32).ln() * i as f32 / half as f32).exp() * t)
            .collect();
        out.extend(freqs.iter().map(|a| a.cos()));
        out.extend(freqs.iter().map(|a| a.sin()));
    }
    Tensor::from_data(TensorData::new(out, [values.len(), dim]), device)
}

/// 1-D linear interpolation matrix `[out, in]` with PyTorch's
/// `align_corners=False` (half-pixel) convention.
fn linear_resize_matrix(n_in: usize, n_out: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n_out * n_in];
    let scale = n_in as f32 / n_out as f32;
    for o in 0..n_out {
        let src = ((o as f32 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (src.floor() as usize).min(n_in - 1);
        let i1 = (i0 + 1).min(n_in - 1);
        let f = src - i0 as f32;
        m[o * n_in + i0] += 1.0 - f;
        m[o * n_in + i1] += f;
    }
    m
}

/// Bilinear resize (`F.interpolate(..., mode="bilinear", align_corners=False)`)
/// written as two matmuls, so it is differentiable on every backend
/// (Burn's ndarray backend has no bilinear-interpolate backward).
pub fn resize_bilinear<B: Backend>(x: Tensor<B, 4>, out_h: usize, out_w: usize) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    let device = x.device();
    let rx = Tensor::<B, 2>::from_data(
        TensorData::new(linear_resize_matrix(w, out_w), [out_w, w]),
        &device,
    );
    let ry = Tensor::<B, 2>::from_data(
        TensorData::new(linear_resize_matrix(h, out_h), [out_h, h]),
        &device,
    );
    let x = x
        .reshape([b * c * h, w])
        .matmul(rx.transpose())
        .reshape([b, c, h, out_w]);
    let x = x
        .swap_dims(2, 3)
        .reshape([b * c * out_w, h])
        .matmul(ry.transpose());
    x.reshape([b, c, out_w, out_h]).swap_dims(2, 3)
}

/// Nearest-neighbour 2x upsampling.
pub fn upsample_nearest2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    x.reshape([b, c, h, 1, w, 1])
        .repeat_dim(3, 2)
        .repeat_dim(5, 2)
        .reshape([b, c, h * 2, w * 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;

    #[test]
    fn group_norm_normalizes() {
        let vb = VarBuilder::<Cpu>::random(&Default::default());
        let gn = GroupNorm::load(&vb.pp("gn"), 2, 4, 1e-5).unwrap();
        let x = Tensor::<Cpu, 4>::random(
            [1, 4, 3, 3],
            burn::tensor::Distribution::Normal(3.0, 2.0),
            &Default::default(),
        );
        let y = gn.forward(x).into_data().to_vec::<f32>().unwrap();
        let mean: f32 = y[..18].iter().sum::<f32>() / 18.0;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn bilinear_resize_matches_torch_convention() {
        let x = Tensor::<Cpu, 4>::from_floats([[[[0.0, 1.0], [2.0, 3.0]]]], &Default::default());
        let y = resize_bilinear(x, 4, 4)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // torch: F.interpolate(x, size=4, mode="bilinear", align_corners=False)[0,0,0]
        assert_eq!(&y[..4], &[0.0, 0.25, 0.75, 1.0]);
        assert!((y[15] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn upsample_shape_and_values() {
        let x = Tensor::<Cpu, 4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &Default::default());
        let y = upsample_nearest2x(x);
        assert_eq!(y.dims(), [1, 1, 4, 4]);
        let v = y.into_data().to_vec::<f32>().unwrap();
        assert_eq!(&v[..4], &[1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn causal_attention_ignores_future() {
        let d = Default::default();
        let q =
            Tensor::<Cpu, 3>::random([1, 3, 4], burn::tensor::Distribution::Normal(0.0, 1.0), &d);
        let k = q.clone();
        let mut vdata = vec![0.0f32; 12];
        vdata[8..].copy_from_slice(&[100.0; 4]); // only the last token has a large value
        let v = Tensor::<Cpu, 3>::from_data(TensorData::new(vdata, [1, 3, 4]), &d);
        let out = attention(q, k, v, 2, Some(causal_mask::<Cpu>(3, &d)));
        let o = out.into_data().to_vec::<f32>().unwrap();
        assert!(o[..8].iter().all(|x| x.abs() < 1e-3));
    }

    #[test]
    fn timestep_embedding_layout() {
        let e = timestep_embedding::<Cpu>(&[0.0], 8, &Default::default());
        let v = e.into_data().to_vec::<f32>().unwrap();
        assert_eq!(&v[..4], &[1.0; 4]);
        assert_eq!(&v[4..], &[0.0; 4]);
    }
}
