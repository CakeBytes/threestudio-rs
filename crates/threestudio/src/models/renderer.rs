//! NeRF volume renderer (`threestudio/models/renderers/nerf_volume_renderer.py`).
//!
//! threestudio uses nerfacc's occupancy-grid ray marching; this port uses
//! stratified sampling between `near` and `far` with dense compositing, which
//! is differentiable with any Burn autodiff backend.

use super::geometry::ImplicitVolume;
use crate::config::RendererConfig;
use burn::prelude::*;
use burn::tensor::Distribution;

/// Renderer outputs, all per ray.
pub struct RenderOutput<B: Backend> {
    /// `[n_rays, 3]` colour composited over the background.
    pub comp_rgb: Tensor<B, 2>,
    /// `[n_rays, 1]` accumulated opacity, in `[0, 1]`.
    pub opacity: Tensor<B, 2>,
    /// `[n_rays, 1]` expected depth.
    pub depth: Tensor<B, 2>,
    /// `[n_rays, n_samples]` compositing weights.
    pub weights: Tensor<B, 2>,
}

#[derive(Debug, Clone)]
pub struct NeRFVolumeRenderer {
    pub cfg: RendererConfig,
}

/// Exclusive prefix sum along dim 1 of a `[n, s]` tensor, via a strictly
/// upper-triangular matmul (differentiable on every backend).
pub fn exclusive_cumsum<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let [_, s] = x.dims();
    let mut tri = vec![0.0f32; s * s];
    for i in 0..s {
        for j in (i + 1)..s {
            tri[i * s + j] = 1.0;
        }
    }
    let tri = Tensor::<B, 2>::from_data(TensorData::new(tri, [s, s]), &x.device());
    x.matmul(tri)
}

/// Compositing weights from `sigma * delta`; tensor version of
/// [`threestudio_kernels::alpha_composite_weights`].
pub fn composite_weights<B: Backend>(sigma_delta: Tensor<B, 2>) -> Tensor<B, 2> {
    let transmittance = exclusive_cumsum(sigma_delta.clone()).neg().exp();
    let alpha = sigma_delta.neg().exp().neg().add_scalar(1.0);
    transmittance * alpha
}

impl NeRFVolumeRenderer {
    pub fn new(cfg: RendererConfig) -> Self {
        Self { cfg }
    }

    /// Render `[n_rays, 3]` rays through `geometry` over a constant background colour.
    pub fn render<B: Backend>(
        &self,
        geometry: &ImplicitVolume<B>,
        rays_o: Tensor<B, 2>,
        rays_d: Tensor<B, 2>,
        background: [f32; 3],
        randomized: bool,
    ) -> RenderOutput<B> {
        let device = rays_o.device();
        let [n_rays, _] = rays_o.dims();
        let s = self.cfg.num_samples_per_ray;
        let (near, far) = (self.cfg.near, self.cfg.far);
        let dt = (far - near) / s as f32;

        let mids: Vec<f32> = (0..s).map(|i| near + (i as f32 + 0.5) * dt).collect();
        let mut t = Tensor::<B, 1>::from_floats(mids.as_slice(), &device)
            .reshape([1, s])
            .repeat_dim(0, n_rays);
        if randomized && self.cfg.randomized {
            t = t + Tensor::random([n_rays, s], Distribution::Uniform(-0.5, 0.5), &device)
                .mul_scalar(dt);
        }

        let positions = rays_o.reshape([n_rays, 1, 3])
            + rays_d.reshape([n_rays, 1, 3]) * t.clone().reshape([n_rays, s, 1]);
        let geo = geometry.forward(positions.reshape([n_rays * s, 3]));
        let density = geo.density.reshape([n_rays, s]);
        let rgb = geo.features.reshape([n_rays, s, 3]);

        let weights = composite_weights(density.mul_scalar(dt));
        let opacity = weights.clone().sum_dim(1);
        let depth = (weights.clone() * t).sum_dim(1);
        let comp = (weights.clone().reshape([n_rays, s, 1]) * rgb)
            .sum_dim(1)
            .reshape([n_rays, 3]);
        let bg = Tensor::<B, 1>::from_floats(background, &device).reshape([1, 3]);
        let comp_rgb = comp + bg * opacity.clone().neg().add_scalar(1.0);

        RenderOutput {
            comp_rgb,
            opacity,
            depth,
            weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use crate::config::GeometryConfig;
    use crate::data::{Camera, RayBatch};

    #[test]
    fn cumsum_is_exclusive() {
        let x = Tensor::<Cpu, 2>::from_floats([[1.0, 2.0, 3.0]], &Default::default());
        let got = exclusive_cumsum(x).into_data().to_vec::<f32>().unwrap();
        assert_eq!(got, vec![0.0, 1.0, 3.0]);
    }

    #[test]
    fn weights_match_cpu_reference_kernel() {
        let sd: Vec<f32> = (0..40).map(|i| ((i * 13) % 7) as f32 * 0.2).collect();
        let x =
            Tensor::<Cpu, 2>::from_data(TensorData::new(sd.clone(), [4, 10]), &Default::default());
        let got = composite_weights(x).into_data().to_vec::<f32>().unwrap();
        let want = threestudio_kernels::alpha_composite_weights(&sd, 10);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rendered_weights_sum_to_at_most_one() {
        let device = Default::default();
        let geo = ImplicitVolume::<Cpu>::new(&GeometryConfig::default(), &device);
        let renderer = NeRFVolumeRenderer::new(RendererConfig {
            num_samples_per_ray: 24,
            ..Default::default()
        });
        let batch = RayBatch::from_camera(Camera::look_at_origin(20.0, 30.0, 1.5, 60.0), 6, 6);
        let (o, d) = batch.to_tensors::<Cpu>(&device);
        let out = renderer.render(&geo, o, d, [1.0, 1.0, 1.0], true);
        assert_eq!(out.comp_rgb.dims(), [36, 3]);
        assert_eq!(out.weights.dims(), [36, 24]);
        let op = out.opacity.into_data().to_vec::<f32>().unwrap();
        assert!(op.iter().all(|x| *x >= 0.0 && *x <= 1.0 + 1e-5), "{op:?}");
        let w = out.weights.into_data().to_vec::<f32>().unwrap();
        assert!(w.iter().all(|x| *x >= 0.0));
        let rgb = out.comp_rgb.into_data().to_vec::<f32>().unwrap();
        assert!(rgb
            .iter()
            .all(|x| x.is_finite() && *x >= -1e-5 && *x <= 1.0 + 1e-5));
    }
}
